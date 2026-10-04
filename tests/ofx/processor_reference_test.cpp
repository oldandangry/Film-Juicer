#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "render_assertions.h"
#include "nlohmann/json.hpp"

#include "Hash.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "juicer_cuda_owner.h"
#include "SpectralProcessing.h"
#include "mainProcessing.h"
#include "ofxsSupportPrivate.h"

#if defined(JUICER_PREPARED_BOUNDARY_TEST)
#include "Cuda/JuicerCudaExecutor.h"
#include "../ffi/prepared_boundary.h"
#endif

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
#include <cstdarg>
#include <cstring>
#include "Cuda/JuicerCudaExecutor.h"

#include "RustAssetBridge.h"
#include "../ffi/juicer_test_api.h"

namespace JuicerAssets::NoiseTest {
    enum class EarlierFailure : unsigned char {
        None,
        Missing,
        Preflight,
        Focused,
        Print
    };
    struct Observation {
        std::array<unsigned, 3> operations{};
        EarlierFailure earlier = EarlierFailure::None;
        std::uint32_t sourceFailure = FJ_STATUS_SUCCESS;
        unsigned exceptionDrops = 0;
        bool exceptionReleasedAfterRecovery = false;
        bool closeBeforeAdmission = false;
        bool corrupt = false;
        bool gateViolation = false;
        bool noiseStep = false;
        bool noiseStepGated = false;
        bool recoveryFinished = false;
        bool releaseAfterRecovery = false;
        FjStatus result{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
    };
    thread_local Observation* observation = nullptr;
    class SourceError final : public std::exception {
    public:
        ~SourceError() override {
            if (observation && _lifetime.use_count() == 1) {
                ++observation->exceptionDrops;
                observation->gateViolation = observation->gateViolation || JuicerCuda::calling_thread_native_gate_active();
                observation->exceptionReleasedAfterRecovery = observation->recoveryFinished;
            }
        }
        const char* what() const noexcept override {
            return "noise source fixture standard exception";
        }

    private:
        // Count final exception storage, including platforms that copy a throw temporary.
        std::shared_ptr<const unsigned char> _lifetime = std::make_shared<const unsigned char>(0);
    };
    void observe(Operation operation) noexcept {
        if (observation) {
            ++observation->operations[static_cast<unsigned>(operation)];
            observation->gateViolation = observation->gateViolation || JuicerCuda::calling_thread_native_gate_active();
            if (operation == Operation::Release) {
                observation->releaseAfterRecovery = observation->recoveryFinished;
            }
        }
    }
    void view(FjStaticNoise& view) {
        if (observation && observation->closeBeforeAdmission) {
            if (fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr).category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("fixture close before admission failed");
            }
        }
        if (observation && observation->sourceFailure != FJ_STATUS_SUCCESS) {
            if (observation->sourceFailure == FJ_STATUS_ALLOCATION_FAILURE) {
                throw std::bad_alloc{};
            }
            if (observation->sourceFailure == FJ_STATUS_INTERNAL_FAILURE) {
                throw SourceError{};
            }
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, {observation->sourceFailure, FJ_API_NONE, 0}, "noise source fixture failure");
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        if (observation && observation->corrupt) {
            view.stbn.count = 1;
        }
    }
    void projection_step() noexcept {
        if (observation) {
            observation->noiseStep = true;
            observation->noiseStepGated = JuicerCuda::calling_thread_native_gate_active();
        }
    }
    void outcome(FjStatus status) noexcept {
        if (observation) {
            observation->result = status;
        }
    }
} // namespace JuicerAssets::NoiseTest

namespace JuicerCuda::ExecutorTest {
    enum class Event : std::uint8_t {
        Injected,
        Message,
        Classified,
        FrameAborted,
        RecoveryStarted,
        RecoveryEnded,
        FatalMapped
    };
    enum class Delivery : std::uint8_t {
        Success,
        Failure,
        Throw
    };
    struct FailureObservation {
        std::array<Event, 8> events{};
        std::size_t count = 0;
        const char* diagnostic = nullptr;
        FjStatus status{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        const char* stage = nullptr;
        Delivery delivery = Delivery::Success;
        JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult scanResult = JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Failed;
        bool overflow = false;
        bool failureStageMatches = false;
        bool failureStatusMatches = false;
        bool nativeRecoveryPending = false;
        bool adapterRecoveryPending = false;
        bool messageMatches = false;
        bool armed = true;
        FjStatus classifiedStatus{};
        bool failureDiagnosticMatches = false;

        void record(Event event) noexcept {
            if (count < events.size()) {
                events[count++] = event;
            } else {
                overflow = true;
            }
        }
    };
    thread_local FailureObservation* observation = nullptr;

    struct ResourceFailures {
        std::optional<Failure> stbn;
        std::optional<Failure> wang;
        cudaError_t eventCreate = cudaSuccess;
        cudaError_t eventRecord = cudaSuccess;
        int stbnAttempts = 0;
        int wangAttempts = 0;
        int injections = 0;

        void record_injection() noexcept {
            if (injections++ == 0 && observation) {
                observation->record(Event::Injected);
            }
        }
    };
    thread_local ResourceFailures* resourceFailures = nullptr;

    bool inject_grain_upload_failure(const char* label, Failure& failure) {
        if (!resourceFailures) {
            return false;
        }
        const std::optional<Failure>* injected = nullptr;
        if (std::strcmp(label, "STBN") == 0) {
            ++resourceFailures->stbnAttempts;
            injected = &resourceFailures->stbn;
        } else if (std::strcmp(label, "Wang.tiles") == 0) {
            ++resourceFailures->wangAttempts;
            injected = &resourceFailures->wang;
        }
        if (!injected || !injected->has_value()) {
            return false;
        }
        failure = **injected;
        resourceFailures->record_injection();
        return true;
    }

    bool inject_defect_event_create(cudaError_t& error) noexcept {
        if (!resourceFailures || resourceFailures->eventCreate == cudaSuccess) {
            return false;
        }
        error = resourceFailures->eventCreate;
        resourceFailures->eventCreate = cudaSuccess;
        resourceFailures->record_injection();
        return true;
    }

    bool inject_defect_event_record(cudaError_t& error) noexcept {
        if (!resourceFailures || resourceFailures->eventRecord == cudaSuccess) {
            return false;
        }
        error = resourceFailures->eventRecord;
        resourceFailures->eventRecord = cudaSuccess;
        resourceFailures->record_injection();
        return true;
    }

    bool inject_scan_error(Failure& failure, JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult& result) {
        if (!observation || !observation->armed) {
            return false;
        }
        observation->armed = false;
        failure.diagnostic = observation->diagnostic;
        failure.status = observation->status;
        result = observation->scanResult;
        observation->record(Event::Injected);
        return true;
    }
    void observe_classification(const char* stage, const Failure& failure, bool recoveryPending) noexcept {
        if (observation) {
            observation->failureStageMatches = std::strcmp(stage, observation->stage) == 0;
            observation->failureStatusMatches = failure.status.category == observation->status.category && failure.status.api == observation->status.api && failure.status.native_code == observation->status.native_code;
            observation->nativeRecoveryPending = recoveryPending;
            observation->classifiedStatus = failure.status;
            observation->failureDiagnosticMatches = failure.diagnostic == observation->diagnostic;
            observation->record(Event::Classified);
        }
    }
    void observe_frame_abort() noexcept {
        if (observation) {
            observation->record(Event::FrameAborted);
        }
    }
    void observe_recovery_start(bool pending) noexcept {
        if (observation) {
            observation->adapterRecoveryPending = pending;
            observation->record(Event::RecoveryStarted);
        }
    }
    void observe_recovery_end() noexcept {
        if (JuicerAssets::NoiseTest::observation) {
            JuicerAssets::NoiseTest::observation->recoveryFinished = true;
        }
        if (observation) {
            observation->record(Event::RecoveryEnded);
        }
    }
} // namespace JuicerCuda::ExecutorTest
#endif

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray&) {}

namespace {
    struct Properties {
        std::map<std::string, std::vector<std::string>> strings;
        std::map<std::string, std::vector<int>> ints;
        std::map<std::string, std::vector<double>> doubles;
        std::map<std::string, std::vector<void*>> pointers;
    };

    template <typename T>
    OfxStatus get_value(
        const std::map<std::string, std::vector<T>>& values,
        const char* name,
        int index,
        T* output) {
        if (!name || !output || index < 0) {
            return kOfxStatErrBadHandle;
        }
        const auto found = values.find(name);
        if (found == values.end() || static_cast<std::size_t>(index) >= found->second.size()) {
            std::cerr << "missing OFX property: " << name << "[" << index << "]\n";
            return kOfxStatErrUnknown;
        }
        *output = found->second[static_cast<std::size_t>(index)];
        return kOfxStatOK;
    }

    OfxStatus set_pointer(OfxPropertySetHandle handle, const char* name, int index, void* value) {
        if (!handle || !name || index < 0) {
            return kOfxStatErrBadHandle;
        }
        auto& items = reinterpret_cast<Properties*>(handle)->pointers[name];
        if (items.size() <= static_cast<std::size_t>(index)) {
            items.resize(static_cast<std::size_t>(index) + 1);
        }
        items[static_cast<std::size_t>(index)] = value;
        return kOfxStatOK;
    }

    OfxStatus get_pointer(OfxPropertySetHandle handle, const char* name, int index, void** output) {
        return handle
                   ? get_value(reinterpret_cast<Properties*>(handle)->pointers, name, index, output)
                   : kOfxStatErrBadHandle;
    }

    OfxStatus get_string(OfxPropertySetHandle handle, const char* name, int index, char** output) {
        if (!handle || !output) {
            return kOfxStatErrBadHandle;
        }
        std::string value;
        const OfxStatus status = get_value(reinterpret_cast<Properties*>(handle)->strings, name, index, &value);
        if (status != kOfxStatOK) {
            return status;
        }
        *output = const_cast<char*>(reinterpret_cast<Properties*>(handle)->strings[name][static_cast<std::size_t>(index)].c_str());
        return kOfxStatOK;
    }

    OfxStatus get_double(OfxPropertySetHandle handle, const char* name, int index, double* output) {
        return handle
                   ? get_value(reinterpret_cast<Properties*>(handle)->doubles, name, index, output)
                   : kOfxStatErrBadHandle;
    }

    OfxStatus get_int(OfxPropertySetHandle handle, const char* name, int index, int* output) {
        return handle
                   ? get_value(reinterpret_cast<Properties*>(handle)->ints, name, index, output)
                   : kOfxStatErrBadHandle;
    }

    OfxStatus effect_properties(OfxImageEffectHandle handle, OfxPropertySetHandle* output) {
        *output = reinterpret_cast<OfxPropertySetHandle>(handle);
        return kOfxStatOK;
    }

    OfxStatus effect_parameters(OfxImageEffectHandle handle, OfxParamSetHandle* output) {
        *output = reinterpret_cast<OfxParamSetHandle>(handle);
        return kOfxStatOK;
    }

    OfxStatus parameter_properties(OfxParamSetHandle handle, OfxPropertySetHandle* output) {
        *output = reinterpret_cast<OfxPropertySetHandle>(handle);
        return kOfxStatOK;
    }

    class ImageLeaseAudit final {
    public:
        void acquire(OfxPropertySetHandle handle, std::string label) {
            auto [position, inserted] =
                _leases.emplace(handle, Lease{std::move(label), 1, 0});
            if (!inserted) {
                position->second.acquisitions += 1;
                set_error(Error::DuplicateAcquisition);
            }
        }

        void release(OfxPropertySetHandle handle) noexcept {
            const auto found = _leases.find(handle);
            if (found == _leases.end()) {
                set_error(Error::UnknownRelease);
                return;
            }
            Lease& lease = found->second;
            lease.releases += 1;
            if (lease.releases > lease.acquisitions) {
                set_error(Error::DuplicateRelease);
            }
        }

        void require_complete(const char* caseName) const {
            if (_error != Error::None) {
                throw std::runtime_error(
                    std::string(caseName) + ": " + error_message());
            }
            if (_leases.size() != 2) {
                throw std::runtime_error(
                    std::string(caseName) +
                    ": expected two acquired image handles");
            }
            for (const auto& [handle, lease] : _leases) {
                (void)handle;
                if (lease.acquisitions != 1 || lease.releases != 1) {
                    throw std::runtime_error(
                        std::string(caseName) +
                        ": image lease was not released exactly once for " +
                        lease.label);
                }
            }
        }

    private:
        enum class Error : std::uint8_t {
            None,
            DuplicateAcquisition,
            UnknownRelease,
            DuplicateRelease
        };

        struct Lease {
            std::string label;
            int acquisitions = 0;
            int releases = 0;
        };

        void set_error(Error error) noexcept {
            if (_error == Error::None) {
                _error = error;
            }
        }

        const char* error_message() const noexcept {
            switch (_error) {
                case Error::DuplicateAcquisition:
                    return "duplicate image acquisition";
                case Error::UnknownRelease:
                    return "unknown image release";
                case Error::DuplicateRelease:
                    return "duplicate image release";
                case Error::None:
                    break;
            }
            return "image lease audit failed";
        }

        std::map<OfxPropertySetHandle, Lease> _leases;
        Error _error = Error::None;
    };

    thread_local ImageLeaseAudit* s_activeImageLeaseAudit = nullptr;

    class ScopedImageLeaseAudit final {
    public:
        explicit ScopedImageLeaseAudit(ImageLeaseAudit& audit) noexcept {
            s_activeImageLeaseAudit = &audit;
        }

        ~ScopedImageLeaseAudit() {
            s_activeImageLeaseAudit = nullptr;
        }

        ScopedImageLeaseAudit(const ScopedImageLeaseAudit&) = delete;
        ScopedImageLeaseAudit& operator=(const ScopedImageLeaseAudit&) = delete;
    };

    OfxStatus release_image(OfxPropertySetHandle handle) {
        if (!s_activeImageLeaseAudit) {
            return kOfxStatErrBadHandle;
        }
        s_activeImageLeaseAudit->release(handle);
        return kOfxStatOK;
    }

    struct AbortObservation {
        std::thread::id thread = std::this_thread::get_id();
        unsigned calls = 0;
        unsigned cancelAt = 0;
        unsigned throwAt = 0;
        bool sameThread = true;
    };
    thread_local AbortObservation* s_abortObservation = nullptr;

    int effect_abort(OfxImageEffectHandle) {
        if (!s_abortObservation) {
            return 0;
        }
        auto& observation = *s_abortObservation;
        ++observation.calls;
        observation.sameThread = observation.sameThread && observation.thread == std::this_thread::get_id();
        if (observation.calls == observation.throwAt) {
            throw std::runtime_error("abort suite exception");
        }
        return observation.calls == observation.cancelAt ? 1 : 0;
    }

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters) Required OFX message-suite callback signature.
    OfxStatus failure_message(void*, const char* type, const char* id, const char* format, ...) {
        auto& observation = *JuicerCuda::ExecutorTest::observation;
        // Acquiring the real non-reentrant gate proves the adapter released it.
        JuicerCuda::NativeCall call(JuicerCuda::borrowed_owner());
        observation.record(JuicerCuda::ExecutorTest::Event::Message);
        std::vector<char> text(std::strlen(observation.diagnostic) + 1);
        va_list args;
        va_start(args, format);
        const int length = std::vsnprintf(text.data(), text.size(), format, args);
        va_end(args);
        observation.messageMatches = length >= 0 &&
                                     static_cast<std::size_t>(length) < text.size() &&
                                     std::strcmp(type, kOfxMessageError) == 0 &&
                                     std::strcmp(id, "FilmJuicerDeferredCudaFailure") == 0 &&
                                     std::strcmp(text.data(), observation.diagnostic) == 0;
        if (observation.delivery == JuicerCuda::ExecutorTest::Delivery::Throw) {
            throw std::runtime_error("injected host message exception");
        }
        return observation.delivery == JuicerCuda::ExecutorTest::Delivery::Failure
                   ? kOfxStatFailed
                   : kOfxStatOK;
    }
#endif

    class NarrowHost final {
    public:
        NarrowHost() {
            _properties.propSetPointer = set_pointer;
            _properties.propGetPointer = get_pointer;
            _properties.propGetString = get_string;
            _properties.propGetDouble = get_double;
            _properties.propGetInt = get_int;
            _effect.getPropertySet = effect_properties;
            _effect.getParamSet = effect_parameters;
            _effect.clipReleaseImage = release_image;
            _effect.abort = effect_abort;
            _parameters.paramSetGetPropertySet = parameter_properties;
            OFX::Private::gPropSuite = &_properties;
            OFX::Private::gEffectSuite = &_effect;
            OFX::Private::gParamSuite = &_parameters;
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
            _messages.message = failure_message;
            OFX::Private::gMessageSuite = &_messages;
#endif
        }

        ~NarrowHost() {
            OFX::Private::gPropSuite = nullptr;
            OFX::Private::gEffectSuite = nullptr;
            OFX::Private::gParamSuite = nullptr;
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
            OFX::Private::gMessageSuite = nullptr;
#endif
        }

    private:
        OfxPropertySuiteV1 _properties{};
        OfxImageEffectSuiteV1 _effect{};
        OfxParameterSuiteV1 _parameters{};
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
        OfxMessageSuiteV1 _messages{};
#endif
    };

    class NarrowEffect final : public OFX::ImageEffect {
    public:
        explicit NarrowEffect(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {}

        void render(const OFX::RenderArguments&) override {}
    };

    struct ImagePropertiesInput {
        OfxRectI bounds{};
        int components = 0;
        int rowBytes = 0;
    };

    void fill_image_properties(Properties& values, void* deviceData, const ImagePropertiesInput& input) {
        const std::array<int, 4> bounds{{input.bounds.x1, input.bounds.y1, input.bounds.x2, input.bounds.y2}};
        values.ints[kOfxImagePropRowBytes] = {input.rowBytes};
        values.ints[kOfxImagePropBounds] = std::vector<int>(bounds.begin(), bounds.end());
        values.ints[kOfxImagePropRegionOfDefinition] = std::vector<int>(bounds.begin(), bounds.end());
        values.doubles[kOfxImagePropPixelAspectRatio] = {1.0};
        values.doubles[kOfxImageEffectPropRenderScale] = {1.0, 1.0};
        values.strings[kOfxImageEffectPropComponents] = {
            input.components == 4 ? kOfxImageComponentRGBA : kOfxImageComponentRGB};
        values.strings[kOfxImageEffectPropPixelDepth] = {kOfxBitDepthFloat};
        values.strings[kOfxImageEffectPropPreMultiplication] = {kOfxImageOpaque};
        values.strings[kOfxImagePropField] = {kOfxImageFieldNone};
        values.strings[kOfxImagePropUniqueIdentifier] = {"pre-rust-processor-input"};
        values.pointers[kOfxImagePropData] = {deviceData};
    }

    void require_cuda(cudaError_t status, const char* action) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
        }
    }

    constexpr int kWidth = 7;
    constexpr int kHeight = 5;
    constexpr float kCanary = -8.0f;

    struct Case {
        const char* name = nullptr;
        Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
        int reconstruction = 0;
        int components = 0;
        int originX = 0;
        int originY = 0;
        bool combined = false;
        bool signedZeroGlare = false;
        bool negativeZero = false;
        bool uniform = false;
        int width = kWidth;
        int height = kHeight;
        double time = 37.0;
        double scaleX = 1.0;
        double scaleY = 1.0;
        const std::vector<float>* sourcePixels = nullptr;
        int fullWidth = 0;
        int fullHeight = 0;
    };

    struct DeviceFrame {
        float* source = nullptr;
        float* destination = nullptr;
        cudaStream_t stream = nullptr;
        bool destinationInSource = false;

        ~DeviceFrame() {
            if (stream) {
                (void)cudaStreamSynchronize(stream);
            }
            if (destination && !destinationInSource) {
                (void)cudaFree(destination);
            }
            if (source) {
                (void)cudaFree(source);
            }
            if (stream) {
                (void)cudaStreamDestroy(stream);
            }
        }
    };

    ParamSnapshot parameters_for(const Case& test) {
        ParamSnapshot parameters;
        parameters.scanRoute = test.route;
        if (Spektrafilm::scan_route_metadata(test.route).capturePolarity ==
            Spektrafilm::ProfilePolarity::Positive) {
            parameters.filmProfileKey = "fujifilm_provia_100f";
        }
        parameters.spectralUpsamplingMode = test.reconstruction;
        parameters.cameraAutoExposureEnabled = 0;
        if (!test.combined) {
            parameters.scannerUnsharpMask = {0.0, 0.0};
        }
        parameters.gateWeaveAmount = 0.0;
        if (test.combined) {
            parameters.cameraDiffusion.active = true;
            parameters.grainControls.active = true;
            std::string diagnostic;
            if (!Spektrafilm::build_scatter_halation_controls(
                    ScatterHalationRawControls{true, 1.0, 1.0, 1.0, 1.0},
                    parameters.scatterHalationControls,
                    diagnostic)) {
                throw std::runtime_error(diagnostic);
            }
        }
        if (test.signedZeroGlare) {
            parameters.glareActive = true;
            parameters.glarePercent = 0.3;
            parameters.cameraExposureCompensationEv = test.negativeZero ? -0.0 : 0.0;
        }
        return parameters;
    }

    enum class ExecutionPath : std::uint8_t {
        Processor,
        Direct,
        Boundary,
        Contract
    };

    enum class DestinationLayout : std::uint8_t {
        Matching,
        Offset,
        Uncovered,
        DisjointRows
    };

    std::vector<float> render_case(const Case& test, const ParamSnapshot& parameters, InstanceState& state, bool emptyWindow = false, ExecutionPath path = ExecutionPath::Processor, DestinationLayout destinationLayout = DestinationLayout::Matching, [[maybe_unused]] std::uint64_t snapshotId = 1) {
        const int width = test.width;
        const int height = test.height;
        const int pitch = width * test.components + 5;
        const std::size_t bytes = static_cast<std::size_t>(pitch * height) * sizeof(float);
        std::vector<float> input(static_cast<std::size_t>(pitch * height), -7.0f);
        const int border = destinationLayout == DestinationLayout::Offset ? 1 : 0;
        const int destinationPitch = pitch + 2 * border * test.components;
        const int destinationHeight = height + 2 * border;
        const std::size_t destinationBytes = static_cast<std::size_t>(destinationPitch * destinationHeight) * sizeof(float);
        std::vector<float> output(static_cast<std::size_t>(destinationPitch * destinationHeight), kCanary);
        std::vector<float> initialDestination(output);
        for (int y = 0; y < height; ++y) {
            const std::size_t rowOffset = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixelOffset = rowOffset +
                                                static_cast<std::size_t>(x) * static_cast<std::size_t>(test.components);
                for (int c = 0; c < test.components; ++c) {
                    input[pixelOffset + static_cast<std::size_t>(c)] =
                        c == 3 ? 0.125f + static_cast<float>((x + y) % 4) * 0.25f
                               : static_cast<float>(((test.uniform ? 0 : x + 3 * y) + 7 * c) % 19) / 19.0f;
                }
            }
        }
        if (test.sourcePixels) {
            const std::size_t rowSamples = static_cast<std::size_t>(width) * static_cast<std::size_t>(test.components);
            if (test.sourcePixels->size() != rowSamples * static_cast<std::size_t>(height)) {
                throw std::runtime_error("accepted capture input shape mismatch");
            }
            for (int y = 0; y < height; ++y) {
                const std::size_t sourceOffset = static_cast<std::size_t>(y) * rowSamples;
                const std::size_t destinationOffset = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch);
                std::copy_n(test.sourcePixels->data() + sourceOffset, rowSamples, input.data() + destinationOffset);
            }
        }
        DeviceFrame device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create stream");
        device.destinationInSource = destinationLayout == DestinationLayout::DisjointRows;
        require_cuda(cudaMalloc(&device.source, bytes + (device.destinationInSource ? destinationBytes : 0)), "allocate source");
        if (device.destinationInSource) {
            device.destination = device.source + bytes / sizeof(float);
        } else {
            require_cuda(cudaMalloc(&device.destination, destinationBytes), "allocate destination");
        }
        require_cuda(cudaMemcpy(device.source, input.data(), bytes, cudaMemcpyHostToDevice), "copy source");
        require_cuda(cudaMemcpy(device.destination, initialDestination.data(), destinationBytes, cudaMemcpyHostToDevice), "copy destination");

        Properties effectProperties;
        effectProperties.strings[kOfxImageEffectPropContext] = {kOfxImageEffectContextFilter};
        Properties sourceProperties;
        Properties destinationProperties;
        const OfxRectI bounds{test.originX, test.originY, test.originX + width, test.originY + height};
        const ImagePropertiesInput imageInput{bounds, test.components, pitch * static_cast<int>(sizeof(float))};
        fill_image_properties(sourceProperties, device.source, imageInput);
        const OfxRectI destinationBounds{bounds.x1 - border, bounds.y1 - border, bounds.x2 + border - (destinationLayout == DestinationLayout::Uncovered ? 1 : 0), bounds.y2 + border};
        fill_image_properties(destinationProperties, device.destination, {destinationBounds, test.components, destinationPitch * static_cast<int>(sizeof(float))});
        if (test.fullWidth > 0 && test.fullHeight > 0) {
            sourceProperties.ints[kOfxImagePropRegionOfDefinition] = {0, 0, test.fullWidth, test.fullHeight};
            destinationProperties.ints[kOfxImagePropRegionOfDefinition] = {0, 0, test.fullWidth, test.fullHeight};
        }
        sourceProperties.doubles[kOfxImageEffectPropRenderScale] = {test.scaleX, test.scaleY};
        destinationProperties.doubles[kOfxImageEffectPropRenderScale] = {test.scaleX, test.scaleY};
        const OfxPropertySetHandle sourceHandle =
            reinterpret_cast<OfxPropertySetHandle>(&sourceProperties);
        const OfxPropertySetHandle destinationHandle =
            reinterpret_cast<OfxPropertySetHandle>(&destinationProperties);
        ImageLeaseAudit imageLeaseAudit;
        imageLeaseAudit.acquire(sourceHandle, "source");
        imageLeaseAudit.acquire(destinationHandle, "destination");
        {
            ScopedImageLeaseAudit activeImageLeaseAudit(imageLeaseAudit);
            NarrowEffect effect(reinterpret_cast<OfxImageEffectHandle>(&effectProperties));
            OFX::Image sourceImage(sourceHandle);
            OFX::Image destinationImage(destinationHandle);
            {
                std::lock_guard<std::mutex> lock(state.pending.m);
                state.pending.value = PendingParamsState::Valid{parameters, hash_params(parameters)};
            }
            PendingRenderAdmissionResult admitted = admit_pending_render_state(state);
#if defined(JUICER_NOISE_TEST_HOOK)
            if (JuicerAssets::NoiseTest::observation) {
                const auto earlier = JuicerAssets::NoiseTest::observation->earlier;
                const auto change = [&](auto& published) {
                    if (!published || earlier == JuicerAssets::NoiseTest::EarlierFailure::None || earlier == JuicerAssets::NoiseTest::EarlierFailure::Missing) {
                        return;
                    }
                    auto copy = std::make_shared<std::remove_const_t<typename std::remove_reference_t<decltype(published)>::element_type>>(*published);
                    if (earlier == JuicerAssets::NoiseTest::EarlierFailure::Preflight) {
                        copy->recipe.spatialOptics.cameraLensBlur.sigmaUm = 1;
                        copy->recipe.spatialOptics.cameraLensBlur.hash = 1;
                    }
                    if (earlier == JuicerAssets::NoiseTest::EarlierFailure::Focused) {
                        copy->recipe.filmRaw.rgbToRawMethod = Spektrafilm::RgbToRawMethod::Mallett2019;
                    }
                    if (earlier == JuicerAssets::NoiseTest::EarlierFailure::Print) {
                        copy->recipe.print.develop.densityCurves.clear();
                    }
                    published = std::move(copy);
                };
                change(admitted.directState);
                change(admitted.printState);
            }
#endif
            const bool print = Spektrafilm::scan_route_is_print(test.route);
            if (admitted.status != (print ? PendingRenderAdmissionStatus::AdmittedPrint
                                          : PendingRenderAdmissionStatus::AdmittedDirect)) {
                throw std::runtime_error(std::string(test.name) + ": admission failed: " + admitted.diagnostic);
            }
            if ((print && !admitted.printState) || (!print && !admitted.directState)) {
                throw std::runtime_error("admitted fixture state unavailable");
            }
            JuicerProcessor processor(effect);
            OFX::RenderArguments args{};
            args.isEnabledCudaRender = true;
            args.pCudaStream = device.stream;
            args.cudaStreamPropertyPresent = true;
            processor.setGPURenderArgs(args);
            processor.setSrcDst({&sourceImage, &destinationImage});
            const OfxRectI renderWindow = emptyWindow
                                              ? OfxRectI{test.originX, test.originY, test.originX, test.originY}
                                              : bounds;
            const OfxRectI fullBounds = test.fullWidth > 0 && test.fullHeight > 0
                                            ? OfxRectI{0, 0, test.fullWidth, test.fullHeight}
                                            : bounds;
            const int physicalWidth = test.fullWidth > 0 ? test.fullWidth : width;
            const float pixelSizeUm = 35'000.0f / static_cast<float>(physicalWidth);
            const Spektrafilm::FilmJuicerEffectsGeometry effectsGeometry{
                {fullBounds.x1, fullBounds.y1, fullBounds.x2 - fullBounds.x1, fullBounds.y2 - fullBounds.y1},
                static_cast<double>(fullBounds.x1),
                static_cast<double>(fullBounds.y1),
                static_cast<double>(fullBounds.x2 - fullBounds.x1),
                static_cast<double>(fullBounds.y2 - fullBounds.y1),
                test.scaleX,
                test.scaleY,
                1.0};
            std::optional<ScatterHalationFrameDescriptor> halation;
            std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
            const auto& recipe = print ? admitted.printState->recipe : admitted.directState->recipe;
            std::string descriptorDiagnostic;
            if (test.combined && (!Spektrafilm::build_scatter_halation_frame_descriptor(
                                      recipe.spatialOptics.scatterHalation, pixelSizeUm, halation, descriptorDiagnostic) ||
                                  !halation)) {
                throw std::runtime_error("halation descriptor: " + descriptorDiagnostic);
            }
            if (parameters.cameraDiffusion.active || parameters.enlargerDiffusion.active) {
                if (!Spektrafilm::build_diffusion_frame_set_descriptor(
                        recipe.spatialOptics, test.route, pixelSizeUm, Spektrafilm::DiffusionFrameDomain{test.originX, test.originY, width, height}, diffusion, descriptorDiagnostic) ||
                    !diffusion) {
                    throw std::runtime_error("diffusion descriptor: " + descriptorDiagnostic);
                }
            }
            if (print) {
                JuicerProcessor::PrintFrameRequest request;
                request.state = admitted.printState;
#if defined(JUICER_NOISE_TEST_HOOK)
                if (JuicerAssets::NoiseTest::observation && JuicerAssets::NoiseTest::observation->earlier == JuicerAssets::NoiseTest::EarlierFailure::Missing) {
                    request.state.reset();
                }
#endif
                request.diffusionFrameSet = diffusion;
                request.scatterHalation = halation;
                request.renderWindow = renderWindow;
                request.fullFrameExtent = fullBounds;
                request.sessionSeed = 0x20260923;
                request.instanceToken = 0x641207 + static_cast<int>(test.route) + (test.combined ? 4 : 0);
                request.clipToken = 0x5312;
                request.frameTime = test.time;
                request.frameRate = 24.0;
                request.pixelSizeUm = pixelSizeUm;
                request.effectsGeometry = effectsGeometry;
                processor.setPrintFrameRequest(request);
            } else {
                JuicerProcessor::DirectFrameRequest request;
                request.state = admitted.directState;
#if defined(JUICER_NOISE_TEST_HOOK)
                if (JuicerAssets::NoiseTest::observation && JuicerAssets::NoiseTest::observation->earlier == JuicerAssets::NoiseTest::EarlierFailure::Missing) {
                    request.state.reset();
                }
#endif
                request.diffusionFrameSet = diffusion;
                request.scatterHalation = halation;
                request.renderWindow = renderWindow;
                request.fullFrameExtent = fullBounds;
                request.sessionSeed = 0x20260923;
                request.instanceToken = 0x641207 + static_cast<int>(test.route) + (test.combined ? 4 : 0);
                request.clipToken = 0x5312;
                request.frameTime = test.time;
                request.frameRate = 24.0;
                request.pixelSizeUm = pixelSizeUm;
                request.effectsGeometry = effectsGeometry;
                processor.setDirectFrameRequest(request);
            }
            processor.setInstanceState(&state);
            JuicerProcess::Root::FramePreparationToken preparation;
#if defined(JUICER_NOISE_TEST_HOOK)
            const bool testingClosedAdmission = JuicerAssets::NoiseTest::observation && JuicerAssets::NoiseTest::observation->closeBeforeAdmission;
#else
            constexpr bool testingClosedAdmission = false;
#endif
            if (!testingClosedAdmission) {
                preparation = JuicerProcess::root().begin_frame_preparation();
                if (!preparation.active()) {
                    throw std::runtime_error("frame preparation guard unavailable");
                }
            }
#if defined(JUICER_PREPARED_BOUNDARY_TEST) || defined(JUICER_NOISE_TEST_HOOK)
            if (path != ExecutionPath::Processor) {
                const auto& payload = print ? admitted.printState->payload : admitted.directState->payload;
                const JuicerCuda::FrameRect nativeBounds{bounds.x1, bounds.y1, bounds.x2, bounds.y2};
                const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(nativeBounds, nativeBounds, recipe.filmRaw.autoExposureMethod);
                const Spektrafilm::FilmJuicerEffectsGeometry geometry{};
                const JuicerCuda::ExecutionFrame frame{nativeBounds, nativeBounds, nativeBounds, reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<unsigned char*>(device.destination), imageInput.rowBytes, imageInput.rowBytes, test.components, device.stream, diffusion, halation, geometry, pixelSizeUm, test.time, 24.0, 0x20260923, 0x5312, meter, false, false};
                JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
                snapshot.instanceToken.value = 0x641207 + static_cast<int>(test.route) + (test.combined ? 4 : 0);
                snapshot.frameToken.value = static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(test.time)));
                snapshot.snapshotId = snapshotId;
                const FjImage rawSource{reinterpret_cast<std::uintptr_t>(device.source), {bounds.x1, bounds.y1, bounds.x2, bounds.y2}, imageInput.rowBytes, static_cast<std::uint32_t>(test.components), FJ_DEPTH_FLOAT32};
                FjFrame raw{};
                raw.source = rawSource;
                raw.destination = rawSource;
                raw.destination.address = reinterpret_cast<std::uintptr_t>(device.destination);
                raw.render_window = rawSource.bounds;
                raw.full_frame_extent = rawSource.bounds;
                raw.stream = reinterpret_cast<std::uintptr_t>(device.stream);
                raw.flags = FJ_FRAME_STREAM_PRESENT;
                FjCudaContext context{};
                if (fj_cuda_inspect(JuicerCuda::borrowed_owner(), &raw, &context, nullptr).category != FJ_STATUS_SUCCESS) {
                    throw std::runtime_error("reference frame inspection failed");
                }
                snapshot.deviceContextKey = {context.device_id, reinterpret_cast<void*>(context.context)};
                snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                    payload.uploadCoreHash, recipe.dirCouplers.hash, payload.scannerHash, meter.hash);
                if (path == ExecutionPath::Boundary || path == ExecutionPath::Contract) {
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
                    const auto descriptors = JuicerCuda::describe_execution(recipe, payload, frame);
                    std::string diagnostic;
                    if (!JuicerCudaTest::execute_boundary(recipe, payload, frame, snapshot, descriptors, diagnostic, path == ExecutionPath::Contract)) {
                        throw std::runtime_error(std::string(test.name) + ": C boundary: " + diagnostic);
                    }
#else
                    throw std::runtime_error("boundary path requires the prepared fixture target");
#endif
                } else {
                    JuicerCuda::PendingContextLossRecovery recovery;
                    if (print) {
                        JuicerCuda::execute_print({recipe, payload, frame, snapshot}, recovery);
                    } else {
                        JuicerCuda::execute_direct({recipe, payload, frame, snapshot}, recovery);
                    }
                }
            } else {
                processor.process();
            }
#else
            (void)path;
            processor.process();
#endif
        }
        imageLeaseAudit.require_complete(test.name);
        if (s_abortObservation && s_abortObservation->cancelAt != 0 &&
            s_abortObservation->calls >= s_abortObservation->cancelAt) {
            return {};
        }
        require_cuda(cudaMemcpyAsync(output.data(), device.destination, destinationBytes, cudaMemcpyDeviceToHost, device.stream),
                     "copy destination");
        require_cuda(cudaStreamSynchronize(device.stream), "synchronize");
        if (emptyWindow) {
            for (float value : output) {
                if (std::bit_cast<std::uint32_t>(value) != std::bit_cast<std::uint32_t>(kCanary)) {
                    throw std::runtime_error("empty processor window wrote output");
                }
            }
            return {};
        }
        std::vector<float> pixels;
        pixels.reserve(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                       static_cast<std::size_t>(test.components));
        for (int y = 0; y < height; ++y) {
            const std::size_t rowOffset = static_cast<std::size_t>(y + border) * static_cast<std::size_t>(destinationPitch);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixelOffset = rowOffset +
                                                static_cast<std::size_t>(x + border) * static_cast<std::size_t>(test.components);
                for (int c = 0; c < test.components; ++c) {
                    const std::size_t index = pixelOffset + static_cast<std::size_t>(c);
                    const float value = output[index];
                    if (!std::isfinite(value)) {
                        throw std::runtime_error(std::string(test.name) + ": nonfinite output");
                    }
                    if (c == 3) {
                        RenderAssertions::require_same_bits(std::string(test.name) + ": alpha", value, input[static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) + static_cast<std::size_t>(x) * static_cast<std::size_t>(test.components) + static_cast<std::size_t>(c)]);
                    }
                    pixels.push_back(value);
                }
            }
        }
        for (int y = 0; y < destinationHeight; ++y) {
            for (int x = 0; x < destinationPitch; ++x) {
                const bool rendered = y >= border && y < border + height &&
                                      x >= border * test.components && x < (border + width) * test.components;
                if (!rendered) {
                    RenderAssertions::require_same_bits(std::string(test.name) + ": destination outside render window", output[static_cast<std::size_t>(y) * static_cast<std::size_t>(destinationPitch) + static_cast<std::size_t>(x)], kCanary);
                }
            }
        }
        return pixels;
    }

    using RenderAssertions::compare_pixels;

} // namespace

namespace {
    [[maybe_unused]] void check_cutover_callbacks(const Case& test, const std::vector<float>& expected) {
        InstanceState state;
        compare_pixels("same allocation disjoint rows", render_case(test, parameters_for(test), state, false, ExecutionPath::Processor, DestinationLayout::DisjointRows), expected);
        Case uniform = test;
        uniform.uniform = true;
        auto uniformParameters = parameters_for(uniform);
        uniformParameters.glareActive = false;
        const auto initialUniform = render_case(uniform, uniformParameters, state);
        // This is a constant-input invariance test, complementary to the
        // independent accepted pixel fixtures above. Glare would add noise.
        const auto uniform_expected = [&](const Case& frame) {
            std::vector<float> pixels(static_cast<std::size_t>(frame.width * frame.height * frame.components));
            for (std::size_t i = 0; i < pixels.size(); ++i) {
                const auto channel = i % static_cast<std::size_t>(frame.components);
                // Alpha is positional even for uniform RGB in this fixture.
                const auto pixel = i / static_cast<std::size_t>(frame.components);
                const auto x = pixel % static_cast<std::size_t>(frame.width);
                const auto y = pixel / static_cast<std::size_t>(frame.width);
                pixels[i] = channel == 3 ? 0.125f + static_cast<float>((x + y) % 4) * 0.25f : initialUniform[channel];
            }
            return pixels;
        };
        compare_pixels("initial uniform callback", initialUniform, uniform_expected(uniform));
        const auto firstId = state.submissionSnapshotLatch.snapshotId;
        uniform.width += 2;
        uniform.height -= 2;
        uniform.originX = -11;
        uniform.originY = 7;
        uniform.time = -17.25;
        uniform.scaleX = 0.5;
        uniform.scaleY = 1.5;
        const auto changedExpected = uniform_expected(uniform);
        compare_pixels("changed geometry time and scale", render_case(uniform, uniformParameters, state), changedExpected);
        const auto changedId = state.submissionSnapshotLatch.snapshotId;
        if (changedId == firstId || state.submissionSnapshotLatch.frameToken.value != static_cast<std::uint64_t>(std::int64_t{-18})) {
            throw std::runtime_error("changed callback facts reused the previous latch");
        }
        // Each render_case allocates a new supplied stream; the same facts reuse the latch.
        compare_pixels("sequential supplied stream", render_case(uniform, uniformParameters, state), changedExpected);
        if (state.submissionSnapshotLatch.snapshotId != changedId) {
            throw std::runtime_error("unchanged callback facts failed to reuse the latch");
        }
        CUcontext current = nullptr;
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS) {
            throw std::runtime_error("get callback context");
        }
        std::exception_ptr workerFailure;
        std::thread worker([&] {
            try {
                if (cuCtxSetCurrent(current) != CUDA_SUCCESS) {
                    throw std::runtime_error("set worker callback context");
                }
                compare_pixels("callback on another host thread", render_case(uniform, uniformParameters, state), changedExpected);
            } catch (...) {
                workerFailure = std::current_exception();
            }
            (void)cuCtxSetCurrent(nullptr);
        });
        worker.join();
        if (workerFailure) {
            std::rethrow_exception(workerFailure);
        }
        CUcontext other = nullptr;
        if (cuCtxCreate(&other, nullptr, 0, 0) != CUDA_SUCCESS) {
            throw std::runtime_error("create fixture context");
        }
        try {
            compare_pixels("second context on same device", render_case(uniform, uniformParameters, state), changedExpected);
            if (state.submissionSnapshotLatch.deviceContextKey.contextOpaque != other) {
                throw std::runtime_error("callback retained prior context identity");
            }
            std::string diagnostic;
            JuicerCuda::NativeCall admission(JuicerCuda::borrowed_owner());
            if (!JuicerProcess::root().retire_idle_context(0, other, diagnostic)) {
                throw std::runtime_error(diagnostic);
            }
        } catch (...) {
            (void)cuCtxDestroy(other);
            (void)cuCtxSetCurrent(current);
            throw;
        }
        if (cuCtxDestroy(other) != CUDA_SUCCESS || cuCtxSetCurrent(current) != CUDA_SUCCESS) {
            throw std::runtime_error("release fixture context");
        }
        for (const unsigned checkpoint : {1U, 3U, 4U, 5U}) {
            AbortObservation observation;
            observation.cancelAt = checkpoint;
            s_abortObservation = &observation;
            try {
                (void)render_case(test, parameters_for(test), state);
            } catch (...) {
                s_abortObservation = nullptr;
                throw;
            }
            s_abortObservation = nullptr;
            if (observation.calls != checkpoint || !observation.sameThread) {
                throw std::runtime_error("production abort delivery or callback lifetime changed");
            }
        }
        AbortObservation throwing;
        throwing.throwAt = 4;
        s_abortObservation = &throwing;
        bool fatal = false;
        try {
            (void)render_case(test, parameters_for(test), state);
        } catch (const OFX::Exception::Suite& error) {
            fatal = error.status() == kOfxStatErrFatal;
        } catch (...) {
            s_abortObservation = nullptr;
            throw;
        }
        s_abortObservation = nullptr;
        if (!fatal || throwing.calls != 4 || !throwing.sameThread) {
            throw std::runtime_error("abort suite exception escaped or became successful cancellation");
        }
        compare_pixels("render after cancellation", render_case(test, parameters_for(test), state), expected);
        // This fixture's next case reuses its deterministic instance token with
        // a fresh counter. Match real instance teardown before that reuse.
        if (fj_cuda_retire_instance(JuicerCuda::borrowed_owner(), state.submissionSnapshotLatch.instanceToken.value, nullptr).category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error("retire cutover fixture instance");
        }
        std::cout << test.name << ": production geometry/time/scale, streams, threads, contexts, disjoint alias and abort contracts passed\n";
    }
} // namespace

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
namespace {
    void check_failure_order(const Case& test,
                             JuicerCuda::ExecutorTest::Delivery delivery,
                             bool deferredDirFailure,
                             FjStatus failureStatus,
                             const char* wording) {
        const bool contextLoss = failureStatus.category == FJ_STATUS_CONTEXT_LOSS;
        using JuicerCuda::ExecutorTest::Event;
        JuicerCuda::ExecutorTest::FailureObservation observation;
        observation.delivery = delivery;
        observation.status = failureStatus;
        observation.diagnostic = wording;
        observation.scanResult = deferredDirFailure
                                     ? JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::DeferredDirFailure
                                     : JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Failed;
        observation.stage = Spektrafilm::scan_route_is_print(test.route)
                                ? "print_scan_error_stage"
                                : "direct_scan_error_stage";
        JuicerCuda::ExecutorTest::observation = &observation;
        InstanceState state;
        bool fatal = false;
        try {
            (void)render_case(test, parameters_for(test), state);
        } catch (const OFX::Exception::Suite& error) {
            fatal = error.status() == kOfxStatErrFatal;
            observation.record(Event::FatalMapped);
        } catch (...) {
            JuicerCuda::ExecutorTest::observation = nullptr;
            throw;
        }
        JuicerCuda::ExecutorTest::observation = nullptr;
        std::vector<Event> expected{Event::Injected, Event::Classified, Event::FrameAborted};
        if (deferredDirFailure) {
            expected.push_back(Event::Message);
        }
        expected.insert(expected.end(), {Event::RecoveryStarted, Event::RecoveryEnded, Event::FatalMapped});
        if (!fatal || observation.overflow || observation.count != expected.size() ||
            !std::equal(expected.begin(), expected.end(), observation.events.begin()) ||
            !observation.failureStageMatches || !observation.failureStatusMatches ||
            observation.nativeRecoveryPending != contextLoss ||
            observation.adapterRecoveryPending != contextLoss ||
            (deferredDirFailure && !observation.messageMatches) ||
            state.submissionSnapshotLatchValid == contextLoss) {
            throw std::runtime_error(std::string(test.name) + ": executor failure order or host mapping changed");
        }
        std::cout << test.name << " dir=" << deferredDirFailure << " context_loss=" << contextLoss
                  << " delivery=" << static_cast<int>(delivery)
                  << " classify/abort/message/recovery/fatal order passed\n";
    }

    void run_failure_order_cases() {
        const std::array<Case, 2> cases{{{"failure-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                                         {"failure-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}}};
        for (const Case& test : cases) {
            for (const auto delivery : {JuicerCuda::ExecutorTest::Delivery::Success,
                                        JuicerCuda::ExecutorTest::Delivery::Failure,
                                        JuicerCuda::ExecutorTest::Delivery::Throw}) {
                check_failure_order(test, delivery, true, JuicerCuda::runtime_failure_status(cudaErrorContextIsDestroyed), "renamed receiver arithmetic failure 100%");
            }
            for (const auto status : {JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED),
                                      JuicerCuda::runtime_failure_status(cudaErrorDeviceUninitialized),
                                      JuicerCuda::runtime_failure_status(cudaErrorInvalidValue),
                                      FjStatus{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}}) {
                for (const char* wording : {"test component=dir 100% device lost out of memory", "renamed arithmetic failure 100%", ""}) {
                    check_failure_order(test, JuicerCuda::ExecutorTest::Delivery::Success, true, status, wording);
                }
            }
            check_failure_order(test, JuicerCuda::ExecutorTest::Delivery::Success, false, JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED), "test component=scanner 100% renamed diagnostic");
            check_failure_order(test, JuicerCuda::ExecutorTest::Delivery::Success, false, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "non-DIR failure mentioning component=dir 100%");
            const std::string longDiagnostic = std::string(8192, 'x') + " 100% complete diagnostic";
            check_failure_order(test, JuicerCuda::ExecutorTest::Delivery::Success, true, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, longDiagnostic.c_str());
        }
    }

    void check_resource_failure(const Case& test,
                                const ParamSnapshot& parameters,
                                InstanceState& state,
                                JuicerCuda::ExecutorTest::ResourceFailures& failures,
                                const JuicerCuda::Failure& expectedFailure,
                                const char* stage,
                                bool preparationFailure = false) {
        using JuicerCuda::ExecutorTest::Event;
        JuicerAssets::NoiseTest::Observation noise;
        if (parameters.grainControls.active) {
            JuicerAssets::NoiseTest::observation = &noise;
        }
        JuicerCuda::ExecutorTest::FailureObservation observation;
        observation.armed = false;
        observation.status = expectedFailure.status;
        observation.diagnostic = expectedFailure.diagnostic.c_str();
        observation.stage = stage;
        JuicerCuda::ExecutorTest::observation = &observation;
        JuicerCuda::ExecutorTest::resourceFailures = &failures;
        bool fatal = false;
        try {
            (void)render_case(test, parameters, state);
        } catch (const OFX::Exception::Suite& error) {
            fatal = error.status() == kOfxStatErrFatal;
            observation.record(Event::FatalMapped);
        } catch (...) {
            JuicerCuda::ExecutorTest::observation = nullptr;
            JuicerCuda::ExecutorTest::resourceFailures = nullptr;
            throw;
        }
        JuicerCuda::ExecutorTest::observation = nullptr;
        JuicerCuda::ExecutorTest::resourceFailures = nullptr;
        JuicerAssets::NoiseTest::observation = nullptr;
        if (parameters.grainControls.active && (noise.operations != std::array<unsigned, 3>{1, 1, 1} || noise.gateViolation || !noise.releaseAfterRecovery)) {
            throw std::runtime_error("noise source did not survive native upload failure and recovery");
        }
        const bool contextLoss = expectedFailure.status.category == FJ_STATUS_CONTEXT_LOSS;
        std::array expectedEvents{Event::Injected, Event::Classified, Event::FrameAborted, Event::RecoveryStarted, Event::RecoveryEnded, Event::FatalMapped};
        if (preparationFailure) {
            // Root aborts failed preparation before returning it to the executor.
            std::swap(expectedEvents[1], expectedEvents[2]);
        }
        if (!fatal || observation.overflow || observation.count != expectedEvents.size() ||
            !std::equal(expectedEvents.begin(), expectedEvents.end(), observation.events.begin()) ||
            !observation.failureStageMatches || !observation.failureStatusMatches ||
            !observation.failureDiagnosticMatches ||
            observation.nativeRecoveryPending != contextLoss ||
            observation.adapterRecoveryPending != contextLoss ||
            state.submissionSnapshotLatchValid == contextLoss) {
            throw std::runtime_error(std::string(test.name) + ": resource failure propagation changed; category=" +
                                     std::to_string(observation.classifiedStatus.category) + " api=" +
                                     std::to_string(observation.classifiedStatus.api) + " code=" +
                                     std::to_string(observation.classifiedStatus.native_code) + " events=" +
                                     std::to_string(observation.count) + " stage_match=" +
                                     std::to_string(observation.failureStageMatches) + " diagnostic_match=" +
                                     std::to_string(observation.failureDiagnosticMatches));
        }
        std::cout << test.name << " stage=" << stage << " category=" << expectedFailure.status.category
                  << " api=" << expectedFailure.status.api << " code=" << expectedFailure.status.native_code
                  << " injections=" << failures.injections
                  << (preparationFailure ? " abort/classify" : " classify/abort")
                  << "/recovery/fatal order passed\n";
    }

    void run_defect_fence_failure_cases() {
        const std::array<Case, 2> cases{{{"defect-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                                         {"defect-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}}};
        for (const Case& test : cases) {
            InstanceState state;
            for (const bool failCreate : {true, false}) {
                for (const auto code : {cudaErrorContextIsDestroyed, cudaErrorInvalidResourceHandle, cudaErrorMemoryAllocation}) {
                    auto parameters = parameters_for(test);
                    parameters.filmDustAmount = 1.0f;
                    (void)render_case(test, parameters, state);
                    parameters.filmDustAmount = 0.0f;
                    parameters.gateDustAmount = 1.0f;
                    JuicerCuda::ExecutorTest::ResourceFailures failures;
                    (failCreate ? failures.eventCreate : failures.eventRecord) = code;
                    check_resource_failure(test, parameters, state, failures, {JuicerCuda::runtime_failure_status(code), "defect attachment retirement fence failed"}, Spektrafilm::scan_route_is_print(test.route) ? "print_focused_workspace_stage" : "direct_focused_workspace_stage");
                    if (failures.injections != 1) {
                        throw std::runtime_error("defect fence failure did not reach the resource owner");
                    }
                }
            }
        }
    }

    void run_noise_lifetime_cases() {
        const std::array cases{Case{"noise-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                               Case{"noise-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}};
        using JuicerAssets::NoiseTest::EarlierFailure;
        using JuicerAssets::NoiseTest::Observation;
        for (const Case& test : {cases[0], cases[1]}) {
            InstanceState state;
            const auto run = [&](Observation& observed, const ParamSnapshot& parameters, bool fatalExpected, unsigned cancelAt = 0, DestinationLayout layout = DestinationLayout::Matching, ExecutionPath path = ExecutionPath::Processor) {
                JuicerAssets::NoiseTest::observation = &observed;
                AbortObservation abort;
                abort.cancelAt = cancelAt;
                s_abortObservation = cancelAt ? &abort : nullptr;
                bool fatal = false;
                std::string failureDiagnostic;
                try {
                    std::uint64_t snapshotId = 1;
                    if (path == ExecutionPath::Direct) {
                        snapshotId = state.submissionSnapshotIdNext.fetch_add(1, std::memory_order_relaxed);
                        state.submissionSnapshotLatchValid = false;
                    }
                    (void)render_case(test, parameters, state, false, path, layout, snapshotId);
                } catch (const OFX::Exception::Suite& error) {
                    fatal = error.status() == kOfxStatErrFatal;
                } catch (const JuicerCuda::ExecutionFailure& failure) {
                    if (path != ExecutionPath::Direct) {
                        throw;
                    }
                    fatal = failure.failure.status.category != FJ_STATUS_SUCCESS;
                    observed.result = failure.failure.status;
                    failureDiagnostic = failure.failure.diagnostic;
                } catch (...) {
                    JuicerAssets::NoiseTest::observation = nullptr;
                    s_abortObservation = nullptr;
                    throw;
                }
                JuicerAssets::NoiseTest::observation = nullptr;
                s_abortObservation = nullptr;
                if (fatal != fatalExpected || observed.gateViolation ||
                    (observed.noiseStep && !observed.noiseStepGated) || fj_test_live_noise_owners() != 0) {
                    throw std::runtime_error("production noise gate/lifetime/failure contract changed: path=" + std::to_string(static_cast<unsigned>(path)) +
                                             " source=" + std::to_string(observed.sourceFailure) + " earlier=" + std::to_string(static_cast<unsigned>(observed.earlier)) +
                                             " fatal=" + std::to_string(fatal) + " expected=" + std::to_string(fatalExpected) +
                                             " gate=" + std::to_string(observed.gateViolation) + " step=" + std::to_string(observed.noiseStep) +
                                             " owners=" + std::to_string(fj_test_live_noise_owners()) + " status=" + std::to_string(observed.result.category) + " " + failureDiagnostic);
                }
            };
            auto active = parameters_for(test);
            active.grainControls.active = true;
            Observation disabled;
            run(disabled, parameters_for(test), false);
            if (disabled.operations != std::array<unsigned, 3>{}) {
                throw std::runtime_error("disabled grain acquired noise");
            }
            JuicerProcess::root().assets().release_cached_payloads();
            for (bool cold : {true, false}) {
                (void)cold;
                Observation success;
                run(success, active, false);
                if (success.operations != std::array<unsigned, 3>{1, 1, 1} || !success.noiseStep || !success.releaseAfterRecovery) {
                    throw std::runtime_error("cold/warm caller source lifetime changed");
                }
            }
            Observation directRoot;
            run(directRoot, active, false, 0, DestinationLayout::Matching, ExecutionPath::Direct);
            if (directRoot.operations != std::array<unsigned, 3>{1, 1, 1} || directRoot.noiseStep) {
                throw std::runtime_error("direct Root source fallback changed");
            }
            Observation directDisabled;
            run(directDisabled, parameters_for(test), false, 0, DestinationLayout::Matching, ExecutionPath::Direct);
            if (directDisabled.operations != std::array<unsigned, 3>{}) {
                throw std::runtime_error("inactive Root fallback acquired noise");
            }
            Observation directFailure;
            directFailure.sourceFailure = FJ_STATUS_ALLOCATION_FAILURE;
            run(directFailure, active, true, 0, DestinationLayout::Matching, ExecutionPath::Direct);
            if (directFailure.result.category != FJ_STATUS_ALLOCATION_FAILURE || directFailure.operations != std::array<unsigned, 3>{1, 1, 1}) {
                throw std::runtime_error("Root fallback source error or abort lifetime changed");
            }
            for (auto earlier : {EarlierFailure::Preflight, EarlierFailure::Focused, EarlierFailure::Print}) {
                if (earlier == EarlierFailure::Print && !Spektrafilm::scan_route_is_print(test.route)) {
                    continue;
                }
                Observation failure;
                failure.earlier = earlier;
                failure.sourceFailure = FJ_STATUS_INTERNAL_FAILURE;
                run(failure, active, true);
                if (failure.noiseStep || failure.operations != std::array<unsigned, 3>{1, 1, 1} || failure.exceptionDrops != 1 || (earlier != EarlierFailure::Preflight && !failure.exceptionReleasedAfterRecovery)) {
                    throw std::runtime_error("earlier failure lost precedence to source error: earlier=" + std::to_string(static_cast<unsigned>(earlier)) +
                                             " step=" + std::to_string(failure.noiseStep) + " operations=" + std::to_string(failure.operations[0]) + "/" +
                                             std::to_string(failure.operations[1]) + "/" + std::to_string(failure.operations[2]) +
                                             " drops=" + std::to_string(failure.exceptionDrops) + " recovery=" + std::to_string(failure.exceptionReleasedAfterRecovery));
                }
            }
            for (auto category : {FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE}) {
                Observation failure;
                failure.sourceFailure = category;
                run(failure, active, true);
                if (!failure.noiseStep || failure.result.category != category || (category == FJ_STATUS_INTERNAL_FAILURE && (failure.exceptionDrops != 1 || !failure.exceptionReleasedAfterRecovery))) {
                    throw std::runtime_error("typed deferred source error changed");
                }
            }
            Observation malformed;
            malformed.corrupt = true;
            run(malformed, active, true);
            if (!malformed.noiseStep || malformed.result.category != FJ_STATUS_UNSUPPORTED_INPUT) {
                throw std::runtime_error("failed source construction was not deferred");
            }
            Observation inspection;
            inspection.sourceFailure = FJ_STATUS_PREPARATION_FAILURE;
            run(inspection, active, true, 0, DestinationLayout::Uncovered);
            if (inspection.noiseStep || inspection.operations != std::array<unsigned, 3>{1, 1, 1}) {
                throw std::runtime_error("inspection/source ordering changed");
            }
            Observation missing;
            missing.earlier = EarlierFailure::Missing;
            run(missing, active, true);
            if (missing.operations != std::array<unsigned, 3>{}) {
                throw std::runtime_error("missing recipe acquired source");
            }
            for (unsigned checkpoint : {1u, 2u, 3u, 4u, 5u}) {
                Observation cancelled;
                run(cancelled, active, false, checkpoint);
                if (cancelled.operations != std::array<unsigned, 3>{1, 1, 1}) {
                    throw std::runtime_error("cancelled source was not released");
                }
            }
            {
                JuicerCuda::NativeCall existing(JuicerCuda::borrowed_owner());
                Observation reentry;
                JuicerAssets::NoiseTest::observation = &reentry;
                bool rejected = false;
                try {
                    (void)render_case(test, active, state);
                } catch (const JuicerCuda::ExecutionFailure& failure) {
                    rejected = failure.failure.status.category == FJ_STATUS_UNSUPPORTED_INPUT;
                } catch (...) {
                    JuicerAssets::NoiseTest::observation = nullptr;
                    throw;
                }
                JuicerAssets::NoiseTest::observation = nullptr;
                if (!rejected || reentry.operations != std::array<unsigned, 3>{}) {
                    throw std::runtime_error("native reentry reached Rust noise source");
                }
            }
            if (fj_cuda_retire_instance(JuicerCuda::borrowed_owner(), state.submissionSnapshotLatch.instanceToken.value, nullptr).category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("noise fixture retirement failed");
            }
        }
        {
            InstanceState state;
            auto active = parameters_for(cases[1]);
            active.grainControls.active = true;
            Observation closed;
            closed.closeBeforeAdmission = true;
            JuicerAssets::NoiseTest::observation = &closed;
            bool rejected = false;
            try {
                (void)render_case(cases[1], active, state);
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                rejected = failure.failure.status.category == FJ_STATUS_PREPARATION_FAILURE;
            } catch (...) {
                JuicerAssets::NoiseTest::observation = nullptr;
                throw;
            }
            JuicerAssets::NoiseTest::observation = nullptr;
            if (!rejected || closed.operations != std::array<unsigned, 3>{1, 1, 1} || closed.noiseStep || closed.gateViolation || fj_test_live_noise_owners() != 0) {
                throw std::runtime_error("final native admission/source destruction changed");
            }
            Observation sourceError;
            sourceError.sourceFailure = FJ_STATUS_INTERNAL_FAILURE;
            sourceError.closeBeforeAdmission = true;
            JuicerAssets::NoiseTest::observation = &sourceError;
            rejected = false;
            try {
                (void)render_case(cases[1], active, state);
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                rejected = failure.failure.status.category == FJ_STATUS_PREPARATION_FAILURE;
            } catch (...) {
                JuicerAssets::NoiseTest::observation = nullptr;
                throw;
            }
            JuicerAssets::NoiseTest::observation = nullptr;
            if (!rejected || sourceError.noiseStep || sourceError.exceptionDrops != 1 || sourceError.gateViolation) {
                throw std::runtime_error("closed native admission/held source exception ordering changed");
            }
        }
        std::puts("PASS production direct/print noise: calling-thread acquire/view/release outside gate, continuous projection gate, deferred typed failures, earlier failure precedence, missing/disabled, cancellation and reentry");
    }

    void run_grain_upload_failure_cases() {
        const std::array<Case, 2> cases{{{"grain-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                                         {"grain-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}}};
        for (const Case& test : cases) {
            for (const char* wording : {"renamed upload component=dir diagnostic", "device lost out of memory 100%"}) {
                const JuicerCuda::Failure capacity{{FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, wording};
                const JuicerCuda::Failure contextLoss{JuicerCuda::runtime_failure_status(cudaErrorContextIsDestroyed), wording};
                const JuicerCuda::Failure driverLoss{JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED), wording};
                const JuicerCuda::Failure ordinary{JuicerCuda::runtime_failure_status(cudaErrorInvalidValue), wording};
                struct UploadCase {
                    std::optional<JuicerCuda::Failure> stbn;
                    std::optional<JuicerCuda::Failure> wang;
                    FjStatus expectedStatus{};
                };
                const std::array<UploadCase, 8> uploads{{{capacity, contextLoss, contextLoss.status},
                                                         {contextLoss, ordinary, contextLoss.status},
                                                         {capacity, std::nullopt, capacity.status},
                                                         {contextLoss, std::nullopt, contextLoss.status},
                                                         {std::nullopt, driverLoss, driverLoss.status},
                                                         {capacity, ordinary, capacity.status},
                                                         {ordinary, capacity, ordinary.status},
                                                         {contextLoss, driverLoss, contextLoss.status}}};
                for (const auto& upload : uploads) {
                    InstanceState state;
                    auto parameters = parameters_for(test);
                    parameters.grainControls.active = true;
                    JuicerCuda::ExecutorTest::ResourceFailures failures;
                    failures.stbn = upload.stbn;
                    failures.wang = upload.wang;
                    if (failures.stbn) {
                        failures.stbn->diagnostic = std::string("STBN: ") + wording;
                    }
                    if (failures.wang) {
                        failures.wang->diagnostic = std::string("Wang: ") + wording;
                    }
                    std::string diagnostic = std::string(upload.stbn ? "STBN: " : "Wang: ") + wording;
                    if (upload.stbn && upload.wang) {
                        diagnostic += " | Wang upload failed: Wang: ";
                        diagnostic += wording;
                    }
                    diagnostic += " route=";
                    diagnostic += Spektrafilm::scan_route_metadata(test.route).key;
                    check_resource_failure(test, parameters, state, failures, {upload.expectedStatus, diagnostic}, "ensure_grain_static_assets_uploaded", true);
                    if (failures.stbnAttempts != 1 || failures.wangAttempts != 1 ||
                        failures.injections != static_cast<int>(upload.stbn.has_value()) + static_cast<int>(upload.wang.has_value())) {
                        throw std::runtime_error("grain upload attempts or failure order changed");
                    }
                }
            }
        }
    }
} // namespace
#endif

#if defined(JUICER_ACCEPTED_CAPTURE_PATH)
namespace {
    void run_accepted_clean_captures() {
        std::ifstream fixture(JUICER_ACCEPTED_CAPTURE_PATH);
        if (!fixture) {
            throw std::runtime_error("accepted CUDA capture fixture unavailable");
        }
        const auto captures = nlohmann::json::parse(fixture);
        const auto decode = [](const nlohmann::json& bits) {
            std::vector<float> pixels;
            pixels.reserve(bits.size());
            for (const auto& value : bits) {
                pixels.push_back(std::bit_cast<float>(value.get<std::uint32_t>()));
            }
            return pixels;
        };
        const auto input = decode(captures.at("input_bits"));
        for (const auto& capture : captures.at("cases")) {
            const auto& settings = capture.at("settings");
            const std::string name = capture.at("name").get<std::string>();
            Case test{};
            test.name = name.c_str();
            test.route = static_cast<Spektrafilm::ScanRoute>(settings.at("route").get<int>());
            test.components = 3;
            test.sourcePixels = &input;
            test.fullWidth = settings.at("width").get<int>();
            test.fullHeight = settings.at("height").get<int>();
            auto parameters = parameters_for(test);
            parameters.filmProfileKey = settings.at("film").get<std::string>();
            parameters.printProfileKey = settings.at("print").get<std::string>();
            parameters.printGammaFactor = settings.at("print_gamma").get<double>();
            parameters.inputColorSpace = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::ITU_R_BT2020);
            parameters.inputCctfDecoding = 0;
            parameters.outputColorSpace = OutputEncoding::toIndex(OutputEncoding::ColorSpace::ITU_R_BT2020);
            parameters.outputCctfEncoding = 0;
            parameters.dirCouplers.active = false;
            parameters.grainControls.active = false;
            parameters.glareActive = false;
            parameters.glarePercent = 0.03;
            InstanceState state;
            const auto pixels = render_case(test, parameters, state);
            compare_pixels(name, pixels, decode(capture.at("output_bits")));
            std::cerr << name << ": exact stored CUDA samples within retained pixel bounds\n";
        }
    }
} // namespace
#endif

int main(int argc, char** argv) {
    JuicerCuda::Owner cudaOwner;
    try {
        cudaOwner.create(JuicerProcess::data_directory());
        NarrowHost host;
        require_cuda(cudaSetDevice(0), "select device");
        require_cuda(cudaFree(nullptr), "initialize CUDA");
        JuicerProcess::root().ensure_bootstrap();
#if defined(JUICER_ACCEPTED_CAPTURE_PATH)
        if (argc == 2 && std::string(argv[1]) == "--accepted-clean-captures") {
            run_accepted_clean_captures();
            if (cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("accepted capture owner cleanup failed");
            }
            return 0;
        }
#endif
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
        if (argc == 2 && std::string(argv[1]) == "--noise-lifetime") {
            run_noise_lifetime_cases();
            if (cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("noise lifetime owner cleanup failed");
            }
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--resource-failure") {
            const std::string resourceFailure = argv[2];
            if (resourceFailure == "DefectFence") {
                run_defect_fence_failure_cases();
            } else if (resourceFailure == "GrainUpload") {
                run_grain_upload_failure_cases();
            } else {
                throw std::runtime_error("unknown resource failure case: " + resourceFailure);
            }
            const auto closed = fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
            if (closed.category != FJ_STATUS_SUCCESS || cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("terminal cleanup after native failure failed");
            }
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--failure-order") {
            run_failure_order_cases();
            const auto closed = fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
            if (closed.category != FJ_STATUS_SUCCESS || cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("terminal cleanup after native failure failed");
            }
            return 0;
        }
#endif
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        const bool renderContract = argc == 3 && std::string(argv[1]) == "--render-contract";
        if (argc != 3 || (!renderContract && std::string(argv[1]) != "--prepared-case")) {
            throw std::runtime_error("usage: JuicerPreparedBoundaryProbe --prepared-case <name>");
        }
        const std::string preparedCase = argv[2];
        bool preparedCaseFound = false;
        constexpr bool emit = false;
#else
        const bool cutoverContract = argc == 2 && std::string(argv[1]) == "--cutover-contract";
        const bool sequentialOwners = argc == 2 && std::string(argv[1]) == "--sequential-owners";
        const bool emit = argc == 2 && std::string(argv[1]) == "--emit-reference";
        if (argc != 1 && !emit && !sequentialOwners && !cutoverContract) {
            throw std::runtime_error("usage: JuicerProcessorReferenceProbe [--emit-reference]");
        }
#endif
        const Case cases[]{{"negative-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                           {"negative-print", Spektrafilm::ScanRoute::NegativePrintScan, 1, 4, 0, 0},
                           {"positive-direct", Spektrafilm::ScanRoute::PositiveDirectScan, 2, 3, 11, -3},
                           {"positive-print", Spektrafilm::ScanRoute::PositivePrintScan, 0, 4, 0, 0},
                           {"combined-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 3, 0, 0, true},
                           {"glare-plus-zero", Spektrafilm::ScanRoute::NegativePrintScan, 0, 3, 0, 0, false, true, false},
                           {"glare-minus-zero", Spektrafilm::ScanRoute::NegativePrintScan, 0, 3, 0, 0, false, true, true}
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
                           ,
                           {"mallett-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 1, 4, 0, 0},
                           {"arctic-print", Spektrafilm::ScanRoute::NegativePrintScan, 2, 4, 0, 0}
#endif
        };
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        constexpr bool numericalFixture = false;
#else
        const bool numericalFixture = !emit && !sequentialOwners && !cutoverContract;
#endif
        std::ifstream fixture;
        if (numericalFixture) {
            fixture.open(JUICER_PROCESSOR_REFERENCE_PATH);
            if (!fixture) {
                throw std::runtime_error("processor reference fixture unavailable");
            }
        }
#if !defined(JUICER_PREPARED_BOUNDARY_TEST)
        void* initialContext = nullptr;
        std::uint64_t previousEpoch = 0;
        if (sequentialOwners) {
            CUcontext current = nullptr;
            if (cuCtxGetCurrent(&current) != CUDA_SUCCESS) {
                throw std::runtime_error("missing sequential-owner CUDA context");
            }
            initialContext = current;
        }
#endif
        std::vector<float> freshPlus;
        std::vector<float> freshMinus;
        for (const Case& test : cases) {
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
            if (preparedCase != test.name) {
                if (numericalFixture) {
                    fixture.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
                continue;
            }
            preparedCaseFound = true;
            // Each CTest case starts a fresh process and owner. C crosses the
            // boundary first, so no other caller can hide its table uploads.
            std::cerr << "cold C case: " << test.name << '\n';
            InstanceState boundaryState;
            const auto boundary = render_case(test, parameters_for(test), boundaryState, false, renderContract ? ExecutionPath::Contract : ExecutionPath::Boundary);
            const auto warmBoundary = render_case(test, parameters_for(test), boundaryState, false, ExecutionPath::Boundary);
            const auto direct = [&] {
                InstanceState directState;
                return render_case(test, parameters_for(test), directState, false, ExecutionPath::Direct);
            }();
#endif
            std::cerr << "processor case: " << test.name << '\n';
            const std::vector<float> pixels = [&] {
                InstanceState state;
                auto rendered = render_case(test, parameters_for(test), state);
#if !defined(JUICER_PREPARED_BOUNDARY_TEST)
                if (sequentialOwners) {
                    const auto& snapshot = state.submissionSnapshotLatch;
                    JuicerCuda::ResourceManager::RegistryContextSnapshot admitted;
                    if (snapshot.deviceContextKey.contextOpaque != initialContext ||
                        !JuicerCuda::ResourceManager::registry_begin_submission(snapshot.deviceContextKey, admitted)) {
                        throw std::runtime_error("sequential owner changed or failed to admit the live context");
                    }
                    const bool ended = JuicerCuda::ResourceManager::registry_note_submission_end(snapshot.deviceContextKey);
                    if (!ended || admitted.contextEpoch <= previousEpoch) {
                        throw std::runtime_error("owner recreation did not advance the native epoch");
                    }
                    std::cerr << "owner transition case=" << test.name << " previous_epoch=" << previousEpoch << " admitted_epoch=" << admitted.contextEpoch << '\n';
                    previousEpoch = admitted.contextEpoch;
                }
#endif
                return rendered;
            }();
            if (test.signedZeroGlare) {
                (test.negativeZero ? freshMinus : freshPlus) = pixels;
            }
            if (emit) {
                std::cout << test.name << ' ' << pixels.size();
                for (float pixel : pixels) {
                    std::cout << ' ' << std::setprecision(std::numeric_limits<float>::max_digits10) << pixel;
                }
                std::cout << '\n';
            } else {
                std::vector<float> expected = pixels;
                if (numericalFixture) {
                    std::string name;
                    std::size_t count = 0;
                    fixture >> name >> count;
                    if (!fixture || name != test.name || count != pixels.size()) {
                        throw std::runtime_error(std::string(test.name) + ": fixture row mismatch");
                    }
                    expected.resize(count);
                    for (float& pixel : expected) {
                        fixture >> pixel;
                    }
                    if (!fixture) {
                        throw std::runtime_error(std::string(test.name) + ": incomplete fixture row");
                    }
                    compare_pixels(test.name, pixels, expected);
                }
#if !defined(JUICER_PREPARED_BOUNDARY_TEST)
                if (cutoverContract && (std::string_view(test.name) == "negative-direct" || std::string_view(test.name) == "negative-print")) {
                    check_cutover_callbacks(test, expected);
                }
                if (sequentialOwners && std::string_view(test.name) == "negative-direct") {
                    Case uniform = test;
                    uniform.uniform = true;
                    std::vector<float> uniformExpected(expected.size());
                    for (std::size_t sample = 0; sample < uniformExpected.size(); ++sample) {
                        uniformExpected[sample] = expected[sample % 3];
                    }
                    // At this fixture's physical scale the DIR FIR has no nonzero
                    // off-center float samples. Normalized diffusion preserves a
                    // constant exposure. Both controls use the immutable first pixel.
                    {
                        InstanceState control;
                        compare_pixels("uniform direct control", render_case(uniform, parameters_for(uniform), control), uniformExpected);
                    }
                    if (cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                        throw std::runtime_error("direct diffusion predecessor close failed");
                    }
                    cudaOwner.create(JuicerProcess::data_directory());
                    JuicerProcess::root().ensure_bootstrap();
                    {
                        auto parameters = parameters_for(uniform);
                        parameters.cameraDiffusion.active = true;
                        InstanceState diffusionState;
                        compare_pixels("uniform direct diffusion", render_case(uniform, parameters, diffusionState), uniformExpected);
                        JuicerCuda::ResourceManager::RegistryContextSnapshot admitted;
                        const auto& key = diffusionState.submissionSnapshotLatch.deviceContextKey;
                        if (key.contextOpaque != initialContext || !JuicerCuda::ResourceManager::registry_begin_submission(key, admitted)) {
                            throw std::runtime_error("direct diffusion context admission failed");
                        }
                        const bool ended = JuicerCuda::ResourceManager::registry_note_submission_end(key);
                        if (!ended || admitted.contextEpoch <= previousEpoch) {
                            throw std::runtime_error("direct diffusion did not use a later admitted epoch");
                        }
                        previousEpoch = admitted.contextEpoch;
                        std::cerr << "uniform direct diffusion matches immutable constant sample at epoch=" << previousEpoch << '\n';
                    }
                }
                if (sequentialOwners && !test.combined) {
                    InstanceState originState;
                    compare_pixels(std::string(test.name) + ": independent destination origin", render_case(test, parameters_for(test), originState, false, ExecutionPath::Processor, DestinationLayout::Offset), expected);
                    InstanceState rejectedState;
                    bool fatal = false;
                    try {
                        (void)render_case(test, parameters_for(test), rejectedState, false, ExecutionPath::Processor, DestinationLayout::Uncovered);
                    } catch (const OFX::Exception::Suite& error) {
                        fatal = error.status() == kOfxStatErrFatal;
                    }
                    if (!fatal || rejectedState.submissionSnapshotLatchValid) {
                        throw std::runtime_error("uncovered destination was not rejected before submission latch publication");
                    }
                }
#endif
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
                RenderAssertions::compare_bits(std::string(test.name) + ": processor/direct", pixels, direct);
                RenderAssertions::compare_bits(std::string(test.name) + ": direct/cold C", direct, boundary);
                RenderAssertions::compare_bits(std::string(test.name) + ": cold/warm C", boundary, warmBoundary);
                std::cerr << test.name << ": processor/direct/cold C/warm C bit-exact\n";
                if (renderContract) {
                    auto grainParameters = parameters_for(test);
                    grainParameters.grainControls.active = true;
                    InstanceState grainState;
                    (void)render_case(test, grainParameters, grainState, false, ExecutionPath::Boundary, DestinationLayout::Matching, 2);
                }
#endif
            }
#if !defined(JUICER_PREPARED_BOUNDARY_TEST)
            if (sequentialOwners) {
                if (cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                    throw std::runtime_error("sequential native owner failed to close");
                }
                cudaOwner.create(JuicerProcess::data_directory());
                JuicerProcess::root().ensure_bootstrap();
                CUcontext current = nullptr;
                if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != initialContext) {
                    throw std::runtime_error("owner close/recreate replaced the live CUDA context");
                }
            }
#endif
        }
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        if (!preparedCaseFound) {
            throw std::runtime_error("unknown prepared-boundary case: " + preparedCase);
        }
#else
        if (!emit) {
            if (std::equal(freshPlus.begin(), freshPlus.end(), freshMinus.begin())) {
                throw std::runtime_error("signed-zero fresh print renders did not differ");
            }
            InstanceState transition;
            compare_pixels("glare plus to minus initial", render_case(cases[5], parameters_for(cases[5]), transition), freshPlus);
            compare_pixels("glare plus to minus", render_case(cases[6], parameters_for(cases[6]), transition), freshMinus);
            compare_pixels("glare minus to plus", render_case(cases[5], parameters_for(cases[5]), transition), freshPlus);
        }
        InstanceState emptyWindowState;
        (void)render_case(cases[0], parameters_for(cases[0]), emptyWindowState, true);
#endif
        if (cudaOwner.close().category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error("final native owner close failed");
        }
        return 0;
    } catch (const std::exception& error) {
        (void)std::fprintf(stderr, "%s\n", error.what());
        try {
            fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
        } catch (...) {
            (void)std::fputs("shutdown failed after processor test error\n", stderr);
        }
        return 1;
    }
}
