#include <algorithm>
#include <chrono>
#include <thread>
#include <array>
#include <bit>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include "JuicerEffect.h"
#include "JuicerState.h"
#include "ParamNames.h"
#include "ProcessRoot.h"
#include "juicer_cuda_owner.h"
#include "ColorTransforms.h"
#include "ofxMemory.h"
#include "ofxMultiThread.h"
#include "ofxsSupportPrivate.h"

namespace RouteFaultTest {
    void fail_on_call(int call);
}

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray&) {}

namespace JuicerTestSupport {

    class EffectTraceObserver final {
    public:
        struct Snapshot {
            bool pendingValid = false;
            bool pendingInvalid = false;
            std::string pendingDiagnostic;
            bool pendingPrintRoute = false;
            Spektrafilm::ScanRoute pendingRoute = Spektrafilm::kDefaultScanRoute;
            double pendingExposureEv = 0.0;
            Spektrafilm::SpatialOpticsControls pendingOptics;
            PhysicalEffectsControls pendingEffects;
            float pendingFilmGamma = 0.0f;
            double pendingPrintGamma = 0.0;
            Spektrafilm::VisualGrainControls pendingGrain;
            DirCouplersControls pendingDir;
            std::uint64_t pendingFullHash = 0;
            std::uint64_t buildCounter = 0;
            std::uint64_t recipeHash = 0;
            std::uint64_t lastParamHash = 0;
            bool paramEventsSuppressed = false;
            bool latchValid = false;
            std::uint64_t latchSnapshotId = 0;
            std::uint64_t latchFrameToken = 0;
        };

        static Snapshot snapshot(JuicerEffect& effect) {
            Snapshot result{};
            InstanceState& state = *effect._state;
            {
                std::lock_guard<std::mutex> lock(state.pending.m);
                if (const auto* valid =
                        std::get_if<PendingParamsState::Valid>(&state.pending.value)) {
                    result.pendingValid = true;
                    result.pendingOptics.cameraDiffusion = valid->params.cameraDiffusion;
                    result.pendingOptics.enlargerDiffusion = valid->params.enlargerDiffusion;
                    result.pendingOptics.scatterHalation = valid->params.scatterHalationControls;
                    result.pendingExposureEv =
                        valid->params.cameraExposureCompensationEv;
                    result.pendingFilmGamma = valid->params.filmGammaFactor;
                    result.pendingEffects = {valid->params.filmDustAmount, valid->params.filmScratchAmount, valid->params.gateDustAmount, valid->params.gateScratchAmount, valid->params.gateWeaveAmount};
                    result.pendingPrintGamma = valid->params.printGammaFactor;
                    result.pendingGrain = valid->params.grainControls;
                    result.pendingDir = valid->params.dirCouplers;
                    result.pendingFullHash = valid->fullHash;
                    result.pendingPrintRoute =
                        Spektrafilm::scan_route_is_print(
                            valid->params.scanRoute);
                    result.pendingRoute = valid->params.scanRoute;
                } else if (std::holds_alternative<
                               PendingParamsState::InvalidSnapshotControls>(
                               state.pending.value)) {
                    result.pendingInvalid = true;
                    result.pendingDiagnostic = std::get<
                                                   PendingParamsState::InvalidSnapshotControls>(
                                                   state.pending.value)
                                                   .diagnostic;
                }
            }
            const auto direct =
                JuicerAtomic::load_shared_ptr(&state.activeDirectState);
            const auto print =
                JuicerAtomic::load_shared_ptr(&state.activePrintState);
            if ((result.pendingPrintRoute || !direct) && print) {
                result.buildCounter = print->buildCounter;
                result.recipeHash = print->recipe.hash;
            } else if (direct) {
                result.buildCounter = direct->buildCounter;
                result.recipeHash = direct->recipe.hash;
            }
            result.lastParamHash = state.lastHash.load(std::memory_order_acquire);
            result.paramEventsSuppressed = state.suppressParamEvents;
            {
                std::lock_guard<std::mutex> lock(
                    state.submissionSnapshotLatchMutex);
                result.latchValid = state.submissionSnapshotLatchValid;
                if (result.latchValid) {
                    result.latchSnapshotId =
                        state.submissionSnapshotLatch.snapshotId;
                    result.latchFrameToken =
                        state.submissionSnapshotLatch.frameToken.value;
                }
            }
            return result;
        }

        static PendingRenderAdmissionResult admit(JuicerEffect& effect) {
            return admit_pending_render_state(*effect._state);
        }
    };

} // namespace JuicerTestSupport

namespace {

    struct PropertyBag {
        using IntWriteObserver =
            void (*)(void*, const char*, const char*, int);

        std::map<std::string, std::vector<std::string>> strings;
        std::map<std::string, std::vector<int>> ints;
        std::map<std::string, std::vector<double>> doubles;
        std::map<std::string, std::vector<void*>> pointers;
        std::string owningParamName;
        void* intWriteContext = nullptr;
        IntWriteObserver intWriteObserver = nullptr;
    };

    template <typename T>
    OfxStatus set_property(
        std::map<std::string, std::vector<T>>& properties,
        const char* name,
        int index,
        T value) {
        if (!name || index < 0) {
            return kOfxStatErrBadHandle;
        }
        auto& values = properties[name];
        if (values.size() <= static_cast<std::size_t>(index)) {
            values.resize(static_cast<std::size_t>(index) + 1);
        }
        values[static_cast<std::size_t>(index)] = std::move(value);
        return kOfxStatOK;
    }

    template <typename T>
    OfxStatus get_property(
        const std::map<std::string, std::vector<T>>& properties,
        const char* name,
        int index,
        T* output) {
        if (!name || !output || index < 0) {
            return kOfxStatErrBadHandle;
        }
        const auto found = properties.find(name);
        if (found == properties.end() ||
            static_cast<std::size_t>(index) >= found->second.size()) {
            return kOfxStatErrUnknown;
        }
        *output = found->second[static_cast<std::size_t>(index)];
        return kOfxStatOK;
    }

    OfxStatus property_set_pointer(
        OfxPropertySetHandle handle,
        const char* name,
        int index,
        void* value) {
        return handle
                   ? set_property(
                         reinterpret_cast<PropertyBag*>(handle)->pointers,
                         name,
                         index,
                         value)
                   : kOfxStatErrBadHandle;
    }

    OfxStatus property_set_string(
        OfxPropertySetHandle handle,
        const char* name,
        int index,
        const char* value) {
        return handle
                   ? set_property(
                         reinterpret_cast<PropertyBag*>(handle)->strings,
                         name,
                         index,
                         std::string(value ? value : ""))
                   : kOfxStatErrBadHandle;
    }

    OfxStatus property_set_double(
        OfxPropertySetHandle handle,
        const char* name,
        int index,
        double value) {
        return handle
                   ? set_property(
                         reinterpret_cast<PropertyBag*>(handle)->doubles,
                         name,
                         index,
                         value)
                   : kOfxStatErrBadHandle;
    }

    OfxStatus property_set_int(
        OfxPropertySetHandle handle,
        const char* name,
        int index,
        int value) {
        if (!handle) {
            return kOfxStatErrBadHandle;
        }
        auto& properties = *reinterpret_cast<PropertyBag*>(handle);
        const OfxStatus status = set_property(
            properties.ints,
            name,
            index,
            value);
        if (status == kOfxStatOK && properties.intWriteObserver &&
            (std::string_view(name) == kOfxParamPropEnabled ||
             std::string_view(name) == kOfxParamPropSecret)) {
            properties.intWriteObserver(
                properties.intWriteContext,
                properties.owningParamName.c_str(),
                name,
                value);
        }
        return status;
    }

    template <typename T, typename Setter>
    OfxStatus set_property_array(
        OfxPropertySetHandle handle,
        const char* name,
        int count,
        const T* values,
        Setter setter) {
        if (!handle || !name || count < 0 || (count > 0 && !values)) {
            return kOfxStatErrBadHandle;
        }
        for (int index = 0; index < count; ++index) {
            const OfxStatus status = setter(handle, name, index, values[index]);
            if (status != kOfxStatOK) {
                return status;
            }
        }
        return kOfxStatOK;
    }

    OfxStatus property_set_pointer_n(
        OfxPropertySetHandle handle,
        const char* name,
        int count,
        void* const* values) {
        return set_property_array(
            handle,
            name,
            count,
            values,
            property_set_pointer);
    }

    OfxStatus property_set_string_n(
        OfxPropertySetHandle handle,
        const char* name,
        int count,
        const char* const* values) {
        return set_property_array(
            handle,
            name,
            count,
            values,
            property_set_string);
    }

    OfxStatus property_set_double_n(
        OfxPropertySetHandle handle,
        const char* name,
        int count,
        const double* values) {
        return set_property_array(
            handle,
            name,
            count,
            values,
            property_set_double);
    }

    OfxStatus property_set_int_n(
        OfxPropertySetHandle handle,
        const char* name,
        int count,
        const int* values) {
        return set_property_array(
            handle,
            name,
            count,
            values,
            property_set_int);
    }

    OfxStatus property_get_pointer(
        OfxPropertySetHandle handle,
        const char* name,
        int index,
        void** output) {
        return handle
                   ? get_property(
                         reinterpret_cast<PropertyBag*>(handle)->pointers,
                         name,
                         index,
                         output)
                   : kOfxStatErrBadHandle;
    }

    OfxStatus property_get_string(
        OfxPropertySetHandle handle,
        const char* name,
        int index,
        char** output) {
        if (!handle || !name || !output || index < 0) {
            return kOfxStatErrBadHandle;
        }
        auto& strings = reinterpret_cast<PropertyBag*>(handle)->strings;
        const auto found = strings.find(name);
        if (found == strings.end() ||
            static_cast<std::size_t>(index) >= found->second.size()) {
            return kOfxStatErrUnknown;
        }
        *output = const_cast<char*>(
            found->second[static_cast<std::size_t>(index)].c_str());
        return kOfxStatOK;
    }

    OfxStatus property_get_double(
        OfxPropertySetHandle handle,
        const char* name,
        int index,
        double* output) {
        return handle
                   ? get_property(
                         reinterpret_cast<PropertyBag*>(handle)->doubles,
                         name,
                         index,
                         output)
                   : kOfxStatErrBadHandle;
    }

    OfxStatus property_get_int(
        OfxPropertySetHandle handle,
        const char* name,
        int index,
        int* output) {
        return handle
                   ? get_property(
                         reinterpret_cast<PropertyBag*>(handle)->ints,
                         name,
                         index,
                         output)
                   : kOfxStatErrBadHandle;
    }

    template <typename T, typename Getter>
    OfxStatus get_property_array(
        OfxPropertySetHandle handle,
        const char* name,
        int count,
        T* values,
        Getter getter) {
        if (!handle || !name || count < 0 || (count > 0 && !values)) {
            return kOfxStatErrBadHandle;
        }
        for (int index = 0; index < count; ++index) {
            const OfxStatus status = getter(handle, name, index, &values[index]);
            if (status != kOfxStatOK) {
                return status;
            }
        }
        return kOfxStatOK;
    }

    OfxStatus property_get_pointer_n(
        OfxPropertySetHandle handle,
        const char* name,
        int count,
        void** values) {
        return get_property_array(
            handle,
            name,
            count,
            values,
            property_get_pointer);
    }

    OfxStatus property_get_string_n(
        OfxPropertySetHandle handle,
        const char* name,
        int count,
        char** values) {
        return get_property_array(
            handle,
            name,
            count,
            values,
            property_get_string);
    }

    OfxStatus property_get_double_n(
        OfxPropertySetHandle handle,
        const char* name,
        int count,
        double* values) {
        return get_property_array(
            handle,
            name,
            count,
            values,
            property_get_double);
    }

    OfxStatus property_get_int_n(
        OfxPropertySetHandle handle,
        const char* name,
        int count,
        int* values) {
        return get_property_array(
            handle,
            name,
            count,
            values,
            property_get_int);
    }

    OfxStatus property_reset(OfxPropertySetHandle handle, const char* name) {
        if (!handle || !name) {
            return kOfxStatErrBadHandle;
        }
        auto& bag = *reinterpret_cast<PropertyBag*>(handle);
        bag.strings.erase(name);
        bag.ints.erase(name);
        bag.doubles.erase(name);
        bag.pointers.erase(name);
        return kOfxStatOK;
    }

    OfxStatus property_dimension(
        OfxPropertySetHandle handle,
        const char* name,
        int* count) {
        if (!handle || !name || !count) {
            return kOfxStatErrBadHandle;
        }
        const auto& bag = *reinterpret_cast<PropertyBag*>(handle);
        const bool exists = bag.strings.contains(name) ||
                            bag.ints.contains(name) ||
                            bag.doubles.contains(name) ||
                            bag.pointers.contains(name);
        auto dimension = [&](const auto& properties) -> std::size_t {
            const auto found = properties.find(name);
            return found == properties.end() ? 0u : found->second.size();
        };
        *count = static_cast<int>(std::max(
            {dimension(bag.strings),
             dimension(bag.ints),
             dimension(bag.doubles),
             dimension(bag.pointers)}));
        return exists ? kOfxStatOK : kOfxStatErrUnknown;
    }

    enum class ParamKind : std::uint8_t {
        Double,
        Double2,
        Double3,
        Int,
        Boolean,
        Choice,
        StrChoice,
        PushButton
    };

    struct ParamValue {
        std::array<double, 3> numbers{};
        int integer = 0;
        std::string string;
    };

    struct NativeParam {
        std::string name;
        ParamKind kind = ParamKind::Double;
        PropertyBag properties;
        ParamValue value;
        std::map<double, ParamValue> keys;

        const ParamValue& value_at(double time) const {
            const auto upper = keys.upper_bound(time);
            if (upper == keys.begin()) {
                return value;
            }
            return std::prev(upper)->second;
        }
    };

    struct NativeClip {
        std::string name;
        PropertyBag properties;
        PropertyBag imageProperties;
    };

    class NativeHost;

    struct NativeEffectHandle {
        PropertyBag properties;
        NativeHost* host = nullptr;
    };

    class DeviceFrame final {
    public:
        static constexpr int kWidth = 7;
        static constexpr int kHeight = 5;
        static constexpr int kComponents = 4;
        static constexpr int kPitch = kWidth * kComponents + 3;
        static constexpr float kCanary = -8.0f;

        DeviceFrame() {
            const std::size_t bytes = byte_count();
            require(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking), "create stream");
            require(cudaMalloc(&_source, bytes), "allocate source");
            require(cudaMalloc(&_destination, bytes), "allocate destination");
            const std::size_t elementCount =
                static_cast<std::size_t>(kPitch) *
                static_cast<std::size_t>(kHeight);
            _input.assign(elementCount, -7.0f);
            _output.assign(elementCount, kCanary);
            for (int y = 0; y < kHeight; ++y) {
                for (int x = 0; x < kWidth; ++x) {
                    const std::size_t offset =
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(kPitch) +
                        static_cast<std::size_t>(x) *
                            static_cast<std::size_t>(kComponents);
                    for (int channel = 0; channel < 3; ++channel) {
                        _input[offset + static_cast<std::size_t>(channel)] =
                            static_cast<float>((x + 3 * y + 7 * channel) % 19) /
                            19.0f;
                    }
                    _input[offset + 3u] =
                        0.125f + static_cast<float>((x + y) % 4) * 0.25f;
                }
            }
        }

        ~DeviceFrame() {
            if (_stream) {
                (void)cudaStreamSynchronize(_stream);
            }
            if (_destination) {
                (void)cudaFree(_destination);
            }
            if (_source) {
                (void)cudaFree(_source);
            }
            if (_stream) {
                (void)cudaStreamDestroy(_stream);
            }
        }

        void reset() {
            std::fill(_output.begin(), _output.end(), kCanary);
            require(
                cudaMemcpyAsync(
                    _source,
                    _input.data(),
                    byte_count(),
                    cudaMemcpyHostToDevice,
                    _stream),
                "upload source");
            require(
                cudaMemcpyAsync(
                    _destination,
                    _output.data(),
                    byte_count(),
                    cudaMemcpyHostToDevice,
                    _stream),
                "reset destination");
            require(cudaStreamSynchronize(_stream), "reset synchronize");
        }

        std::uint64_t collect_signature(bool validateRenderedOutput) {
            require(
                cudaMemcpyAsync(
                    _output.data(),
                    _destination,
                    byte_count(),
                    cudaMemcpyDeviceToHost,
                    _stream),
                "download destination");
            require(cudaStreamSynchronize(_stream), "render synchronize");
            std::uint64_t hash = 1469598103934665603ull;
            for (int y = 0; y < kHeight; ++y) {
                for (int x = 0; x < kWidth; ++x) {
                    const std::size_t offset =
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(kPitch) +
                        static_cast<std::size_t>(x) *
                            static_cast<std::size_t>(kComponents);
                    for (int channel = 0; channel < kComponents; ++channel) {
                        const float value =
                            _output[offset + static_cast<std::size_t>(channel)];
                        if (!std::isfinite(value)) {
                            throw std::runtime_error("adapter trace produced nonfinite output");
                        }
                        if (validateRenderedOutput && channel == 3 &&
                            std::bit_cast<std::uint32_t>(value) !=
                                std::bit_cast<std::uint32_t>(
                                    _input[offset + 3u])) {
                            throw std::runtime_error("adapter trace changed alpha");
                        }
                        hash ^= std::bit_cast<std::uint32_t>(value);
                        hash *= 1099511628211ull;
                    }
                }
                for (int index = kWidth * kComponents; index < kPitch; ++index) {
                    const std::size_t offset =
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(kPitch) +
                        static_cast<std::size_t>(index);
                    const float value = _output[offset];
                    if (std::bit_cast<std::uint32_t>(value) !=
                        std::bit_cast<std::uint32_t>(kCanary)) {
                        throw std::runtime_error(
                            "adapter trace changed destination padding");
                    }
                }
            }
            return hash;
        }

        float* source() const noexcept {
            return _source;
        }

        float* destination() const noexcept {
            return _destination;
        }

        cudaStream_t stream() const noexcept {
            return _stream;
        }

    private:
        static std::size_t byte_count() {
            return static_cast<std::size_t>(kPitch * kHeight) * sizeof(float);
        }

        static void require(cudaError_t status, const char* operation) {
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(operation) + ": " + cudaGetErrorString(status));
            }
        }

        float* _source = nullptr;
        float* _destination = nullptr;
        cudaStream_t _stream = nullptr;
        std::vector<float> _input;
        std::vector<float> _output;
    };

    class NativeHost final {
    public:
        NativeHost() {
            if (s_active) {
                throw std::runtime_error("only one native adapter host may be active");
            }
            s_active = this;
            configure_suites();
            configure_host();
            configure_effect();
            configure_descriptor();
            configure_clips();
            configure_params();
            OFX::Private::gPropSuite = &_propertySuite;
            OFX::Private::gEffectSuite = &_effectSuite;
            OFX::Private::gParamSuite = &_parameterSuite;
            OFX::Private::gMessageSuite = &_messageSuite;
        }

        ~NativeHost() {
            OFX::Private::gMessageSuite = nullptr;
            OFX::Private::gParamSuite = nullptr;
            OFX::Private::gEffectSuite = nullptr;
            OFX::Private::gPropSuite = nullptr;
            s_active = nullptr;
        }

        OfxImageEffectHandle effect_handle() noexcept {
            return reinterpret_cast<OfxImageEffectHandle>(&_effectHandle);
        }

        OfxImageEffectHandle descriptor_handle() noexcept {
            return reinterpret_cast<OfxImageEffectHandle>(
                &_descriptorEffectHandle);
        }

        OfxPropertySetHandle describe_context_handle() noexcept {
            return reinterpret_cast<OfxPropertySetHandle>(
                &_describeContextArguments);
        }

        void set_descriptor_module_path(
            const std::filesystem::path& modulePath) {
            _descriptorEffectHandle.properties.strings
                [kOfxPluginPropFilePath] = {modulePath.string()};
        }

        OfxHost* ofx_host() noexcept {
            return &_ofxHost;
        }

        void bind_effect(JuicerEffect& effect) noexcept {
            _effect = &effect;
        }

        void unbind_effect() noexcept {
            _effect = nullptr;
        }

        void set_time(double time) noexcept {
            _time = time;
        }

        void set_double_key(const char* name, double time, double value) {
            NativeParam& param = require_param(name, ParamKind::Double);
            ParamValue keyed = param.value_at(time);
            keyed.numbers[0] = value;
            param.keys[time] = std::move(keyed);
            append_trace("AUTHOR", name, time, value);
        }

        void set_int_value(const char* name, int value) {
            NativeParam& param = find_param(name);
            param.value.integer = value;
            append_trace("AUTHOR", name, _time, value);
        }

        void set_double_value(const char* name, double value) {
            NativeParam& param = require_param(name, ParamKind::Double);
            param.value.numbers[0] = value;
            append_trace("AUTHOR", name, _time, value);
        }

        void set_str_choice_value(const char* name, std::string value) {
            NativeParam& param = require_param(name, ParamKind::StrChoice);
            param.value.string = std::move(value);
        }

        void set_double3_value(const char* name, const std::array<double, 3>& value) {
            require_param(name, ParamKind::Double3).value.numbers = value;
        }

        void set_double2_value(const char* name, const std::array<double, 2>& value) {
            auto& numbers = require_param(name, ParamKind::Double2).value.numbers;
            std::copy(value.begin(), value.end(), numbers.begin());
        }

        double double_value(const char* name) {
            const NativeParam& param = require_param(name, ParamKind::Double);
            return param.value_at(_time).numbers[0];
        }

        int param_property_int(const char* name, const char* property) const {
            const NativeParam& param = find_param(name);
            int value = 0;
            if (get_property(param.properties.ints, property, 0, &value) !=
                kOfxStatOK) {
                throw std::runtime_error(
                    "native host parameter property missing: " +
                    std::string(name) + "/" + property);
            }
            return value;
        }

        void arm_property_reentry(
            const char* owningParam,
            const char* property,
            const char* callbackParam) {
            _propertyReentryOwningParam = owningParam;
            _propertyReentryProperty = property;
            _propertyReentryCallbackParam = callbackParam;
            _propertyReentryArmed = true;
        }

        std::uint64_t property_reentry_count() const noexcept {
            return _propertyReentryCount;
        }

        bool property_reentry_failed() const noexcept {
            return _propertyReentryFailed;
        }

        void dispatch_event(
            JuicerEffect& effect,
            const char* name,
            OFX::InstanceChangeReason reason) {
            std::ostringstream line;
            line << "EVENT name=" << name << " reason="
                 << static_cast<int>(reason) << " time=" << std::setprecision(17)
                 << _time;
            _trace.push_back(line.str());
            OFX::InstanceChangedArgs arguments{};
            arguments.reason = reason;
            arguments.time = _time;
            arguments.renderScale = {1.0, 1.0};
            effect.changedParam(arguments, name);
        }

        void render_stream_cases(OfxPlugin& plugin) {
            if (cudaSetDevice(0) != cudaSuccess) {
                throw std::runtime_error("stream test CUDA initialization failed");
            }
            DeviceFrame frame;
            std::uint64_t signature = 0;
            for (int streamCase = 0; streamCase < 3; ++streamCase) {
                const bool present = streamCase != 2;
                cudaStream_t stream = streamCase == 0 ? frame.stream() : nullptr;
                for (bool delayed : {false, true}) {
                    frame.reset();
                    configure_image_properties(_source.imageProperties, frame.source(), "stream-source");
                    configure_image_properties(_output.imageProperties, frame.destination(), "stream-output");
                    PropertyBag args;
                    args.doubles[kOfxPropTime] = {0.0};
                    args.doubles[kOfxImageEffectPropRenderScale] = {1.0, 1.0};
                    args.ints[kOfxImageEffectPropRenderWindow] = {0, 0, DeviceFrame::kWidth, DeviceFrame::kHeight};
                    args.ints[kOfxImageEffectPropCudaEnabled] = {1};
                    args.strings[kOfxImageEffectPropFieldToRender] = {kOfxImageFieldNone};
                    if (present) {
                        args.pointers[kOfxImageEffectPropCudaStream] = {stream};
                    }
                    begin_image_audit();
                    _observeStreamRelease = delayed;
                    _streamWasSupplied = present;
                    _releaseStream = stream;
                    _streamReleaseMatched = true;
                    _pendingStreamReleases = 0;
                    if (delayed && cudaLaunchHostFunc(stream, [](void*) {
                                       std::this_thread::sleep_for(std::chrono::milliseconds(500));
                                   },
                                                      nullptr) != cudaSuccess) {
                        throw std::runtime_error("stream delay enqueue failed");
                    }
                    const OfxStatus result = plugin.mainEntry(kOfxImageEffectActionRender, effect_handle(), reinterpret_cast<OfxPropertySetHandle>(&args), nullptr);
                    _observeStreamRelease = false;
                    const int releases = finish_image_audit("stream-contract");
                    if (cudaStreamSynchronize(stream) != cudaSuccess || result != kOfxStatOK || releases != 2) {
                        throw std::runtime_error("stream action or lease completion failed: " + std::to_string(streamCase));
                    }
                    if (!_streamReleaseMatched) {
                        throw std::runtime_error("stream release completion contract failed: " + std::to_string(streamCase));
                    }
                    const auto output = frame.collect_signature(true);
                    if (signature && output != signature) {
                        throw std::runtime_error("stream presence changed output");
                    }
                    signature = output;
                    _trace.push_back("STREAM_CASE case=" + std::to_string(streamCase) + " delayed=" + std::to_string(delayed) + " pending_releases=" + std::to_string(_pendingStreamReleases) + " passed=1");
                }
            }
        }

        struct RenderResult {
            bool succeeded = false;
            bool rejected = false;
            std::uint64_t outputSignature = 0;
            int releasedImages = 0;
            JuicerTestSupport::EffectTraceObserver::Snapshot state;
        };

        RenderResult render(
            JuicerEffect& effect,
            DeviceFrame& frame,
            std::string_view phase,
            bool expectRejection = false) {
            frame.reset();
            configure_image_properties(
                _source.imageProperties,
                frame.source(),
                "adapter-source");
            configure_image_properties(
                _output.imageProperties,
                frame.destination(),
                "adapter-output");
            begin_image_audit();
            _trace.push_back(
                "RENDER_BEGIN phase=" + std::string(phase) +
                " time=" + number_string(_time));
            RenderResult result{};
            std::exception_ptr unexpected;
            OFX::RenderArguments arguments{};
            arguments.time = _time;
            arguments.renderWindow = {
                0,
                0,
                DeviceFrame::kWidth,
                DeviceFrame::kHeight};
            arguments.renderScale = {1.0, 1.0};
            arguments.isEnabledCudaRender = true;
            arguments.pCudaStream = frame.stream();
            arguments.cudaStreamPropertyPresent = true;
            try {
                effect.render(arguments);
                result.succeeded = true;
            } catch (...) {
                result.rejected = true;
                if (!expectRejection) {
                    unexpected = std::current_exception();
                }
            }
            result.releasedImages = finish_image_audit(phase);
            if (unexpected) {
                std::rethrow_exception(unexpected);
            }
            result.outputSignature = frame.collect_signature(result.succeeded);
            result.state =
                JuicerTestSupport::EffectTraceObserver::snapshot(effect);
            _trace.push_back(
                "RENDER_END phase=" + std::string(phase) +
                " success=" + (result.succeeded ? "1" : "0") +
                " rejected=" + (result.rejected ? "1" : "0") +
                " release_count=" + std::to_string(result.releasedImages) +
                " exposure=" + number_string(result.state.pendingExposureEv) +
                " build=" + std::to_string(result.state.buildCounter) +
                " recipe=" + std::to_string(result.state.recipeHash) +
                " latch=" + std::to_string(result.state.latchSnapshotId) +
                " output=" + std::to_string(result.outputSignature));
            if (expectRejection != result.rejected) {
                throw std::runtime_error(
                    "adapter render rejection expectation mismatch in " +
                    std::string(phase));
            }
            if (result.releasedImages != 2) {
                throw std::runtime_error(
                    "adapter render did not release two OFX images in " +
                    std::string(phase));
            }
            return result;
        }

        std::size_t trace_size() const noexcept {
            return _trace.size();
        }

        const std::vector<std::string>& trace() const noexcept {
            return _trace;
        }

        std::uint64_t current_get_count() const noexcept {
            return _currentGetCount;
        }

        std::uint64_t at_time_get_count() const noexcept {
            return _atTimeGetCount;
        }

        std::uint64_t nested_event_count() const noexcept {
            return _nestedEventCount;
        }

        bool nested_event_failed() const noexcept {
            return _nestedEventFailed;
        }

        bool nested_contract_failed() const noexcept {
            return _nestedContractFailed;
        }

        const std::string& nested_contract_diagnostic() const noexcept {
            return _nestedContractDiagnostic;
        }

        void write_trace(const std::filesystem::path& path) const {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream output(path, std::ios::trunc);
            if (!output) {
                throw std::runtime_error("could not create adapter trace " + path.string());
            }
            output << "schema=film-juicer-pre-rust-adapter-trace-v1\n";
            output << "getter=current-value\n";
            output << "current_get_count=" << _currentGetCount << '\n';
            output << "at_time_get_count=" << _atTimeGetCount << '\n';
            output << "nested_event_count=" << _nestedEventCount << '\n';
            for (std::size_t index = 0; index < _trace.size(); ++index) {
                output << std::setw(6) << std::setfill('0') << index << ' '
                       << _trace[index] << '\n';
            }
        }

    private:
        struct ImageLeaseRecord {
            std::string label;
            int acquisitions = 0;
            int releases = 0;
        };

        struct NativeMutex {
            std::recursive_mutex value;
        };

        void set_image_audit_error(std::string message) {
            if (_imageAuditError.empty()) {
                _imageAuditError = std::move(message);
            }
        }

        void begin_image_audit() {
            if (_imageAuditActive) {
                throw std::runtime_error("adapter image audit already active");
            }
            _imageLeases.clear();
            _imageAuditError.clear();
            _imageAuditActive = true;
        }

        void record_image_acquisition(
            OfxPropertySetHandle handle,
            std::string_view label) {
            if (!_imageAuditActive || !handle) {
                set_image_audit_error(
                    "image acquisition occurred outside an active render");
                return;
            }
            ImageLeaseRecord& record = _imageLeases[handle];
            if (record.acquisitions != 0) {
                set_image_audit_error(
                    "duplicate image acquisition for " + std::string(label));
            }
            if (record.label.empty()) {
                record.label = label;
            } else if (record.label != label) {
                set_image_audit_error(
                    "one image handle was acquired for multiple clips");
            }
            ++record.acquisitions;
        }

        void record_image_release(OfxPropertySetHandle handle) {
            if (!_imageAuditActive || !handle) {
                set_image_audit_error(
                    "image release occurred outside an active render");
                return;
            }
            const auto found = _imageLeases.find(handle);
            if (found == _imageLeases.end()) {
                set_image_audit_error("unknown image handle was released");
                return;
            }
            if (found->second.releases != 0) {
                set_image_audit_error(
                    "duplicate image release for " + found->second.label);
            }
            ++found->second.releases;
            _trace.push_back("IMAGE_RELEASE clip=" + found->second.label);
        }

        int finish_image_audit(std::string_view phase) {
            if (!_imageAuditActive) {
                throw std::runtime_error("adapter image audit is not active");
            }
            int releases = 0;
            if (_imageLeases.size() != 2u) {
                set_image_audit_error(
                    "expected two distinct acquired image handles");
            }
            for (const auto& [handle, record] : _imageLeases) {
                (void)handle;
                releases += record.releases;
                if (record.acquisitions != 1 || record.releases != 1) {
                    set_image_audit_error(
                        "image lease imbalance for " + record.label +
                        " acquisitions=" +
                        std::to_string(record.acquisitions) + " releases=" +
                        std::to_string(record.releases));
                }
            }
            _imageAuditActive = false;
            if (!_imageAuditError.empty()) {
                throw std::runtime_error(
                    "adapter image audit failed in " + std::string(phase) +
                    ": " + _imageAuditError);
            }
            return releases;
        }

        static bool same_effect_state(
            const JuicerTestSupport::EffectTraceObserver::Snapshot& lhs,
            const JuicerTestSupport::EffectTraceObserver::Snapshot& rhs) {
            return lhs.pendingValid == rhs.pendingValid &&
                   lhs.pendingInvalid == rhs.pendingInvalid &&
                   lhs.pendingPrintRoute == rhs.pendingPrintRoute &&
                   std::bit_cast<std::uint64_t>(lhs.pendingExposureEv) ==
                       std::bit_cast<std::uint64_t>(rhs.pendingExposureEv) &&
                   lhs.pendingFullHash == rhs.pendingFullHash &&
                   lhs.buildCounter == rhs.buildCounter &&
                   lhs.recipeHash == rhs.recipeHash &&
                   lhs.lastParamHash == rhs.lastParamHash &&
                   lhs.paramEventsSuppressed == rhs.paramEventsSuppressed &&
                   lhs.latchValid == rhs.latchValid &&
                   lhs.latchSnapshotId == rhs.latchSnapshotId &&
                   lhs.latchFrameToken == rhs.latchFrameToken;
        }

        static std::string number_string(double value) {
            std::ostringstream output;
            output << std::setprecision(17) << value;
            return output.str();
        }

        template <typename T>
        void append_trace(
            const char* action,
            const char* name,
            double time,
            T value) {
            std::ostringstream line;
            line << action << " name=" << name << " time="
                 << std::setprecision(17) << time << " value=" << value;
            _trace.push_back(line.str());
        }

        NativeParam& add_param(
            const char* name,
            ParamKind kind,
            const char* type) {
            auto param = std::make_unique<NativeParam>();
            param->name = name;
            param->kind = kind;
            param->properties.strings[kOfxParamPropType] = {type};
            param->properties.strings[kOfxPropName] = {name};
            param->properties.strings[kOfxPropLabel] = {name};
            param->properties.strings[kOfxParamPropHint] = {""};
            param->properties.ints[kOfxParamPropEnabled] = {1};
            param->properties.ints[kOfxParamPropSecret] = {0};
            param->properties.owningParamName = name;
            param->properties.intWriteContext = this;
            param->properties.intWriteObserver = observe_property_int_write;
            auto [position, inserted] = _params.emplace(name, std::move(param));
            if (!inserted) {
                throw std::runtime_error("duplicate native parameter " + std::string(name));
            }
            return *position->second;
        }

        void add_double(const char* name, double value) {
            add_param(name, ParamKind::Double, kOfxParamTypeDouble)
                .value.numbers[0] = value;
        }

        void add_double2(const char* name, double x, double y) {
            auto& value =
                add_param(name, ParamKind::Double2, kOfxParamTypeDouble2D).value;
            value.numbers = {x, y, 0.0};
        }

        void add_double3(const char* name, double x, double y, double z) {
            auto& value =
                add_param(name, ParamKind::Double3, kOfxParamTypeDouble3D).value;
            value.numbers = {x, y, z};
        }

        void add_int(const char* name, int value) {
            add_param(name, ParamKind::Int, kOfxParamTypeInteger).value.integer = value;
        }

        void add_boolean(const char* name, bool value) {
            add_param(name, ParamKind::Boolean, kOfxParamTypeBoolean)
                .value.integer = value ? 1 : 0;
        }

        void add_choice(const char* name, int value) {
            add_param(name, ParamKind::Choice, kOfxParamTypeChoice)
                .value.integer = value;
        }

        void add_str_choice(const char* name, const char* value) {
            add_param(name, ParamKind::StrChoice, kOfxParamTypeStrChoice)
                .value.string = value;
        }

        void add_push_button(const char* name) {
            (void)add_param(name, ParamKind::PushButton, kOfxParamTypePushButton);
        }

        void configure_host() {
            _hostProperties.strings[kOfxPropType] = {
                kOfxTypeImageEffectHost};
            _hostProperties.strings[kOfxPropName] = {
                "FilmJuicerNativeFixture"};
            _hostProperties.strings[kOfxPropLabel] = {
                "Film-Juicer native fixture"};
            _hostProperties.ints[kOfxPropAPIVersion] = {1, 4};
            _hostProperties.ints[kOfxPropVersion] = {1, 0, 0};
            _hostProperties.strings[kOfxPropVersionLabel] = {"1.0"};
            _hostProperties.strings[kOfxImageEffectPropSupportedComponents] = {
                kOfxImageComponentRGBA};
            _hostProperties.strings[kOfxImageEffectPropSupportedContexts] = {
                kOfxImageEffectContextFilter};
            _hostProperties.strings[kOfxImageEffectPropSupportedPixelDepths] = {
                kOfxBitDepthFloat};
            _hostProperties.strings[kOfxImageEffectPropCudaRenderSupported] = {
                "true"};
            _hostProperties.strings[kOfxImageEffectPropCudaStreamSupported] = {
                "true"};
            _hostProperties.strings[kOfxImageEffectPropOpenCLRenderSupported] = {
                "false"};
            _hostProperties.strings[kOfxImageEffectPropMetalRenderSupported] = {
                "false"};
            _hostProperties.strings[kOfxImageEffectHostPropNativeOrigin] = {
                kOfxHostNativeOriginBottomLeft};
            _hostProperties.ints[kOfxImageEffectHostPropIsBackground] = {0};
            _hostProperties.ints[kOfxImageEffectPropSupportsOverlays] = {0};
            _hostProperties.ints[kOfxImageEffectPropSupportsMultiResolution] = {1};
            _hostProperties.ints[kOfxImageEffectPropSupportsTiles] = {0};
            _hostProperties.ints[kOfxImageEffectPropTemporalClipAccess] = {0};
            _hostProperties.ints[kOfxImageEffectPropSupportsMultipleClipDepths] = {0};
            _hostProperties.ints[kOfxImageEffectPropSupportsMultipleClipPARs] = {0};
            _hostProperties.ints[kOfxImageEffectPropSetableFrameRate] = {0};
            _hostProperties.ints[kOfxImageEffectPropSetableFielding] = {0};
            _hostProperties.ints[kOfxImageEffectInstancePropSequentialRender] = {0};
            _hostProperties.ints[kOfxParamHostPropSupportsStringAnimation] = {1};
            _hostProperties.ints[kOfxParamHostPropSupportsCustomInteract] = {
                0};
            _hostProperties.ints[kOfxParamHostPropSupportsChoiceAnimation] = {1};
            _hostProperties.ints[kOfxParamHostPropSupportsStrChoiceAnimation] = {1};
            _hostProperties.ints[kOfxParamHostPropSupportsBooleanAnimation] = {1};
            _hostProperties.ints[kOfxParamHostPropSupportsCustomAnimation] = {
                0};
            _hostProperties.ints[kOfxParamHostPropMaxParameters] = {1024};
            _hostProperties.ints[kOfxParamHostPropMaxPages] = {0};
            _hostProperties.ints[kOfxParamHostPropPageRowColumnCount] = {
                0,
                0};
            _ofxHost.host = reinterpret_cast<OfxPropertySetHandle>(
                &_hostProperties);
            _ofxHost.fetchSuite = fetch_suite;
        }

        void configure_effect() {
            _effectHandle.host = this;
            _effectHandle.properties.strings[kOfxPropType] = {
                kOfxTypeImageEffectInstance};
            _effectHandle.properties.strings[kOfxImageEffectPropContext] = {
                kOfxImageEffectContextFilter};
            _effectHandle.properties.ints[kOfxImageEffectInstancePropSequentialRender] = {0};
            _effectHandle.properties.pointers[kOfxPropInstanceData] = {nullptr};
            _effectHandle.properties.pointers[kOfxImageEffectPropPluginHandle] = {
                nullptr};
            _effectHandle.properties.doubles[kOfxImageEffectPropFrameRate] = {
                24.0};
        }

        void configure_descriptor() {
            _descriptorEffectHandle.host = this;
            PropertyBag& properties = _descriptorEffectHandle.properties;
            properties.strings[kOfxPropType] = {kOfxTypeImageEffect};
            properties.strings[kOfxPropLabel] = {""};
            properties.strings[kOfxPropShortLabel] = {""};
            properties.strings[kOfxPropLongLabel] = {""};
            properties.strings[kOfxImageEffectPluginPropGrouping] = {""};
            properties.strings[kOfxPluginPropFilePath] = {
                "FilmJuicerNativeFixture"};
            properties.strings[kOfxImageEffectPluginRenderThreadSafety] = {
                kOfxImageEffectRenderFullySafe};
            properties.strings[kOfxImageEffectPropSupportedContexts] = {};
            properties.strings[kOfxImageEffectPropSupportedPixelDepths] = {};
            properties.strings[kOfxImageEffectPropClipPreferencesSlaveParam] = {};
            properties.ints[kOfxImageEffectPluginPropSingleInstance] = {0};
            properties.ints[kOfxImageEffectPluginPropHostFrameThreading] = {0};
            properties.ints[kOfxImageEffectPropSupportsMultiResolution] = {1};
            properties.ints[kOfxImageEffectPropSupportsTiles] = {1};
            properties.ints[kOfxImageEffectPropTemporalClipAccess] = {0};
            properties.ints
                [kOfxImageEffectPluginPropFieldRenderTwiceAlways] = {1};
            properties.ints[kOfxImageEffectPropSupportsMultipleClipDepths] = {
                0};
            properties.ints[kOfxImageEffectPropSupportsMultipleClipPARs] = {
                0};
            properties.pointers[kOfxImageEffectPluginPropOverlayInteractV1] = {
                nullptr};
            properties.pointers[kOfxImageEffectPluginPropOverlayInteractV2] = {
                nullptr};
            _describeContextArguments.strings[kOfxImageEffectPropContext] = {
                kOfxImageEffectContextFilter};
        }

        void configure_clips() {
            configure_clip(_source, kOfxImageEffectSimpleSourceClipName);
            configure_clip(_output, kOfxImageEffectOutputClipName);
        }

        static void configure_clip(NativeClip& clip, const char* name) {
            clip.name = name;
            clip.properties.strings[kOfxPropType] = {kOfxTypeClip};
            clip.properties.strings[kOfxPropName] = {name};
            clip.properties.strings[kOfxPropLabel] = {name};
            clip.properties.strings[kOfxPropShortLabel] = {name};
            clip.properties.strings[kOfxPropLongLabel] = {name};
            clip.properties.strings[kOfxImageClipPropFieldExtraction] = {
                kOfxImageFieldDoubled};
            clip.properties.strings[kOfxImageEffectPropSupportedComponents] =
                {};
            clip.properties.strings[kOfxImageEffectPropPixelDepth] = {
                kOfxBitDepthFloat};
            clip.properties.strings[kOfxImageEffectPropComponents] = {
                kOfxImageComponentRGBA};
            clip.properties.strings[kOfxImageClipPropUnmappedPixelDepth] = {
                kOfxBitDepthFloat};
            clip.properties.strings[kOfxImageClipPropUnmappedComponents] = {
                kOfxImageComponentRGBA};
            clip.properties.strings[kOfxImageEffectPropPreMultiplication] = {
                kOfxImageOpaque};
            clip.properties.ints[kOfxImageClipPropConnected] = {1};
            clip.properties.ints[kOfxImageClipPropContinuousSamples] = {0};
            clip.properties.ints[kOfxImageEffectPropTemporalClipAccess] = {
                0};
            clip.properties.ints[kOfxImageClipPropOptional] = {0};
            clip.properties.ints[kOfxImageClipPropIsMask] = {0};
            clip.properties.ints[kOfxImageEffectPropSupportsTiles] = {1};
            clip.properties.doubles[kOfxImagePropPixelAspectRatio] = {1.0};
            clip.properties.doubles[kOfxImageEffectPropFrameRate] = {24.0};
            clip.properties.doubles[kOfxImageEffectPropFrameRange] = {0.0, 100.0};
        }

        static void configure_image_properties(
            PropertyBag& properties,
            void* data,
            const char* identifier) {
            properties = PropertyBag{};
            properties.ints[kOfxImagePropRowBytes] = {
                DeviceFrame::kPitch * static_cast<int>(sizeof(float))};
            properties.ints[kOfxImagePropBounds] = {
                0,
                0,
                DeviceFrame::kWidth,
                DeviceFrame::kHeight};
            properties.ints[kOfxImagePropRegionOfDefinition] = {
                0,
                0,
                DeviceFrame::kWidth,
                DeviceFrame::kHeight};
            properties.doubles[kOfxImagePropPixelAspectRatio] = {1.0};
            properties.doubles[kOfxImageEffectPropRenderScale] = {1.0, 1.0};
            properties.strings[kOfxImageEffectPropComponents] = {
                kOfxImageComponentRGBA};
            properties.strings[kOfxImageEffectPropPixelDepth] = {
                kOfxBitDepthFloat};
            properties.strings[kOfxImageEffectPropPreMultiplication] = {
                kOfxImageOpaque};
            properties.strings[kOfxImagePropField] = {kOfxImageFieldNone};
            properties.strings[kOfxImagePropUniqueIdentifier] = {identifier};
            properties.pointers[kOfxImagePropData] = {data};
        }

        void configure_params() {
            add_double(kParamExposure, 0.0);
            add_boolean(kParamCameraAutoExposure, false);
            add_choice(JuicerParams::kCameraFilmFormatPreset, 0);
            add_double(JuicerParams::kCameraFilmFormatMm, 35.0);
            add_choice(JuicerParams::kCameraMeteringMethod, 0);
            add_str_choice(
                JuicerParams::kFilmProfileKey,
                Spektrafilm::kDefaultFilmProfileKey);
            add_choice(kParamSpectralMode, 0);
            add_str_choice(
                JuicerParams::kPrintProfileKey,
                Spektrafilm::kDefaultPrintProfileKey);
            add_choice("ReferenceIlluminant", 0);
            add_choice("EnlargerIlluminant", 3);
            add_choice(
                JuicerParams::kInputColorSpace,
                ParamSnapshot{}.inputColorSpace);
            add_boolean(JuicerParams::kInputCctfDecoding, false);
            add_boolean(JuicerParams::kInputCompression, true);
            add_boolean(JuicerParams::kHanatos2025AdaptationWindow, true);
            add_boolean(JuicerParams::kHanatos2025AdaptationSurface, false);
            add_str_choice(
                JuicerParams::kParamScanRoute,
                Spektrafilm::scan_route_key(Spektrafilm::kDefaultScanRoute));
            add_double(JuicerParams::kFilmGammaFactor, 1.0);
            add_double(JuicerParams::kPrintGammaFactor, 1.0);
            add_choice(kParamOutputColorSpace, 0);
            add_boolean(kParamOutputCctfEncoding, true);
            add_boolean(JuicerParams::kOutputGamutCompression, true);

            add_boolean(JuicerParams::kDirCouplersActive, false);
            add_double(JuicerParams::kDirCouplersAmount, 1.0);
            add_double(JuicerParams::kDirCouplersInhibitionSameLayer, 1.0);
            add_double(JuicerParams::kDirCouplersInhibitionInterlayer, 1.0);
            add_double(JuicerParams::kDirCouplersDiffusionSizeUm, 20.0);
            add_double3(JuicerParams::kDirCouplersLangmuirDonorKRgb, 1.0, 1.0, 1.0);
            add_double3(JuicerParams::kDirCouplersLangmuirReceiverKRgb, 1.0, 1.0, 1.0);
            add_double(JuicerParams::kDirCouplersDiffusionTailUm, 200.0);
            add_double(JuicerParams::kDirCouplersDiffusionTailWeight, 0.03);
            add_boolean(JuicerParams::kDirCouplersGammaUseStock, true);
            add_double3(JuicerParams::kDirCouplersGammaSameLayerRgb, 0.341, 0.324, 0.273);
            add_double2(JuicerParams::kDirCouplersGammaInterlayerRToGb, 0.355, 0.305);
            add_double2(JuicerParams::kDirCouplersGammaInterlayerGToRb, 0.154, 0.358);
            add_double2(JuicerParams::kDirCouplersGammaInterlayerBToRg, 0.171, 0.225);

            add_double(JuicerParams::kScannerLensBlurSigmaPx, 0.0);
            add_double2(JuicerParams::kScannerUnsharpMask, 0.0, 0.0);
            add_boolean(JuicerParams::kScannerBlackCorrection, false);
            add_boolean(JuicerParams::kScannerWhiteCorrection, false);
            add_double(JuicerParams::kScannerBlackLevel, 0.01);
            add_double(JuicerParams::kScannerWhiteLevel, 0.98);
            add_int(JuicerParams::kScannerLutResolution, 17);

            add_double("PrintExposure", 1.0);
            add_double("PrintPreflash", 0.0);
            add_boolean("PrintExposureCompensation", true);
            add_double("EnlargerY", 0.0);
            add_double("EnlargerM", 0.0);
            add_double("EnlargerC", 0.0);

            add_boolean(JuicerParams::kHalationActive, false);
            add_double(JuicerParams::kHalationScatterAmount, 1.0);
            add_double(JuicerParams::kHalationScatterSpatialScale, 1.0);
            add_double(JuicerParams::kHalationAmount, 1.0);
            add_double(JuicerParams::kHalationSpatialScale, 1.0);

            add_boolean(JuicerParams::kGrainActive, false);
            add_boolean(JuicerParams::kGrainSublayersActive, true);
            add_choice(JuicerParams::kGrainPreset, 1);
            add_double(JuicerParams::kGrainParticleAreaUm2, 0.25);
            add_double(JuicerParams::kGrainAmplitude, -1.20);
            add_double(JuicerParams::kGrainSharpness, 0.5);
            add_double(JuicerParams::kGrainChroma, 0.3);
            add_double(JuicerParams::kGrainTexture, 0.55);
            add_double(JuicerParams::kGrainParticleScaleMaster, 1.48);
            add_double(JuicerParams::kGrainParticleScaleLayersMaster, 1.922);
            add_double(JuicerParams::kGrainDensityMinMaster, 0.08);
            add_double(JuicerParams::kGrainUniformityMaster, 0.97);
            add_double3(JuicerParams::kGrainParticleScale, 1.48, 1.48, 1.48);
            add_double3(JuicerParams::kGrainParticleScaleLayers, 1.922, 1.922, 1.922);
            add_double3(JuicerParams::kGrainDensityMin, 0.08, 0.08, 0.08);
            add_double3(JuicerParams::kGrainUniformity, 0.97, 0.97, 0.97);
            add_double(JuicerParams::kGrainBlur, 0.50);
            add_double(JuicerParams::kGrainBlurDyeCloudsUm, 1.0);
            add_double(JuicerParams::kGrainSizeMixWeight, 0.226);
            add_double(JuicerParams::kGrainSizeMixWeightMid, 0.0);
            add_double(JuicerParams::kGrainSizeMixScale, 19.0);
            add_double(JuicerParams::kGrainClumpTemporalMix, 0.30);
            add_double(JuicerParams::kGrainClumpMorphPeriodSec, 8.0);
            add_choice(JuicerParams::kGrainDebugView, 0);
            add_double2(JuicerParams::kGrainMicroStructure, 60.0, 170.0);
            add_push_button(JuicerParams::kGrainResetAdvanced);
            add_double(JuicerParams::kGateWeaveAmount, 0.0);
            add_double(JuicerParams::kFilmDustAmount, 0.0);
            add_double(JuicerParams::kGateDustAmount, 0.0);
            add_double(JuicerParams::kFilmScratchAmount, 0.0);
            add_double(JuicerParams::kGateScratchAmount, 0.0);

            configure_diffusion_params("Camera", false);
            configure_diffusion_params("Print", false);
            add_boolean(JuicerParams::kGlareActive, true);
            add_double(JuicerParams::kGlarePercent, 0.03);
            add_double(JuicerParams::kGlareRoughness, 0.7);
            add_double(JuicerParams::kGlareBlurSigmaPx, 0.5);
            add_double(JuicerParams::kPrintShadowCompensationFactor, 0.0);
            add_double(JuicerParams::kPrintShadowCompensationDensity, 1.2);
            add_double(JuicerParams::kPrintShadowCompensationTransition, 0.3);

            find_param(JuicerParams::kGrainPreset)
                .properties.strings[kOfxPropLabel] = {"Grain Preset"};
            find_param(JuicerParams::kGrainChroma)
                .properties.strings[kOfxParamPropHint] = {
                "0 = achromatic, 1 = independent RGB grain."};
        }

        void configure_diffusion_params(const char* prefix, bool enabled) {
            const bool camera = std::string_view(prefix) == "Camera";
            const char* enabledName = camera
                                          ? JuicerParams::kCameraDiffusionEnabled
                                          : JuicerParams::kPrintDiffusionEnabled;
            const char* familyName = camera
                                         ? JuicerParams::kCameraDiffusionFamily
                                         : JuicerParams::kPrintDiffusionFamily;
            add_boolean(enabledName, enabled);
            add_choice(familyName, 1);
            const std::array<const char*, 9> names = camera
                                                         ? std::array<const char*, 9>{
                                                               JuicerParams::kCameraDiffusionStrength,
                                                               JuicerParams::kCameraDiffusionSpatialScale,
                                                               JuicerParams::kCameraDiffusionHaloWarmth,
                                                               JuicerParams::kCameraDiffusionCoreIntensity,
                                                               JuicerParams::kCameraDiffusionCoreSize,
                                                               JuicerParams::kCameraDiffusionHaloIntensity,
                                                               JuicerParams::kCameraDiffusionHaloSize,
                                                               JuicerParams::kCameraDiffusionBloomIntensity,
                                                               JuicerParams::kCameraDiffusionBloomSize}
                                                         : std::array<const char*, 9>{JuicerParams::kPrintDiffusionStrength, JuicerParams::kPrintDiffusionSpatialScale, JuicerParams::kPrintDiffusionHaloWarmth, JuicerParams::kPrintDiffusionCoreIntensity, JuicerParams::kPrintDiffusionCoreSize, JuicerParams::kPrintDiffusionHaloIntensity, JuicerParams::kPrintDiffusionHaloSize, JuicerParams::kPrintDiffusionBloomIntensity, JuicerParams::kPrintDiffusionBloomSize};
            const std::array<double, 9> defaults{
                0.5,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0};
            for (std::size_t index = 0; index < names.size(); ++index) {
                add_double(names[index], defaults[index]);
            }
        }

        NativeParam& find_param(const char* name) {
            const auto found = _params.find(name ? name : "");
            if (found == _params.end()) {
                throw std::runtime_error(
                    "native host parameter missing: " +
                    std::string(name ? name : "<null>"));
            }
            return *found->second;
        }

        const NativeParam& find_param(const char* name) const {
            const auto found = _params.find(name ? name : "");
            if (found == _params.end()) {
                throw std::runtime_error(
                    "native host parameter missing: " +
                    std::string(name ? name : "<null>"));
            }
            return *found->second;
        }

        NativeParam& require_param(const char* name, ParamKind kind) {
            NativeParam& param = find_param(name);
            if (param.kind != kind) {
                throw std::runtime_error("native host parameter type mismatch");
            }
            return param;
        }

        static NativeHost& active() {
            if (!s_active) {
                throw std::runtime_error("native adapter host is not active");
            }
            return *s_active;
        }

        static const void* fetch_suite(
            OfxPropertySetHandle,
            const char* name,
            int version) {
            NativeHost& host = active();
            if (!name || version != 1) {
                return nullptr;
            }
            const std::string_view suiteName(name);
            if (suiteName == kOfxPropertySuite) {
                return &host._propertySuite;
            }
            if (suiteName == kOfxImageEffectSuite) {
                return &host._effectSuite;
            }
            if (suiteName == kOfxParameterSuite) {
                return &host._parameterSuite;
            }
            if (suiteName == kOfxMemorySuite) {
                return &host._memorySuite;
            }
            if (suiteName == kOfxMultiThreadSuite) {
                return &host._multiThreadSuite;
            }
            if (suiteName == kOfxMessageSuite) {
                return &host._messageSuite;
            }
            return nullptr;
        }

        static OfxStatus memory_allocate(
            void*,
            std::size_t bytes,
            void** output) {
            if (!output) {
                return kOfxStatErrBadHandle;
            }
            *output = std::malloc(bytes == 0 ? 1u : bytes);
            return *output ? kOfxStatOK : kOfxStatErrMemory;
        }

        static OfxStatus memory_free(void* allocation) {
            std::free(allocation);
            return kOfxStatOK;
        }

        static OfxStatus run_threads(
            OfxThreadFunctionV1 function,
            unsigned int threadCount,
            void* context) {
            if (!function) {
                return kOfxStatErrBadHandle;
            }
            const unsigned int actualCount = threadCount == 0 ? 1u : threadCount;
            for (unsigned int index = 0; index < actualCount; ++index) {
                function(index, actualCount, context);
            }
            return kOfxStatOK;
        }

        static OfxStatus thread_cpu_count(unsigned int* count) {
            if (!count) {
                return kOfxStatErrBadHandle;
            }
            *count = 1;
            return kOfxStatOK;
        }

        static OfxStatus thread_index(unsigned int* index) {
            if (!index) {
                return kOfxStatErrBadHandle;
            }
            *index = 0;
            return kOfxStatOK;
        }

        static int is_spawned_thread() {
            return 0;
        }

        static OfxStatus mutex_create(OfxMutexHandle* output, int lockCount) {
            if (!output || lockCount < 0) {
                return kOfxStatErrBadHandle;
            }
            auto mutex = std::make_unique<NativeMutex>();
            for (int index = 0; index < lockCount; ++index) {
                mutex->value.lock();
            }
            *output = reinterpret_cast<OfxMutexHandle>(mutex.release());
            return kOfxStatOK;
        }

        static OfxStatus mutex_destroy(const OfxMutexHandle handle) {
            delete reinterpret_cast<NativeMutex*>(handle);
            return kOfxStatOK;
        }

        static OfxStatus mutex_lock(const OfxMutexHandle handle) {
            if (!handle) {
                return kOfxStatErrBadHandle;
            }
            reinterpret_cast<NativeMutex*>(handle)->value.lock();
            return kOfxStatOK;
        }

        static OfxStatus mutex_unlock(const OfxMutexHandle handle) {
            if (!handle) {
                return kOfxStatErrBadHandle;
            }
            reinterpret_cast<NativeMutex*>(handle)->value.unlock();
            return kOfxStatOK;
        }

        static OfxStatus mutex_try_lock(const OfxMutexHandle handle) {
            if (!handle) {
                return kOfxStatErrBadHandle;
            }
            return reinterpret_cast<NativeMutex*>(handle)->value.try_lock()
                       ? kOfxStatOK
                       : kOfxStatFailed;
        }

        static void observe_property_int_write(
            void* context,
            const char* owningParam,
            const char* property,
            int value) {
            auto& host = *static_cast<NativeHost*>(context);
            host._trace.push_back(
                "PROPERTY_SET_INT param=" + std::string(owningParam) +
                " property=" + property + " value=" +
                std::to_string(value));
            if (!host._propertyReentryArmed ||
                host._propertyReentryOwningParam != owningParam ||
                host._propertyReentryProperty != property) {
                return;
            }

            host._propertyReentryArmed = false;
            ++host._propertyReentryCount;
            host._trace.push_back(
                "EVENT_REENTRANT boundary=PROPERTY_SET_INT callback=" +
                host._propertyReentryCallbackParam);
            try {
                if (!host._effect) {
                    throw std::runtime_error(
                        "property reentry has no bound effect");
                }
                OFX::InstanceChangedArgs arguments{};
                arguments.reason = OFX::eChangePluginEdit;
                arguments.time = host._time;
                arguments.renderScale = {1.0, 1.0};
                host._effect->changedParam(
                    arguments,
                    host._propertyReentryCallbackParam);
            } catch (...) {
                host._propertyReentryFailed = true;
            }
        }

        static OfxStatus effect_properties(
            OfxImageEffectHandle handle,
            OfxPropertySetHandle* output) {
            if (!handle || !output) {
                return kOfxStatErrBadHandle;
            }
            *output = reinterpret_cast<OfxPropertySetHandle>(
                &reinterpret_cast<NativeEffectHandle*>(handle)->properties);
            return kOfxStatOK;
        }

        static OfxStatus effect_parameters(
            OfxImageEffectHandle handle,
            OfxParamSetHandle* output) {
            if (!handle || !output) {
                return kOfxStatErrBadHandle;
            }
            *output = reinterpret_cast<OfxParamSetHandle>(
                reinterpret_cast<NativeEffectHandle*>(handle)->host);
            return kOfxStatOK;
        }

        static OfxStatus clip_define(
            OfxImageEffectHandle,
            const char* name,
            OfxPropertySetHandle* properties) {
            if (!name || !properties) {
                return kOfxStatErrBadHandle;
            }
            NativeHost& host = active();
            NativeClip* selected = nullptr;
            if (std::string_view(name) == kOfxImageEffectSimpleSourceClipName) {
                selected = &host._source;
            } else if (std::string_view(name) == kOfxImageEffectOutputClipName) {
                selected = &host._output;
            }
            if (!selected) {
                host._trace.push_back(
                    "DESCRIBE_CLIP_REJECT name=" + std::string(name));
                return kOfxStatErrUnknown;
            }
            host._trace.push_back(
                "DESCRIBE_CLIP name=" + std::string(name));
            *properties = reinterpret_cast<OfxPropertySetHandle>(
                &selected->properties);
            return kOfxStatOK;
        }

        static OfxStatus clip_get_handle(
            OfxImageEffectHandle,
            const char* name,
            OfxImageClipHandle* clip,
            OfxPropertySetHandle* properties) {
            if (!name || !clip) {
                return kOfxStatErrBadHandle;
            }
            NativeHost& host = active();
            NativeClip* selected = nullptr;
            if (std::string_view(name) == kOfxImageEffectSimpleSourceClipName) {
                selected = &host._source;
            } else if (std::string_view(name) == kOfxImageEffectOutputClipName) {
                selected = &host._output;
            }
            if (!selected) {
                return kOfxStatErrUnknown;
            }
            *clip = reinterpret_cast<OfxImageClipHandle>(selected);
            if (properties) {
                *properties = reinterpret_cast<OfxPropertySetHandle>(
                    &selected->properties);
            }
            return kOfxStatOK;
        }

        static OfxStatus clip_properties(
            OfxImageClipHandle clip,
            OfxPropertySetHandle* output) {
            if (!clip || !output) {
                return kOfxStatErrBadHandle;
            }
            *output = reinterpret_cast<OfxPropertySetHandle>(
                &reinterpret_cast<NativeClip*>(clip)->properties);
            return kOfxStatOK;
        }

        static OfxStatus clip_get_image(
            OfxImageClipHandle clip,
            OfxTime time,
            const OfxRectD*,
            OfxPropertySetHandle* output) {
            if (!clip || !output) {
                return kOfxStatErrBadHandle;
            }
            NativeHost& host = active();
            host._trace.push_back(
                "IMAGE_GET clip=" + reinterpret_cast<NativeClip*>(clip)->name +
                " time=" + number_string(time));
            *output = reinterpret_cast<OfxPropertySetHandle>(
                &reinterpret_cast<NativeClip*>(clip)->imageProperties);
            host.record_image_acquisition(
                *output,
                reinterpret_cast<NativeClip*>(clip)->name);
            return kOfxStatOK;
        }

        static OfxStatus clip_release_image(OfxPropertySetHandle image) {
            NativeHost& host = active();
            if (host._observeStreamRelease) {
                const auto completion = cudaStreamQuery(host._releaseStream);
                // A supplied stream may already have completed; absence must complete before either lease is released.
                host._streamReleaseMatched &= completion == cudaSuccess || (host._streamWasSupplied && completion == cudaErrorNotReady);
                host._pendingStreamReleases += completion == cudaErrorNotReady ? 1 : 0;
            }
            host.record_image_release(image);
            return kOfxStatOK;
        }

        static OfxStatus clip_region(
            OfxImageClipHandle,
            OfxTime,
            OfxRectD* bounds) {
            if (!bounds) {
                return kOfxStatErrBadHandle;
            }
            *bounds = {
                0.0,
                0.0,
                static_cast<double>(DeviceFrame::kWidth),
                static_cast<double>(DeviceFrame::kHeight)};
            return kOfxStatOK;
        }

        static int effect_abort(OfxImageEffectHandle) {
            return 0;
        }

        static OfxStatus parameter_set_properties(
            OfxParamSetHandle,
            OfxPropertySetHandle* output) {
            if (!output) {
                return kOfxStatErrBadHandle;
            }
            *output = reinterpret_cast<OfxPropertySetHandle>(
                &active()._paramSetProperties);
            return kOfxStatOK;
        }

        static OfxStatus parameter_define(
            OfxParamSetHandle,
            const char* type,
            const char* name,
            OfxPropertySetHandle* properties) {
            if (!type || !name || !properties) {
                return kOfxStatErrBadHandle;
            }
            auto [position, inserted] =
                active()._descriptorParams.try_emplace(name);
            if (!inserted) {
                active()._trace.push_back(
                    "DESCRIBE_PARAM_DUPLICATE name=" + std::string(name) +
                    " type=" + type);
                return kOfxStatErrExists;
            }
            active()._trace.push_back(
                "DESCRIBE_PARAM name=" + std::string(name) +
                " type=" + type);
            PropertyBag& descriptor = position->second;
            descriptor.strings[kOfxParamPropType] = {type};
            descriptor.strings[kOfxPropName] = {name};
            descriptor.strings[kOfxPropLabel] = {name};
            descriptor.strings[kOfxParamPropHint] = {""};
            descriptor.strings[kOfxParamPropChoiceOption] = {};
            descriptor.strings[kOfxParamPropChoiceEnum] = {};
            descriptor.ints[kOfxParamPropEnabled] = {1};
            descriptor.ints[kOfxParamPropSecret] = {0};
            *properties = reinterpret_cast<OfxPropertySetHandle>(
                &descriptor);
            return kOfxStatOK;
        }

        static OfxStatus parameter_get_handle(
            OfxParamSetHandle,
            const char* name,
            OfxParamHandle* output,
            OfxPropertySetHandle* properties) {
            if (!name || !output) {
                return kOfxStatErrBadHandle;
            }
            NativeParam& param = active().find_param(name);
            *output = reinterpret_cast<OfxParamHandle>(&param);
            if (properties) {
                *properties = reinterpret_cast<OfxPropertySetHandle>(
                    &param.properties);
            }
            return kOfxStatOK;
        }

        static OfxStatus parameter_properties(
            OfxParamHandle handle,
            OfxPropertySetHandle* output) {
            if (!handle || !output) {
                return kOfxStatErrBadHandle;
            }
            *output = reinterpret_cast<OfxPropertySetHandle>(
                &reinterpret_cast<NativeParam*>(handle)->properties);
            return kOfxStatOK;
        }

        static OfxStatus read_param_value(
            OfxParamHandle handle,
            double time,
            bool atTime,
            va_list arguments) {
            if (!handle) {
                return kOfxStatErrBadHandle;
            }
            NativeHost& host = active();
            NativeParam& param = *reinterpret_cast<NativeParam*>(handle);
            const ParamValue& value = param.value_at(time);
            if (atTime) {
                ++host._atTimeGetCount;
            } else {
                ++host._currentGetCount;
            }
            std::string describedValue;
            switch (param.kind) {
                case ParamKind::Double:
                    describedValue = number_string(value.numbers[0]);
                    break;
                case ParamKind::Double2:
                    describedValue = number_string(value.numbers[0]) + "," +
                                     number_string(value.numbers[1]);
                    break;
                case ParamKind::Double3:
                    describedValue = number_string(value.numbers[0]) + "," +
                                     number_string(value.numbers[1]) + "," +
                                     number_string(value.numbers[2]);
                    break;
                case ParamKind::Int:
                case ParamKind::Boolean:
                case ParamKind::Choice:
                    describedValue = std::to_string(value.integer);
                    break;
                case ParamKind::StrChoice:
                    describedValue = value.string;
                    break;
                case ParamKind::PushButton:
                    describedValue = "<push>";
                    break;
            }
            host._trace.push_back(
                std::string(atTime ? "GET_AT_TIME" : "GET_CURRENT") +
                " name=" + param.name + " time=" + number_string(time) +
                " value=" + describedValue);
            switch (param.kind) {
                case ParamKind::Double:
                    *va_arg(arguments, double*) = value.numbers[0];
                    break;
                case ParamKind::Double2:
                    *va_arg(arguments, double*) = value.numbers[0];
                    *va_arg(arguments, double*) = value.numbers[1];
                    break;
                case ParamKind::Double3:
                    *va_arg(arguments, double*) = value.numbers[0];
                    *va_arg(arguments, double*) = value.numbers[1];
                    *va_arg(arguments, double*) = value.numbers[2];
                    break;
                case ParamKind::Int:
                case ParamKind::Boolean:
                case ParamKind::Choice:
                    *va_arg(arguments, int*) = value.integer;
                    break;
                case ParamKind::StrChoice:
                    *va_arg(arguments, char**) =
                        const_cast<char*>(value.string.c_str());
                    break;
                case ParamKind::PushButton:
                    return kOfxStatErrUnsupported;
            }
            return kOfxStatOK;
        }

        static OfxStatus parameter_get_value(OfxParamHandle handle, ...) {
            va_list arguments;
            va_start(arguments, handle);
            const OfxStatus status = read_param_value(
                handle,
                active()._time,
                false,
                arguments);
            va_end(arguments);
            return status;
        }

        static OfxStatus parameter_get_value_at_time(
            OfxParamHandle handle,
            OfxTime time,
            ...) {
            va_list arguments;
            va_start(arguments, time);
            const OfxStatus status = read_param_value(
                handle,
                time,
                true,
                arguments);
            va_end(arguments);
            return status;
        }

        static OfxStatus write_param_value(
            OfxParamHandle handle,
            va_list arguments) {
            if (!handle) {
                return kOfxStatErrBadHandle;
            }
            NativeHost& host = active();
            NativeParam& param = *reinterpret_cast<NativeParam*>(handle);
            switch (param.kind) {
                case ParamKind::Double:
                    param.value.numbers[0] = va_arg(arguments, double);
                    break;
                case ParamKind::Double2:
                    param.value.numbers[0] = va_arg(arguments, double);
                    param.value.numbers[1] = va_arg(arguments, double);
                    break;
                case ParamKind::Double3:
                    param.value.numbers[0] = va_arg(arguments, double);
                    param.value.numbers[1] = va_arg(arguments, double);
                    param.value.numbers[2] = va_arg(arguments, double);
                    break;
                case ParamKind::Int:
                case ParamKind::Boolean:
                case ParamKind::Choice:
                    param.value.integer = va_arg(arguments, int);
                    break;
                case ParamKind::StrChoice: {
                    const char* value = va_arg(arguments, const char*);
                    param.value.string = value ? value : "";
                    break;
                }
                case ParamKind::PushButton:
                    break;
            }
            host._trace.push_back(
                "SET_CURRENT name=" + param.name +
                " time=" + number_string(host._time));
            host.dispatch_nested_event(param.name);
            return kOfxStatOK;
        }

        static OfxStatus parameter_set_value(OfxParamHandle handle, ...) {
            va_list arguments;
            va_start(arguments, handle);
            const OfxStatus status = write_param_value(handle, arguments);
            va_end(arguments);
            return status;
        }

        static OfxStatus parameter_set_value_at_time(
            OfxParamHandle handle,
            OfxTime time,
            ...) {
            if (!handle) {
                return kOfxStatErrBadHandle;
            }
            va_list arguments;
            va_start(arguments, time);
            NativeParam& param = *reinterpret_cast<NativeParam*>(handle);
            const ParamValue previous = param.value;
            const OfxStatus status = write_param_value(handle, arguments);
            va_end(arguments);
            if (status == kOfxStatOK) {
                param.keys[time] = param.value;
                param.value = previous;
            }
            return status;
        }

        void dispatch_nested_event(const std::string& name) noexcept {
            try {
                if (!_effect || _insideNestedEvent) {
                    return;
                }
                const auto before =
                    JuicerTestSupport::EffectTraceObserver::snapshot(*_effect);
                ++_nestedEventCount;
                _trace.push_back(
                    "EVENT_NESTED name=" + name +
                    " reason=" +
                    std::to_string(static_cast<int>(OFX::eChangePluginEdit)) +
                    " time=" + number_string(_time));
                _insideNestedEvent = true;
                OFX::InstanceChangedArgs arguments{};
                arguments.reason = OFX::eChangePluginEdit;
                arguments.time = _time;
                arguments.renderScale = {1.0, 1.0};
                _effect->changedParam(arguments, name);
                const auto after =
                    JuicerTestSupport::EffectTraceObserver::snapshot(*_effect);
                if (!before.paramEventsSuppressed) {
                    _nestedContractFailed = true;
                    _nestedContractDiagnostic =
                        "nested callback was dispatched without suppression";
                } else if (!same_effect_state(before, after)) {
                    _nestedContractFailed = true;
                    _nestedContractDiagnostic =
                        "suppressed nested callback changed effect state";
                }
            } catch (...) {
                _nestedEventFailed = true;
            }
            _insideNestedEvent = false;
        }

        static OfxStatus message(
            void*,
            const char* messageType,
            const char* messageId,
            const char* format,
            ...) {
            NativeHost& host = active();
            host._trace.push_back(
                "MESSAGE type=" + std::string(messageType ? messageType : "") +
                " id=" + std::string(messageId ? messageId : "") +
                " text=" + std::string(format ? format : ""));
            return kOfxStatOK;
        }

        void configure_suites() {
            _propertySuite.propSetPointer = property_set_pointer;
            _propertySuite.propSetString = property_set_string;
            _propertySuite.propSetDouble = property_set_double;
            _propertySuite.propSetInt = property_set_int;
            _propertySuite.propSetPointerN = property_set_pointer_n;
            _propertySuite.propSetStringN = property_set_string_n;
            _propertySuite.propSetDoubleN = property_set_double_n;
            _propertySuite.propSetIntN = property_set_int_n;
            _propertySuite.propGetPointer = property_get_pointer;
            _propertySuite.propGetString = property_get_string;
            _propertySuite.propGetDouble = property_get_double;
            _propertySuite.propGetInt = property_get_int;
            _propertySuite.propGetPointerN = property_get_pointer_n;
            _propertySuite.propGetStringN = property_get_string_n;
            _propertySuite.propGetDoubleN = property_get_double_n;
            _propertySuite.propGetIntN = property_get_int_n;
            _propertySuite.propReset = property_reset;
            _propertySuite.propGetDimension = property_dimension;

            _effectSuite.getPropertySet = effect_properties;
            _effectSuite.getParamSet = effect_parameters;
            _effectSuite.clipDefine = clip_define;
            _effectSuite.clipGetHandle = clip_get_handle;
            _effectSuite.clipGetPropertySet = clip_properties;
            _effectSuite.clipGetImage = clip_get_image;
            _effectSuite.clipReleaseImage = clip_release_image;
            _effectSuite.clipGetRegionOfDefinition = clip_region;
            _effectSuite.abort = effect_abort;

            _parameterSuite.paramSetGetPropertySet = parameter_set_properties;
            _parameterSuite.paramDefine = parameter_define;
            _parameterSuite.paramGetHandle = parameter_get_handle;
            _parameterSuite.paramGetPropertySet = parameter_properties;
            _parameterSuite.paramGetValue = parameter_get_value;
            _parameterSuite.paramGetValueAtTime = parameter_get_value_at_time;
            _parameterSuite.paramSetValue = parameter_set_value;
            _parameterSuite.paramSetValueAtTime = parameter_set_value_at_time;
            _memorySuite.memoryAlloc = memory_allocate;
            _memorySuite.memoryFree = memory_free;
            _multiThreadSuite.multiThread = run_threads;
            _multiThreadSuite.multiThreadNumCPUs = thread_cpu_count;
            _multiThreadSuite.multiThreadIndex = thread_index;
            _multiThreadSuite.multiThreadIsSpawnedThread = is_spawned_thread;
            _multiThreadSuite.mutexCreate = mutex_create;
            _multiThreadSuite.mutexDestroy = mutex_destroy;
            _multiThreadSuite.mutexLock = mutex_lock;
            _multiThreadSuite.mutexUnLock = mutex_unlock;
            _multiThreadSuite.mutexTryLock = mutex_try_lock;
            _messageSuite.message = message;
        }

        inline static NativeHost* s_active = nullptr;
        OfxPropertySuiteV1 _propertySuite{};
        OfxImageEffectSuiteV1 _effectSuite{};
        OfxParameterSuiteV1 _parameterSuite{};
        OfxMemorySuiteV1 _memorySuite{};
        OfxMultiThreadSuiteV1 _multiThreadSuite{};
        OfxMessageSuiteV1 _messageSuite{};
        PropertyBag _hostProperties;
        OfxHost _ofxHost{};
        NativeEffectHandle _effectHandle{};
        NativeEffectHandle _descriptorEffectHandle{};
        PropertyBag _describeContextArguments;
        PropertyBag _paramSetProperties;
        NativeClip _source;
        NativeClip _output;
        std::map<std::string, std::unique_ptr<NativeParam>> _params;
        std::map<std::string, PropertyBag> _descriptorParams;
        JuicerEffect* _effect = nullptr;
        double _time = 0.0;
        std::map<OfxPropertySetHandle, ImageLeaseRecord> _imageLeases;
        std::string _imageAuditError;
        bool _imageAuditActive = false;
        bool _observeStreamRelease = false;
        bool _streamWasSupplied = false;
        bool _streamReleaseMatched = true;
        int _pendingStreamReleases = 0;
        cudaStream_t _releaseStream = nullptr;
        std::vector<std::string> _trace;
        std::uint64_t _currentGetCount = 0;
        std::uint64_t _atTimeGetCount = 0;
        std::uint64_t _nestedEventCount = 0;
        bool _insideNestedEvent = false;
        bool _nestedEventFailed = false;
        bool _nestedContractFailed = false;
        std::string _nestedContractDiagnostic;
        std::string _propertyReentryOwningParam;
        std::string _propertyReentryProperty;
        std::string _propertyReentryCallbackParam;
        std::uint64_t _propertyReentryCount = 0;
        bool _propertyReentryArmed = false;
        bool _propertyReentryFailed = false;
    };

    void require(bool condition, const std::string& message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    bool exposure_matches(
        const NativeHost::RenderResult& result,
        double expected) {
        return result.state.pendingValid &&
               !result.state.pendingInvalid &&
               std::bit_cast<std::uint64_t>(result.state.pendingExposureEv) ==
                   std::bit_cast<std::uint64_t>(expected);
    }

    bool has_valid_latch(const NativeHost::RenderResult& result) {
        return result.state.latchValid &&
               result.state.latchSnapshotId != 0;
    }

    bool double_matches(double actual, double expected) {
        return std::bit_cast<std::uint64_t>(actual) ==
               std::bit_cast<std::uint64_t>(expected);
    }

    struct UiControlObservation {
        std::string_view param;
        bool expectsSecret;
    };

    // Only the controls written by updateDiffusionControlState/updateGammaControlState.
    constexpr std::array<UiControlObservation, 27> kRouteUiControls{{{JuicerParams::kCameraDiffusionEnabled, true},
                                                                     {JuicerParams::kCameraDiffusionFamily, true},
                                                                     {JuicerParams::kCameraDiffusionStrength, true},
                                                                     {JuicerParams::kCameraDiffusionSpatialScale, true},
                                                                     {JuicerParams::kCameraDiffusionHaloWarmth, true},
                                                                     {JuicerParams::kCameraDiffusionCoreIntensity, true},
                                                                     {JuicerParams::kCameraDiffusionCoreSize, true},
                                                                     {JuicerParams::kCameraDiffusionHaloIntensity, true},
                                                                     {JuicerParams::kCameraDiffusionHaloSize, true},
                                                                     {JuicerParams::kCameraDiffusionBloomIntensity, true},
                                                                     {JuicerParams::kCameraDiffusionBloomSize, true},
                                                                     {JuicerParams::kPrintDiffusionEnabled, true},
                                                                     {JuicerParams::kPrintDiffusionFamily, true},
                                                                     {JuicerParams::kPrintDiffusionStrength, true},
                                                                     {JuicerParams::kPrintDiffusionSpatialScale, true},
                                                                     {JuicerParams::kPrintDiffusionHaloWarmth, true},
                                                                     {JuicerParams::kPrintDiffusionCoreIntensity, true},
                                                                     {JuicerParams::kPrintDiffusionCoreSize, true},
                                                                     {JuicerParams::kPrintDiffusionHaloIntensity, true},
                                                                     {JuicerParams::kPrintDiffusionHaloSize, true},
                                                                     {JuicerParams::kPrintDiffusionBloomIntensity, true},
                                                                     {JuicerParams::kPrintDiffusionBloomSize, true},
                                                                     {JuicerParams::kPrintGammaFactor, false},
                                                                     {JuicerParams::kDirCouplersLangmuirDonorKRgb, false},
                                                                     {JuicerParams::kDirCouplersLangmuirReceiverKRgb, false},
                                                                     {JuicerParams::kDirCouplersDiffusionTailWeight, false},
                                                                     {JuicerParams::kDirCouplersDiffusionTailUm, false}}};

    struct UiTraceSummary {
        struct RouteGetter {
            std::size_t index;
            bool filmProfile;
        };
        struct ControlWrites {
            bool enabled = false;
            bool secret = false;
        };
        std::vector<RouteGetter> routeGetters;
        std::vector<std::size_t> routeDependentWrites;
        std::array<ControlWrites, kRouteUiControls.size()> controls{};
    };

    UiTraceSummary summarize_ui_trace(
        const std::vector<std::string>& trace,
        std::size_t start) {
        UiTraceSummary summary;
        const std::string filmGetter =
            "GET_CURRENT name=" + std::string(JuicerParams::kFilmProfileKey) + " ";
        const std::string routeGetter =
            "GET_CURRENT name=" + std::string(JuicerParams::kParamScanRoute) + " ";
        for (std::size_t index = start; index < trace.size(); ++index) {
            const std::string& line = trace[index];
            if (line.starts_with(filmGetter) || line.starts_with(routeGetter)) {
                summary.routeGetters.push_back({index, line.starts_with(filmGetter)});
            }
            for (std::size_t control = 0; control < kRouteUiControls.size(); ++control) {
                const std::string prefix =
                    "PROPERTY_SET_INT param=" +
                    std::string(kRouteUiControls[control].param) + " property=";
                const bool enabled = line.starts_with(prefix + kOfxParamPropEnabled + " ");
                const bool secret = line.starts_with(prefix + kOfxParamPropSecret + " ");
                if (enabled || secret) {
                    summary.routeDependentWrites.push_back(index);
                    summary.controls[control].enabled |= enabled;
                    summary.controls[control].secret |= secret;
                    break;
                }
            }
        }
        return summary;
    }

    bool has_route_getter_pairs(const UiTraceSummary& summary, std::size_t count) {
        if (summary.routeGetters.size() != count * 2u) {
            return false;
        }
        for (std::size_t pair = 0; pair < count; ++pair) {
            if (!summary.routeGetters[pair * 2u].filmProfile ||
                summary.routeGetters[pair * 2u + 1u].filmProfile) {
                return false;
            }
        }
        return true;
    }

    bool ui_writes_precede_snapshot(const UiTraceSummary& summary) {
        if (!has_route_getter_pairs(summary, 3u) ||
            summary.routeDependentWrites.empty() ||
            summary.routeGetters[3].index >= summary.routeDependentWrites.front() ||
            summary.routeDependentWrites.back() >= summary.routeGetters[4].index) {
            return false;
        }
        for (std::size_t control = 0; control < kRouteUiControls.size(); ++control) {
            if (!summary.controls[control].enabled ||
                (kRouteUiControls[control].expectsSecret &&
                 !summary.controls[control].secret)) {
                return false;
            }
        }
        return true;
    }

    bool failure_ui_trace_matches(const UiTraceSummary& summary, int failedCall) {
        return failedCall < 3
                   ? has_route_getter_pairs(summary, static_cast<std::size_t>(failedCall)) &&
                         summary.routeDependentWrites.empty()
                   : ui_writes_precede_snapshot(summary);
    }

    void check_ui_trace_rejections(const std::vector<std::string>& eventTrace) {
        const UiTraceSummary baseline = summarize_ui_trace(eventTrace, 0);
        require(ui_writes_precede_snapshot(baseline),
                "UI trace negative cases need a complete valid event");
        const std::string gammaPrefix =
            "PROPERTY_SET_INT param=" + std::string(JuicerParams::kPrintGammaFactor) +
            " property=" + kOfxParamPropEnabled + " ";
        const auto gamma = std::find_if(eventTrace.begin(), eventTrace.end(), [&](const std::string& line) {
            return line.starts_with(gammaPrefix);
        });
        require(gamma != eventTrace.end(), "UI trace is missing the print gamma write");
        const std::string& gammaWrite = *gamma;
        auto earlyGamma = eventTrace;
        earlyGamma.insert(earlyGamma.begin(), gammaWrite);
        require(!ui_writes_precede_snapshot(summarize_ui_trace(earlyGamma, 0)),
                "UI trace accepted a gamma write before route preflight");

        for (int failedCall : {1, 2}) {
            std::vector<std::string> failedTrace(
                eventTrace.begin(),
                eventTrace.begin() + static_cast<std::ptrdiff_t>(
                                         baseline.routeGetters[static_cast<std::size_t>(failedCall) * 2u - 1u].index + 1u));
            require(failure_ui_trace_matches(summarize_ui_trace(failedTrace, 0), failedCall),
                    "UI trace rejected a write-free failed preflight");
            failedTrace.push_back(gammaWrite);
            require(!failure_ui_trace_matches(summarize_ui_trace(failedTrace, 0), failedCall),
                    "UI trace accepted a gamma write during a failed preflight");
        }

        auto lateGamma = eventTrace;
        lateGamma.erase(lateGamma.begin() + std::distance(eventTrace.begin(), gamma));
        lateGamma.push_back(gammaWrite);
        require(!failure_ui_trace_matches(summarize_ui_trace(lateGamma, 0), 3),
                "UI trace accepted a gamma write after snapshot acquisition");

        for (const std::size_t write : baseline.routeDependentWrites) {
            auto missingWrite = eventTrace;
            missingWrite.erase(missingWrite.begin() + static_cast<std::ptrdiff_t>(write));
            require(!ui_writes_precede_snapshot(summarize_ui_trace(missingWrite, 0)),
                    "UI trace accepted a missing expected property write: " + eventTrace[write]);
        }
        for (std::size_t pair = 0; pair < 3u; ++pair) {
            auto wrongGetter = eventTrace;
            wrongGetter[baseline.routeGetters[pair * 2u].index] =
                eventTrace[baseline.routeGetters[pair * 2u + 1u].index];
            require(!ui_writes_precede_snapshot(summarize_ui_trace(wrongGetter, 0)),
                    "UI trace accepted route getters without the matching film getter");
        }
        std::cout << "UI trace negative cases passed: early gamma, failures 1/2, late gamma, "
                  << baseline.routeDependentWrites.size()
                  << " missing writes, and three malformed getter pairs\n";
    }

    class LoadedOfxModule final {
    public:
        explicit LoadedOfxModule(const std::filesystem::path& path) {
#if defined(_WIN32)
            _handle = LoadLibraryW(path.c_str());
            if (!_handle) {
                throw std::runtime_error(
                    "could not load OFX module: Windows error " +
                    std::to_string(GetLastError()));
            }
#else
            _handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!_handle) {
                const char* const error = dlerror();
                throw std::runtime_error(
                    "could not load OFX module: " +
                    std::string(error ? error : "unknown error"));
            }
#endif
        }

        ~LoadedOfxModule() {
            if (_handle) {
#if defined(_WIN32)
                (void)FreeLibrary(_handle);
#else
                (void)dlclose(_handle);
#endif
            }
        }

        LoadedOfxModule(const LoadedOfxModule&) = delete;
        LoadedOfxModule& operator=(const LoadedOfxModule&) = delete;

        template <typename Function>
        Function symbol(const char* name) const {
#if defined(_WIN32)
            const FARPROC address = GetProcAddress(_handle, name);
            if (!address) {
                throw std::runtime_error(
                    "missing OFX module symbol " + std::string(name) +
                    ": Windows error " + std::to_string(GetLastError()));
            }
#else
            dlerror();
            void* const address = dlsym(_handle, name);
            const char* const error = dlerror();
            if (error || !address) {
                throw std::runtime_error(
                    "missing OFX module symbol " + std::string(name) +
                    ": " + (error ? error : "null address"));
            }
#endif
            return reinterpret_cast<Function>(address);
        }

    private:
#if defined(_WIN32)
        HMODULE _handle = nullptr;
#else
        void* _handle = nullptr;
#endif
    };

    void run_loaded_module_callback(
        NativeHost& host,
        const std::filesystem::path& modulePath,
        const std::filesystem::path& tracePath,
        bool streamContract = false) {
        using GetPluginCount = int (*)();
        using GetPlugin = OfxPlugin* (*)(int);

        LoadedOfxModule module(modulePath);
        const GetPluginCount getPluginCount =
            module.symbol<GetPluginCount>("OfxGetNumberOfPlugins");
        const GetPlugin getPlugin = module.symbol<GetPlugin>("OfxGetPlugin");
        require(getPluginCount() == 1, "loaded OFX module did not report one plugin");
        OfxPlugin* const plugin = getPlugin(0);
        require(plugin && plugin->setHost && plugin->mainEntry,
                "loaded OFX plugin descriptor is incomplete");
        host.set_descriptor_module_path(modulePath);
        plugin->setHost(host.ofx_host());

        bool loaded = false;
        bool instanceCreated = false;
        try {
            require(
                plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr) ==
                    kOfxStatOK,
                "loaded OFX module rejected the load action");
            loaded = true;
            require(
                plugin->mainEntry(
                    kOfxActionDescribe,
                    host.descriptor_handle(),
                    nullptr,
                    nullptr) == kOfxStatOK,
                "loaded OFX module rejected the describe action");
            require(
                plugin->mainEntry(
                    kOfxImageEffectActionDescribeInContext,
                    host.descriptor_handle(),
                    host.describe_context_handle(),
                    nullptr) == kOfxStatOK,
                "loaded OFX module rejected the describe-in-context action");
            require(
                plugin->mainEntry(
                    kOfxActionCreateInstance,
                    host.effect_handle(),
                    nullptr,
                    nullptr) == kOfxStatOK,
                "loaded OFX module rejected the create-instance action");
            instanceCreated = true;
            if (streamContract) {
                host.render_stream_cases(*plugin);
            }
            const OfxStatus destroyStatus = plugin->mainEntry(
                kOfxActionDestroyInstance,
                host.effect_handle(),
                nullptr,
                nullptr);
            instanceCreated = false;
            require(
                destroyStatus == kOfxStatOK,
                "loaded OFX module rejected the destroy-instance action");
            const OfxStatus unloadStatus =
                plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr);
            loaded = false;
            require(
                unloadStatus == kOfxStatOK,
                "loaded OFX module rejected the unload action");
        } catch (...) {
            if (instanceCreated) {
                (void)plugin->mainEntry(
                    kOfxActionDestroyInstance,
                    host.effect_handle(),
                    nullptr,
                    nullptr);
            }
            if (loaded) {
                (void)plugin->mainEntry(
                    kOfxActionUnload,
                    nullptr,
                    nullptr,
                    nullptr);
            }
            host.write_trace(tracePath);
            throw;
        }

        host.write_trace(tracePath);
        std::cout << "PASS adapter/loaded-module-create-callback\n"
                  << "MODULE " << modulePath.string() << '\n'
                  << "TRACE " << tracePath.string() << " entries="
                  << host.trace_size() << '\n';
    }
} // namespace

namespace {
    nlohmann::json physical_words(const DefectDustRecipe& value) {
        return nlohmann::json::array({std::bit_cast<std::uint32_t>(value.cellWidthMm),
                                      std::bit_cast<std::uint32_t>(value.cellHeightMm),
                                      std::bit_cast<std::uint32_t>(value.slotProbability),
                                      std::bit_cast<std::uint32_t>(value.softnessMinMm),
                                      std::bit_cast<std::uint32_t>(value.softnessMaxMm),
                                      std::bit_cast<std::uint32_t>(value.softnessSizeCapFraction),
                                      std::bit_cast<std::uint32_t>(value.supportXMm),
                                      std::bit_cast<std::uint32_t>(value.supportYMm),
                                      std::bit_cast<std::uint32_t>(value.fiberFraction),
                                      std::bit_cast<std::uint32_t>(value.fiberDriftFraction),
                                      std::bit_cast<std::uint32_t>(value.fiberFirstKnotMin),
                                      std::bit_cast<std::uint32_t>(value.fiberFirstKnotMax),
                                      std::bit_cast<std::uint32_t>(value.fiberSecondKnotMin),
                                      std::bit_cast<std::uint32_t>(value.fiberSecondKnotMax),
                                      std::bit_cast<std::uint32_t>(value.fiberInteriorWidthMinFraction),
                                      std::bit_cast<std::uint32_t>(value.fiberInteriorWidthMaxFraction),
                                      std::bit_cast<std::uint32_t>(value.diameterMinMm),
                                      std::bit_cast<std::uint32_t>(value.diameterBulkMaxMm),
                                      std::bit_cast<std::uint32_t>(value.diameterMaxMm),
                                      std::bit_cast<std::uint32_t>(value.diameterTailFraction),
                                      std::bit_cast<std::uint32_t>(value.fiberLengthMinMm),
                                      std::bit_cast<std::uint32_t>(value.fiberLengthMaxMm),
                                      std::bit_cast<std::uint32_t>(value.fiberWidthMinMm),
                                      std::bit_cast<std::uint32_t>(value.fiberWidthMaxMm),
                                      std::bit_cast<std::uint32_t>(value.opacityFaintCumulative),
                                      std::bit_cast<std::uint32_t>(value.opacityIntermediateCumulative),
                                      std::bit_cast<std::uint32_t>(value.compactOpacityMin),
                                      std::bit_cast<std::uint32_t>(value.compactOpacityFaintEnd),
                                      std::bit_cast<std::uint32_t>(value.compactOpacityIntermediateEnd),
                                      std::bit_cast<std::uint32_t>(value.compactOpacityMax),
                                      std::bit_cast<std::uint32_t>(value.fiberOpacityMin),
                                      std::bit_cast<std::uint32_t>(value.fiberOpacityFaintEnd),
                                      std::bit_cast<std::uint32_t>(value.fiberOpacityIntermediateEnd),
                                      std::bit_cast<std::uint32_t>(value.fiberOpacityMax),
                                      std::bit_cast<std::uint32_t>(value.compactDominantAspectMin),
                                      std::bit_cast<std::uint32_t>(value.compactDominantAspectMax),
                                      std::bit_cast<std::uint32_t>(value.compactSubsidiaryScaleMin),
                                      std::bit_cast<std::uint32_t>(value.compactSubsidiaryScaleMax),
                                      std::bit_cast<std::uint32_t>(value.compactSubsidiaryAspectMin),
                                      std::bit_cast<std::uint32_t>(value.compactSubsidiaryAspectMax),
                                      std::bit_cast<std::uint32_t>(value.compactSubsidiaryOffsetMax),
                                      std::bit_cast<std::uint32_t>(value.compactSubsidiaryAngleMaxRadians)});
    }
    nlohmann::json physical_words(const DefectScratchRecipe& value) {
        return nlohmann::json::array({std::bit_cast<std::uint32_t>(value.cellWidthMm),
                                      std::bit_cast<std::uint32_t>(value.cellHeightMm),
                                      std::bit_cast<std::uint32_t>(value.slotProbability),
                                      std::bit_cast<std::uint32_t>(value.softnessMinMm),
                                      std::bit_cast<std::uint32_t>(value.softnessMaxMm),
                                      std::bit_cast<std::uint32_t>(value.softnessSizeCapFraction),
                                      std::bit_cast<std::uint32_t>(value.supportXMm),
                                      std::bit_cast<std::uint32_t>(value.supportYMm),
                                      std::bit_cast<std::uint32_t>(value.lengthMinMm),
                                      std::bit_cast<std::uint32_t>(value.lengthBulkMaxMm),
                                      std::bit_cast<std::uint32_t>(value.lengthMaxMm),
                                      std::bit_cast<std::uint32_t>(value.lengthTailFraction),
                                      std::bit_cast<std::uint32_t>(value.widthMinMm),
                                      std::bit_cast<std::uint32_t>(value.widthBulkMaxMm),
                                      std::bit_cast<std::uint32_t>(value.widthMaxMm),
                                      std::bit_cast<std::uint32_t>(value.widthTailFraction),
                                      std::bit_cast<std::uint32_t>(value.driftFraction),
                                      std::bit_cast<std::uint32_t>(value.firstKnotMin),
                                      std::bit_cast<std::uint32_t>(value.firstKnotMax),
                                      std::bit_cast<std::uint32_t>(value.secondKnotMin),
                                      std::bit_cast<std::uint32_t>(value.secondKnotMax),
                                      std::bit_cast<std::uint32_t>(value.interiorWidthMinFraction),
                                      std::bit_cast<std::uint32_t>(value.interiorWidthMaxFraction),
                                      std::bit_cast<std::uint32_t>(value.interiorDepthMinFraction),
                                      std::bit_cast<std::uint32_t>(value.interiorDepthMaxFraction),
                                      std::bit_cast<std::uint32_t>(value.endpointAbruptProbability),
                                      std::bit_cast<std::uint32_t>(value.interruptionProbability),
                                      std::bit_cast<std::uint32_t>(value.gapCenterMin),
                                      std::bit_cast<std::uint32_t>(value.gapCenterMax),
                                      std::bit_cast<std::uint32_t>(value.gapSpanMin),
                                      std::bit_cast<std::uint32_t>(value.gapSpanMax),
                                      std::bit_cast<std::uint32_t>(value.scuffProbability),
                                      std::bit_cast<std::uint32_t>(value.scuffLengthMaxMm),
                                      std::bit_cast<std::uint32_t>(value.scuffAngleMaxRadians),
                                      std::bit_cast<std::uint32_t>(value.strengthMin),
                                      std::bit_cast<std::uint32_t>(value.strengthMax)});
    }
    using Json = nlohmann::json;
    template <class T>
    Json optics_words(const T& a) {
        Json out = Json::array();
        for (auto v : a) {
            if constexpr (sizeof(v) == 4)
                out.push_back(std::bit_cast<std::uint32_t>(v));
            else
                out.push_back(std::bit_cast<std::uint64_t>(v));
        }
        return out;
    }
    using OpticsAuthored = Spektrafilm::DiffusionFilterAuthoredControls;
    std::array<double OpticsAuthored::*, 9> optics_lanes() {
        return {&OpticsAuthored::strength, &OpticsAuthored::spatialScale, &OpticsAuthored::haloWarmth, &OpticsAuthored::coreIntensity, &OpticsAuthored::coreSize, &OpticsAuthored::haloIntensity, &OpticsAuthored::haloSize, &OpticsAuthored::bloomIntensity, &OpticsAuthored::bloomSize};
    }
    Json optics_project(const OpticsAuthored& a) {
        std::array<double, 9> v{};
        auto fields = optics_lanes();
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = a.*fields[i];
        return {{"active", a.active}, {"family", static_cast<unsigned>(a.family)}, {"f64", optics_words(v)}};
    }
    Json optics_project(const DiffusionFilterOpticsRecipe& a) {
        const auto& r = a.resolved;
        return {{"domain", static_cast<unsigned>(a.domain)}, {"active", r.active}, {"family", static_cast<unsigned>(r.family)}, {"f64", optics_words(std::array<double, 9>{r.scatterFraction, r.groupWeightsCoreHaloBloom[0], r.groupWeightsCoreHaloBloom[1], r.groupWeightsCoreHaloBloom[2], r.groupCenterLambdaUm[0], r.groupCenterLambdaUm[1], r.groupCenterLambdaUm[2], r.effectiveWarmth, r.spatialScale})}, {"resolved_hash", r.hash}, {"hash", a.hash}};
    }
    Json optics_project(const SpatialOptics& a) {
        const auto& h = a.scatterHalation;
        return {{"camera", optics_project(a.cameraDiffusion)}, {"lens_f32", std::bit_cast<std::uint32_t>(a.cameraLensBlur.sigmaUm)}, {"lens_hash", a.cameraLensBlur.hash}, {"halation", {{"scatter_active", h.scatterActive}, {"back_reflection_active", h.backReflectionActive}, {"f32", optics_words(std::array<float, 9>{h.scatterAmount, h.scatterSpatialScale, h.halationSpatialScale, h.halationFirstSigmaUm[0], h.halationFirstSigmaUm[1], h.halationFirstSigmaUm[2], h.totalStrength[0], h.totalStrength[1], h.totalStrength[2]})}, {"hash", h.hash}}}, {"enlarger", optics_project(a.enlargerDiffusion)}, {"hash", a.hash}};
    }
    struct OpticsCase {
        Spektrafilm::SpatialOpticsControls controls;
        Profiles::ProfileDigest digest;
        bool print = false;
    };
    Json optics_inputs(const OpticsCase& c) {
        const auto& h = c.controls.scatterHalation;
        return {{"camera", optics_project(c.controls.cameraDiffusion)}, {"lens_f32", std::bit_cast<std::uint32_t>(c.controls.cameraLensBlurUm)}, {"halation_active", h.active}, {"halation_f32", optics_words(std::array<float, 4>{h.scatterAmount, h.scatterSpatialScale, h.halationAmount, h.halationSpatialScale})}, {"first_sigma_f32", optics_words(c.digest.halationFirstSigmaUm)}, {"primary_f32", optics_words(c.digest.halationPrimaryAmount)}, {"enlarger", optics_project(c.controls.enlargerDiffusion)}, {"print", c.print}};
    }

    void check_optics_capture(const std::filesystem::path& grainManifest, const char* preset) {
        const auto path = grainManifest.parent_path().parent_path() / "optics-fixtures/manifest.json";
        std::ifstream manifestStream(path);
        require(bool(manifestStream), "optics manifest open");
        const auto manifest = Json::parse(manifestStream);
        std::ifstream stream(path.parent_path() / manifest.at("presets").at(preset).at("ofx").get<std::string>());
        require(bool(stream), "optics capture payload open");
        const auto frozen = Json::parse(stream).at("ofx");
        require(frozen.size() == 96, "complete frozen optics capture membership");
        for (unsigned route = 0; route < 4; ++route) {
            NativeHost host;
            JuicerEffect effect(host.effect_handle());
            host.bind_effect(effect);
            host.set_time(0);
            host.set_str_choice_value(JuicerParams::kFilmProfileKey, route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f");
            host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(route % 2 ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan));
            const std::array<const char*, 9> camera{JuicerParams::kCameraDiffusionStrength, JuicerParams::kCameraDiffusionSpatialScale, JuicerParams::kCameraDiffusionHaloWarmth, JuicerParams::kCameraDiffusionCoreIntensity, JuicerParams::kCameraDiffusionCoreSize, JuicerParams::kCameraDiffusionHaloIntensity, JuicerParams::kCameraDiffusionHaloSize, JuicerParams::kCameraDiffusionBloomIntensity, JuicerParams::kCameraDiffusionBloomSize};
            const std::array<const char*, 9> enlarger{JuicerParams::kPrintDiffusionStrength, JuicerParams::kPrintDiffusionSpatialScale, JuicerParams::kPrintDiffusionHaloWarmth, JuicerParams::kPrintDiffusionCoreIntensity, JuicerParams::kPrintDiffusionCoreSize, JuicerParams::kPrintDiffusionHaloIntensity, JuicerParams::kPrintDiffusionHaloSize, JuicerParams::kPrintDiffusionBloomIntensity, JuicerParams::kPrintDiffusionBloomSize};
            const std::array<const char*, 4> halation{JuicerParams::kHalationScatterAmount, JuicerParams::kHalationScatterSpatialScale, JuicerParams::kHalationAmount, JuicerParams::kHalationSpatialScale};
            for (int n = 0; n < 24; ++n) {
                bool enabled = n % 3 != 0;
                host.set_int_value(JuicerParams::kHalationActive, enabled);
                host.set_int_value(JuicerParams::kCameraDiffusionEnabled, n % 2);
                host.set_int_value(JuicerParams::kPrintDiffusionEnabled, n % 2);
                host.set_int_value(JuicerParams::kCameraDiffusionFamily, n == 20 ? -4 : n == 21 ? 9
                                                                                                : n % 4);
                host.set_int_value(JuicerParams::kPrintDiffusionFamily, (n + 1) % 4);
                for (std::size_t i = 0; i < 9; ++i) {
                    host.set_double_value(camera[i], i == 2 ? 0.1234567890123 : i == 0 ? 0.731234567890123
                                                                                       : 1.0);
                    host.set_double_value(enlarger[i], i == 2 ? 0.0 : i == 0 ? 0.81234567890123
                                                                             : 1.0);
                }
                if (n == 22)
                    host.set_double_value(JuicerParams::kCameraDiffusionStrength, std::numeric_limits<double>::quiet_NaN());
                for (const auto* field : halation)
                    host.set_double_value(field, enabled ? 1.234567890123 : std::numeric_limits<double>::quiet_NaN());
                if (n == 19)
                    host.set_double_value(JuicerParams::kHalationAmount, 2.0000000000000004);
                auto timed = host.at_time_get_count();
                auto start = host.trace_size();
                host.dispatch_event(effect, JuicerParams::kGrainActive, OFX::eChangeTime);
                auto observed = JuicerTestSupport::EffectTraceObserver::snapshot(effect);
                require(host.at_time_get_count() == timed, "current getters");
                Json getters = Json::array();
                for (std::size_t i = start; i < host.trace_size(); ++i)
                    if (host.trace()[i].starts_with("GET_CURRENT"))
                        getters.push_back(host.trace()[i]);
                auto admission = JuicerTestSupport::EffectTraceObserver::admit(effect);
                const RenderRecipe* recipe = admission.printState ? &admission.printState->recipe : admission.directState ? &admission.directState->recipe
                                                                                                                          : nullptr;
                OpticsCase captured;
                captured.controls.cameraDiffusion = observed.pendingOptics.cameraDiffusion;
                captured.controls.enlargerDiffusion = observed.pendingOptics.enlargerDiffusion;
                captured.controls.scatterHalation = observed.pendingOptics.scatterHalation;
                captured.print = route % 2;
                const auto& expected = frozen.at(route * 24 + n);
                const Json actual{{"route", route}, {"index", n}, {"pending_valid", observed.pendingValid}, {"pending_invalid", observed.pendingInvalid}, {"pending_diagnostic", observed.pendingDiagnostic}, {"snapshot_hash", observed.pendingFullHash}, {"admitted", recipe != nullptr}, {"input", optics_inputs(captured)}, {"recipe", recipe ? optics_project(recipe->spatialOptics) : Json()}, {"final_hash", recipe ? recipe->hash : 0}, {"getters", getters}};
                require(actual == expected, "all frozen OFX optics capture/getter/publication fields route=" + std::to_string(route) + " index=" + std::to_string(n));
            }
            host.unbind_effect();
        }
        std::cout << "PASS adapter/96-frozen-optics-capture-publications-four-routes\n";
    }
    void check_physical_capture(NativeHost& host, const std::filesystem::path& grainManifest, const char* preset) {
        const auto manifestPath = grainManifest.parent_path().parent_path() / "effects-fixtures/manifest.json";
        std::ifstream manifestStream(manifestPath);
        require(bool(manifestStream), "effects capture manifest open");
        const auto manifest = nlohmann::json::parse(manifestStream);
        std::ifstream stream(manifestPath.parent_path() / manifest.at("presets").at(preset).at("ofx").get<std::string>());
        require(bool(stream), "effects capture fixture open");
        const auto captured = nlohmann::json::parse(stream);
        require(captured.at("ofx").size() == 608, "complete frozen effects capture membership");
        JuicerEffect effect(host.effect_handle());
        host.bind_effect(effect);
        host.set_time(0);
        const std::array<const char*, 5> getters{JuicerParams::kFilmDustAmount, JuicerParams::kGateDustAmount, JuicerParams::kFilmScratchAmount, JuicerParams::kGateScratchAmount, JuicerParams::kGateWeaveAmount};
        unsigned count = 0;
        for (const auto& row : captured.at("ofx")) {
            const unsigned policy = row.at("policy");
            const unsigned index = row.at("index");
            if (!((index == 5 && (policy == 0 || policy == 1 || policy == 4 || policy == 7)) || (index == 15 && policy == 0)))
                continue;
            const unsigned route = row.at("route");
            const bool print = route % 2;
            host.set_str_choice_value(JuicerParams::kFilmProfileKey, route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f");
            host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan));
            host.set_int_value(JuicerParams::kGrainActive, policy != 0);
            host.set_int_value(JuicerParams::kGrainDebugView, policy > 1 ? static_cast<int>(policy - 1) : 0);
            for (std::size_t i = 0; i < getters.size(); ++i)
                host.set_double_value(getters[i], std::bit_cast<double>(row.at("authored_getter_order_f64")[i].get<std::uint64_t>()));
            const auto start = host.trace_size();
            const auto timed = host.at_time_get_count();
            host.dispatch_event(effect, JuicerParams::kGrainActive, OFX::eChangeTime);
            const auto observed = JuicerTestSupport::EffectTraceObserver::snapshot(effect);
            require(observed.pendingValid && host.at_time_get_count() == timed, "physical capture uses current getters even during disabled/debug grain");
            bool ordered = false;
            for (std::size_t i = start; i < host.trace_size(); ++i) {
                std::size_t j = 0;
                while (j < getters.size() && i + j < host.trace_size() && host.trace()[i + j].starts_with("GET_CURRENT name=" + std::string(getters[j]) + " "))
                    ++j;
                ordered = ordered || j == getters.size();
            }
            require(ordered, "physical getter order film dust/gate dust/film scratch/gate scratch/weave");
            const auto& controls = observed.pendingEffects;
            const auto words = nlohmann::json::array({std::bit_cast<std::uint32_t>(controls.filmDustAmount), std::bit_cast<std::uint32_t>(controls.filmScratchAmount), std::bit_cast<std::uint32_t>(controls.gateDustAmount), std::bit_cast<std::uint32_t>(controls.gateScratchAmount)});
            require(words == row.at("controls").at("amounts") && std::bit_cast<std::uint64_t>(controls.gateWeaveAmount) == row.at("controls").at("weave") && observed.pendingFullHash == row.at("snapshot_hash"), "frozen native captured f32/f64 controls and snapshot identity");
            const auto admitted = JuicerTestSupport::EffectTraceObserver::admit(effect);
            require(print ? bool(admitted.printState) : bool(admitted.directState), "physical capture completes actual route publication");
            const auto& recipe = print ? admitted.printState->recipe : admitted.directState->recipe;
            const auto& effects = recipe.filmJuicerEffects;
            const nlohmann::json projection{{"film_dust", physical_words(effects.filmDust)}, {"film_scratch", physical_words(effects.filmScratch)}, {"gate_dust", physical_words(effects.gateDust)}, {"gate_scratch", physical_words(effects.gateScratch)}, {"weave", std::bit_cast<std::uint64_t>(effects.gateWeaveAmount)}, {"active", effects.active}, {"hash", effects.hash}};
            require(projection == row.at("recipe") && recipe.visualGrain.hash == row.at("grain_hash") && recipe.hash == row.at("final_hash"), "actual OFX snapshot to completed grain to exact Rust effects and final publication");
            ++count;
        }
        require(count == 20, "bounded physical OFX witness membership");
        for (const auto* getter : getters)
            host.set_double_value(getter, 0);
        host.set_str_choice_value(JuicerParams::kFilmProfileKey, Spektrafilm::kDefaultFilmProfileKey);
        host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(Spektrafilm::kDefaultScanRoute));
        host.set_int_value(JuicerParams::kGrainActive, 0);
        host.set_int_value(JuicerParams::kGrainDebugView, 0);
        host.unbind_effect();
        std::cout << "PASS adapter/20-frozen-physical-capture-publications-four-routes\n";
    }
} // namespace

int main(int argc, char** argv) { // NOLINT(bugprone-exception-escape): top-level catches map all fixture failures.
    JuicerCuda::Owner cudaOwner;
    try {
        if (argc == 5 && (std::string_view(argv[1]) == "--loaded-module" || std::string_view(argv[1]) == "--stream-contract") &&
            std::string_view(argv[3]) == "--trace-output") {
            NativeHost host;
            run_loaded_module_callback(host, argv[2], argv[4], std::string_view(argv[1]) == "--stream-contract");
            return 0;
        }
        if ((argc != 3 && !(argc == 6 && std::string_view(argv[3]) == "--grain-fixtures")) || std::string_view(argv[1]) != "--trace-output") {
            throw std::runtime_error(
                "usage: JuicerAdapterTraceProbe --trace-output PATH or "
                "--loaded-module MODULE --trace-output PATH");
        }
        const std::filesystem::path tracePath = argv[2];
        cudaOwner.create(JuicerProcess::data_directory());
        if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
            throw std::runtime_error("CUDA initialization failed");
        }
        JuicerProcess::root().ensure_bootstrap();
        if (argc == 6)
            check_optics_capture(argv[4], argv[5]);
        NativeHost host;
        if (argc == 6)
            check_physical_capture(host, argv[4], argv[5]);
        ParamSnapshot routeSnapshot;
        routeSnapshot.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        FocusedRenderStateBuildProduct routeProduct;
        std::string routeDiagnostic;
        require(
            build_direct_render_state_product(routeSnapshot, routeProduct, routeDiagnostic),
            "matching negative profile route did not build: " + routeDiagnostic);
        routeSnapshot.scanRoute = Spektrafilm::ScanRoute::PositiveDirectScan;
        routeDiagnostic.clear();
        require(
            !build_direct_render_state_product(routeSnapshot, routeProduct, routeDiagnostic) &&
                routeDiagnostic.find("route mismatch") != std::string::npos,
            "mismatched negative profile route was not rejected");
        routeSnapshot.filmProfileKey = "fujifilm_provia_100f";
        routeDiagnostic.clear();
        require(
            build_direct_render_state_product(routeSnapshot, routeProduct, routeDiagnostic),
            "matching positive profile route did not build: " + routeDiagnostic);
        routeSnapshot.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        routeDiagnostic.clear();
        require(
            !build_direct_render_state_product(routeSnapshot, routeProduct, routeDiagnostic) &&
                routeDiagnostic.find("route mismatch") != std::string::npos,
            "mismatched positive profile route was not rejected");
        RouteFaultTest::fail_on_call(1);
        routeSnapshot.scanRoute = Spektrafilm::ScanRoute::PositiveDirectScan;
        routeDiagnostic.clear();
        require(build_direct_render_state_product(routeSnapshot, routeProduct, routeDiagnostic),
                "completed film family unexpectedly used the scalar route bridge");
        Spektrafilm::ScanRoute unusedRoute{};
        require(!Spektrafilm::resolve_scan_route(Spektrafilm::ProfilePolarity::Positive,
                                                 routeSnapshot.scanRoute,
                                                 unusedRoute,
                                                 routeDiagnostic) &&
                    routeDiagnostic.find("RouteResolutionFailure") != std::string::npos,
                "remaining scalar route fault was consumed by the migrated builder");
        DeviceFrame frame;
        for (int failedCall : {1, 2}) {
            const std::size_t traceStart = host.trace_size();
            RouteFaultTest::fail_on_call(failedCall);
            bool creationFailed = false;
            try {
                JuicerEffect failedEffect(host.effect_handle());
            } catch (const OFX::Exception::Suite&) {
                creationFailed = true;
            }
            require(creationFailed, "route bridge failure did not fail instance creation");
            const auto& trace = host.trace();
            require(
                std::any_of(
                    trace.begin() + static_cast<std::ptrdiff_t>(traceStart),
                    trace.end(),
                    [](const std::string& line) {
                        return line.find("id=FilmJuicerRouteResolution") != std::string::npos &&
                               line.find("RouteResolutionFailure") != std::string::npos;
                    }),
                "constructor route bridge failure omitted its host diagnostic");
        }
        {
            JuicerEffect effect(host.effect_handle());
            host.bind_effect(effect);

            host.set_time(0.0);
            RouteFaultTest::fail_on_call(1);
            const auto initialRejected =
                host.render(effect, frame, "initial-route-failure", true);
            require(initialRejected.rejected && initialRejected.state.pendingInvalid &&
                        initialRejected.state.pendingDiagnostic.find("RouteResolutionFailure") !=
                            std::string::npos,
                    "initial bridge failure did not block admission");
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
            const auto first = host.render(effect, frame, "first-render");
            require(first.succeeded && exposure_matches(first, 0.0) &&
                        has_valid_latch(first),
                    "first render did not acquire the current zero-EV value");

            constexpr std::array<double, 3> recoveryExposureEv{{0.125,
                                                                -0.25,
                                                                0.375}};
            for (std::size_t failureIndex = 0;
                 failureIndex < recoveryExposureEv.size();
                 ++failureIndex) {
                const int failedCall = static_cast<int>(failureIndex) + 1;
                const PendingRenderAdmissionResult retained =
                    JuicerTestSupport::EffectTraceObserver::admit(effect);
                const bool retainedPrint =
                    retained.status == PendingRenderAdmissionStatus::AdmittedPrint;
                require(
                    (retainedPrint && retained.printState &&
                     !retained.directState) ||
                        (retained.status ==
                             PendingRenderAdmissionStatus::AdmittedDirect &&
                         retained.directState && !retained.printState),
                    "could not retain an admitted route state before failure");
                const void* const retainedStateIdentity = retainedPrint
                                                              ? static_cast<const void*>(
                                                                    retained.printState.get())
                                                              : static_cast<const void*>(
                                                                    retained.directState.get());
                const std::uint64_t retainedBuildCounter =
                    retainedPrint ? retained.printState->buildCounter
                                  : retained.directState->buildCounter;
                const std::uint64_t retainedRecipeHash =
                    retainedPrint ? retained.printState->recipe.hash
                                  : retained.directState->recipe.hash;
                const Spektrafilm::ScanRoute retainedRoute =
                    retained.snapshot.scanRoute;
                const double retainedExposureEv =
                    retained.snapshot.cameraExposureCompensationEv;

                const auto retained_state_unchanged = [&]() {
                    if (retained.snapshot.scanRoute != retainedRoute ||
                        !double_matches(retained.snapshot.cameraExposureCompensationEv,
                                        retainedExposureEv)) {
                        return false;
                    }
                    return retainedPrint
                               ? retained.status == PendingRenderAdmissionStatus::AdmittedPrint &&
                                     !retained.directState && retained.printState &&
                                     retained.printState.get() ==
                                         retainedStateIdentity &&
                                     retained.printState->buildCounter ==
                                         retainedBuildCounter &&
                                     retained.printState->recipe.hash ==
                                         retainedRecipeHash
                               : retained.status == PendingRenderAdmissionStatus::AdmittedDirect &&
                                     !retained.printState && retained.directState &&
                                     retained.directState.get() ==
                                         retainedStateIdentity &&
                                     retained.directState->buildCounter ==
                                         retainedBuildCounter &&
                                     retained.directState->recipe.hash ==
                                         retainedRecipeHash;
                };

                host.set_double_value(
                    kParamExposure,
                    recoveryExposureEv[failureIndex]);
                const std::size_t traceStart = host.trace_size();
                RouteFaultTest::fail_on_call(failedCall);
                host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
                const auto invalid =
                    JuicerTestSupport::EffectTraceObserver::snapshot(effect);
                const PendingRenderAdmissionResult blocked =
                    JuicerTestSupport::EffectTraceObserver::admit(effect);
                const auto rejected = host.render(
                    effect, frame, "route-bridge-failure", true);
                const UiTraceSummary failureTrace =
                    summarize_ui_trace(host.trace(), traceStart);
                require(invalid.pendingInvalid &&
                            invalid.pendingDiagnostic.find("RouteResolutionFailure") !=
                                std::string::npos &&
                            blocked.status ==
                                PendingRenderAdmissionStatus::InvalidSnapshotControls &&
                            !blocked.directState && !blocked.printState &&
                            rejected.rejected && rejected.state.pendingInvalid &&
                            failure_ui_trace_matches(failureTrace, failedCall) &&
                            retained_state_unchanged(),
                        "route bridge failure did not block new admission while retaining the admitted state");
                host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
                const PendingRenderAdmissionResult recoveredAdmission =
                    JuicerTestSupport::EffectTraceObserver::admit(effect);
                const auto recoveredRoute =
                    host.render(effect, frame, "route-bridge-recovery");
                require(
                    retainedPrint
                        ? recoveredAdmission.status ==
                                  PendingRenderAdmissionStatus::AdmittedPrint &&
                              recoveredAdmission.printState &&
                              !recoveredAdmission.directState
                        : recoveredAdmission.status ==
                                  PendingRenderAdmissionStatus::AdmittedDirect &&
                              recoveredAdmission.directState &&
                              !recoveredAdmission.printState,
                    "valid route recovery did not return the expected admission type");
                const void* const recoveredStateIdentity = retainedPrint
                                                               ? static_cast<const void*>(
                                                                     recoveredAdmission.printState.get())
                                                               : static_cast<const void*>(
                                                                     recoveredAdmission.directState.get());
                const std::uint64_t recoveredBuildCounter = retainedPrint
                                                                ? recoveredAdmission.printState->buildCounter
                                                                : recoveredAdmission.directState->buildCounter;
                const std::uint64_t recoveredRecipeHash = retainedPrint
                                                              ? recoveredAdmission.printState->recipe.hash
                                                              : recoveredAdmission.directState->recipe.hash;
                require(recoveredRoute.succeeded &&
                            !recoveredRoute.state.pendingInvalid &&
                            exposure_matches(
                                recoveredRoute,
                                recoveryExposureEv[failureIndex]) &&
                            recoveredAdmission.snapshot.scanRoute == retainedRoute &&
                            double_matches(
                                recoveredAdmission.snapshot
                                    .cameraExposureCompensationEv,
                                recoveryExposureEv[failureIndex]) &&
                            recoveredStateIdentity != retainedStateIdentity &&
                            recoveredBuildCounter > retainedBuildCounter &&
                            recoveredRecipeHash != retainedRecipeHash &&
                            recoveredRoute.state.recipeHash ==
                                recoveredRecipeHash &&
                            retained_state_unchanged(),
                        "route bridge failure did not replace publication while preserving the retained admission");
            }

            host.set_str_choice_value(
                JuicerParams::kParamScanRoute,
                Spektrafilm::scan_route_key(
                    Spektrafilm::ScanRoute::NegativeDirectScan));
            std::size_t uiTraceStart = host.trace_size();
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
            const UiTraceSummary directUiTrace =
                summarize_ui_trace(host.trace(), uiTraceStart);
            require(
                ui_writes_precede_snapshot(directUiTrace) &&
                    host.param_property_int(
                        JuicerParams::kCameraDiffusionEnabled,
                        kOfxParamPropSecret) == 0 &&
                    host.param_property_int(
                        JuicerParams::kCameraDiffusionEnabled,
                        kOfxParamPropEnabled) == 1 &&
                    host.param_property_int(
                        JuicerParams::kPrintDiffusionEnabled,
                        kOfxParamPropSecret) == 1 &&
                    host.param_property_int(
                        JuicerParams::kPrintDiffusionEnabled,
                        kOfxParamPropEnabled) == 0 &&
                    host.param_property_int(
                        JuicerParams::kPrintGammaFactor,
                        kOfxParamPropEnabled) == 0,
                "direct UI event did not preflight both routes before applying visibility/enabled state");

            check_ui_trace_rejections(std::vector<std::string>(
                host.trace().begin() + static_cast<std::ptrdiff_t>(uiTraceStart),
                host.trace().end()));

            host.set_str_choice_value(
                JuicerParams::kParamScanRoute,
                Spektrafilm::scan_route_key(
                    Spektrafilm::ScanRoute::NegativePrintScan));
            uiTraceStart = host.trace_size();
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
            const UiTraceSummary printUiTrace =
                summarize_ui_trace(host.trace(), uiTraceStart);
            require(
                ui_writes_precede_snapshot(printUiTrace) &&
                    host.param_property_int(
                        JuicerParams::kPrintDiffusionEnabled,
                        kOfxParamPropSecret) == 0 &&
                    host.param_property_int(
                        JuicerParams::kPrintDiffusionEnabled,
                        kOfxParamPropEnabled) == 1 &&
                    host.param_property_int(
                        JuicerParams::kPrintDiffusionFamily,
                        kOfxParamPropSecret) == 0 &&
                    host.param_property_int(
                        JuicerParams::kPrintDiffusionFamily,
                        kOfxParamPropEnabled) == 0 &&
                    host.param_property_int(
                        JuicerParams::kPrintGammaFactor,
                        kOfxParamPropEnabled) == 1,
                "print UI event did not preflight both routes before applying visibility/enabled state");

            host.set_str_choice_value(
                JuicerParams::kParamScanRoute,
                Spektrafilm::scan_route_key(
                    Spektrafilm::ScanRoute::NegativeDirectScan));
            host.set_double_value(kParamExposure, 0.625);
            host.arm_property_reentry(
                JuicerParams::kCameraDiffusionEnabled,
                kOfxParamPropSecret,
                kParamExposure);
            const std::uint64_t reentryCountBefore =
                host.property_reentry_count();
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
            const auto reentryState =
                JuicerTestSupport::EffectTraceObserver::snapshot(effect);
            const auto reentryRender =
                host.render(effect, frame, "ui-property-reentry");
            require(
                host.property_reentry_count() == reentryCountBefore + 1 &&
                    !host.property_reentry_failed() &&
                    reentryState.pendingValid && !reentryState.pendingInvalid &&
                    !reentryState.paramEventsSuppressed &&
                    reentryState.pendingRoute ==
                        Spektrafilm::ScanRoute::NegativeDirectScan &&
                    double_matches(reentryState.pendingExposureEv, 0.625) &&
                    reentryRender.succeeded &&
                    exposure_matches(reentryRender, 0.625),
                "one-shot property-write reentry did not complete with stable route, exposure, and suppression state");

            host.set_double_value(kParamExposure, 0.0);
            host.set_str_choice_value(
                JuicerParams::kParamScanRoute,
                Spektrafilm::scan_route_key(Spektrafilm::kDefaultScanRoute));
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);

            host.set_str_choice_value(JuicerParams::kFilmProfileKey, "missing-profile");
            host.set_str_choice_value(JuicerParams::kParamScanRoute, "missing-route");
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
            require(
                JuicerTestSupport::EffectTraceObserver::snapshot(effect).pendingRoute ==
                    Spektrafilm::ScanRoute::NegativePrintScan,
                "unknown profile and route keys did not use the negative print default");

            host.set_str_choice_value(JuicerParams::kFilmProfileKey, "fujifilm_provia_100f");
            host.set_str_choice_value(JuicerParams::kParamScanRoute, "");
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
            require(
                JuicerTestSupport::EffectTraceObserver::snapshot(effect).pendingRoute ==
                    Spektrafilm::ScanRoute::PositiveDirectScan,
                "empty route key did not use the positive direct default");

            host.set_str_choice_value(
                JuicerParams::kFilmProfileKey, Spektrafilm::kDefaultFilmProfileKey);
            host.set_str_choice_value(
                JuicerParams::kParamScanRoute,
                Spektrafilm::scan_route_key(Spektrafilm::kDefaultScanRoute));
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
            const auto restoredRoute = host.render(effect, frame, "route-fallback-restored");
            require(restoredRoute.succeeded &&
                        restoredRoute.state.recipeHash == first.state.recipeHash,
                    "route fallback test did not restore the original route");

            host.set_time(10.0);
            host.set_double_key(kParamExposure, 10.0, 0.25);
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
            const auto timeTen = host.render(effect, frame, "authored-time-10");
            require(timeTen.succeeded && exposure_matches(timeTen, 0.25) &&
                        has_valid_latch(timeTen) &&
                        timeTen.state.latchSnapshotId !=
                            first.state.latchSnapshotId,
                    "time-10 event did not admit its authored value");

            host.set_time(20.0);
            host.set_double_key(kParamExposure, 20.0, -0.5);
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
            const auto timeTwenty = host.render(effect, frame, "authored-time-20");
            require(timeTwenty.succeeded && exposure_matches(timeTwenty, -0.5) &&
                        has_valid_latch(timeTwenty) &&
                        timeTwenty.state.latchSnapshotId !=
                            timeTen.state.latchSnapshotId,
                    "time-20 event did not admit its authored value");
            require(timeTen.state.recipeHash != timeTwenty.state.recipeHash &&
                        timeTen.outputSignature != timeTwenty.outputSignature,
                    "distinct authored exposure values did not affect recipe/output: " +
                        std::to_string(timeTen.state.recipeHash) + "/" +
                        std::to_string(timeTwenty.state.recipeHash) + " outputs=" +
                        std::to_string(timeTen.outputSignature) + "/" +
                        std::to_string(timeTwenty.outputSignature));

            host.set_time(10.0);
            host.dispatch_event(effect, kParamExposure, OFX::eChangeTime);
            const auto backward = host.render(effect, frame, "backward-seek-10");
            const auto repeated = host.render(effect, frame, "repeat-time-10");
            require(has_valid_latch(backward) &&
                        backward.state.latchSnapshotId !=
                            timeTwenty.state.latchSnapshotId &&
                        exposure_matches(backward, 0.25) &&
                        backward.state.recipeHash == timeTen.state.recipeHash &&
                        backward.outputSignature == timeTen.outputSignature,
                    "backward seek did not restore the time-10 contract");
            require(has_valid_latch(repeated) &&
                        repeated.state.buildCounter == backward.state.buildCounter &&
                        repeated.state.recipeHash == backward.state.recipeHash &&
                        repeated.state.latchSnapshotId ==
                            backward.state.latchSnapshotId &&
                        repeated.outputSignature == backward.outputSignature,
                    "same-time repeat did not reuse build/recipe/latch/output");

            const std::uint64_t nestedBeforePreset =
                host.nested_event_count();
            host.set_int_value(JuicerParams::kGrainPreset, 2);
            host.dispatch_event(
                effect,
                JuicerParams::kGrainPreset,
                OFX::eChangeUserEdit);
            const auto presetState =
                JuicerTestSupport::EffectTraceObserver::snapshot(effect);
            require(
                host.nested_event_count() - nestedBeforePreset == 22 &&
                    !host.nested_event_failed() &&
                    !host.nested_contract_failed() &&
                    !presetState.paramEventsSuppressed &&
                    presetState.pendingValid &&
                    presetState.pendingGrain.amplitude == static_cast<float>(std::exp2(-0.57)) &&
                    presetState.pendingGrain.chromaMix == 0.50f &&
                    presetState.pendingGrain.chromaSharedWeight == std::sqrt(0.50f) &&
                    presetState.pendingGrain.chromaIndependentWeight == std::sqrt(0.50f) &&
                    presetState.pendingGrain.particleAreaUm2 == 0.33f &&
                    presetState.pendingGrain.correlationSigmaPx == 0.56f &&
                    presetState.pendingGrain.sizeMixScale == 16.0f &&
                    double_matches(
                        host.double_value(JuicerParams::kGrainAmplitude),
                        -0.57) &&
                    double_matches(
                        host.double_value(JuicerParams::kGrainBlur),
                        0.56) &&
                    double_matches(
                        host.double_value(JuicerParams::kGrainChroma),
                        0.50) &&
                    double_matches(
                        host.double_value(JuicerParams::kGrainTexture),
                        0.35) &&
                    double_matches(
                        host.double_value(JuicerParams::kGrainParticleAreaUm2),
                        0.33) &&
                    double_matches(
                        host.double_value(JuicerParams::kGrainSizeMixScale),
                        16.0),
                "grain preset did not apply the coarse values with 22 "
                "synchronously suppressed nested callbacks: " +
                    host.nested_contract_diagnostic());
            const auto preset = host.render(effect, frame, "grain-preset");

            host.set_double_value(JuicerParams::kGrainParticleAreaUm2, 0.75);
            host.set_double_value(JuicerParams::kGrainSizeMixScale, 27.0);
            require(
                double_matches(
                    host.double_value(JuicerParams::kGrainParticleAreaUm2),
                    0.75) &&
                    double_matches(
                        host.double_value(JuicerParams::kGrainSizeMixScale),
                        27.0),
                "grain reset precondition values were not authored");
            const std::uint64_t nestedBeforeReset =
                host.nested_event_count();
            host.dispatch_event(
                effect,
                JuicerParams::kGrainResetAdvanced,
                OFX::eChangeUserEdit);
            const auto resetState =
                JuicerTestSupport::EffectTraceObserver::snapshot(effect);
            require(
                host.nested_event_count() - nestedBeforeReset == 16 &&
                    !host.nested_event_failed() &&
                    !host.nested_contract_failed() &&
                    !resetState.paramEventsSuppressed &&
                    double_matches(
                        host.double_value(JuicerParams::kGrainParticleAreaUm2),
                        0.33) &&
                    double_matches(
                        host.double_value(JuicerParams::kGrainSizeMixScale),
                        16.0),
                "grain reset did not restore coarse advanced values with 16 "
                "synchronously suppressed nested callbacks: " +
                    host.nested_contract_diagnostic());
            const auto reset = host.render(effect, frame, "grain-reset");
            require(preset.succeeded && reset.succeeded &&
                        has_valid_latch(preset) &&
                        has_valid_latch(reset) &&
                        preset.state.latchSnapshotId ==
                            backward.state.latchSnapshotId &&
                        reset.state.latchSnapshotId ==
                            preset.state.latchSnapshotId &&
                        preset.outputSignature == backward.outputSignature &&
                        reset.outputSignature == backward.outputSignature &&
                        !host.nested_event_failed() &&
                        !host.nested_contract_failed(),
                    "preset/reset suppression or inactive-output contract failed");

            host.set_double_key(kParamExposure, 10.0, 0.5);
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
            const auto edit = host.render(effect, frame, "edit-ev-0.5");
            host.set_double_key(kParamExposure, 10.0, 0.25);
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
            const auto undo = host.render(effect, frame, "undo-ev-0.25");
            host.set_double_key(kParamExposure, 10.0, 0.5);
            host.dispatch_event(effect, kParamExposure, OFX::eChangeUserEdit);
            const auto redo = host.render(effect, frame, "redo-ev-0.5");
            require(has_valid_latch(edit) && has_valid_latch(undo) &&
                        has_valid_latch(redo) &&
                        edit.state.latchSnapshotId !=
                            reset.state.latchSnapshotId &&
                        undo.state.latchSnapshotId !=
                            edit.state.latchSnapshotId &&
                        redo.state.latchSnapshotId !=
                            undo.state.latchSnapshotId &&
                        edit.state.recipeHash == redo.state.recipeHash &&
                        edit.outputSignature == redo.outputSignature &&
                        undo.state.recipeHash == timeTen.state.recipeHash &&
                        undo.outputSignature == timeTen.outputSignature,
                    "undo/redo-like edits did not restore recipe/output identity");

            host.set_double_key(
                JuicerParams::kCameraFilmFormatMm,
                10.0,
                std::numeric_limits<double>::quiet_NaN());
            host.dispatch_event(
                effect,
                JuicerParams::kCameraFilmFormatMm,
                OFX::eChangeUserEdit);
            const auto invalidBeforeRender =
                JuicerTestSupport::EffectTraceObserver::snapshot(effect);
            const auto rejected = host.render(
                effect,
                frame,
                "invalid-film-format",
                true);
            require(invalidBeforeRender.pendingInvalid && rejected.rejected &&
                        rejected.state.pendingInvalid &&
                        has_valid_latch(rejected) &&
                        rejected.state.latchSnapshotId ==
                            redo.state.latchSnapshotId &&
                        rejected.state.buildCounter == redo.state.buildCounter &&
                        rejected.state.recipeHash == redo.state.recipeHash,
                    "invalid controls did not block admission while retaining publication");

            host.set_double_key(JuicerParams::kCameraFilmFormatMm, 10.0, 35.0);
            host.dispatch_event(
                effect,
                JuicerParams::kCameraFilmFormatMm,
                OFX::eChangeUserEdit);
            const auto recovered = host.render(effect, frame, "valid-recovery");
            require(recovered.succeeded && exposure_matches(recovered, 0.5) &&
                        has_valid_latch(recovered) &&
                        recovered.state.latchSnapshotId ==
                            redo.state.latchSnapshotId &&
                        recovered.state.recipeHash == redo.state.recipeHash &&
                        recovered.outputSignature == redo.outputSignature,
                    "valid controls did not recover the prior effective contract");

            // Observe real OFX capture, including the one film narrowing and
            // already resolved grain controls before development construction.
            for (bool print : {false, true}) {
                host.set_str_choice_value(JuicerParams::kParamScanRoute,
                                          Spektrafilm::scan_route_key(print ? Spektrafilm::ScanRoute::NegativePrintScan
                                                                            : Spektrafilm::ScanRoute::NegativeDirectScan));
                for (double filmGamma : {0.05, 1.234567890123, 4.0}) {
                    for (double printGamma : {0.5, 1.234567890123, 2.0}) {
                        host.set_double_value(JuicerParams::kFilmGammaFactor, filmGamma);
                        host.set_double_value(JuicerParams::kPrintGammaFactor, printGamma);
                        host.dispatch_event(effect, JuicerParams::kFilmGammaFactor, OFX::eChangeUserEdit);
                        const auto observed = JuicerTestSupport::EffectTraceObserver::snapshot(effect);
                        require(observed.pendingValid &&
                                    std::bit_cast<std::uint32_t>(observed.pendingFilmGamma) ==
                                        std::bit_cast<std::uint32_t>(static_cast<float>(filmGamma)) &&
                                    std::bit_cast<std::uint64_t>(observed.pendingPrintGamma) ==
                                        std::bit_cast<std::uint64_t>(printGamma),
                                "OFX film-f32/print-f64 capture bits changed");
                    }
                }
                host.set_double_value(JuicerParams::kFilmGammaFactor, 1.0);
                host.set_double_value(JuicerParams::kPrintGammaFactor, 0.25);
                host.dispatch_event(effect, JuicerParams::kPrintGammaFactor, OFX::eChangeUserEdit);
                const auto printInvalid = JuicerTestSupport::EffectTraceObserver::snapshot(effect);
                require(print ? printInvalid.pendingInvalid : printInvalid.pendingValid,
                        "OFX print gamma validation lost its print-only range");
                host.set_double_value(JuicerParams::kPrintGammaFactor, 1.0);
                host.dispatch_event(effect, JuicerParams::kPrintGammaFactor, OFX::eChangeUserEdit);
                const auto retained = JuicerTestSupport::EffectTraceObserver::admit(effect);
                require(print ? bool(retained.printState) : bool(retained.directState),
                        "gamma recovery failed to admit a complete state");
                for (double invalid : {std::nextafter(0.05, 0.0), std::nextafter(4.0, 5.0), std::numeric_limits<double>::quiet_NaN()}) {
                    host.set_double_value(JuicerParams::kFilmGammaFactor, invalid);
                    host.dispatch_event(effect, JuicerParams::kFilmGammaFactor, OFX::eChangeUserEdit);
                    require(JuicerTestSupport::EffectTraceObserver::snapshot(effect).pendingInvalid &&
                                JuicerTestSupport::EffectTraceObserver::admit(effect).status ==
                                    PendingRenderAdmissionStatus::InvalidSnapshotControls,
                            "invalid authored film gamma admitted an older publication");
                }
                host.set_double_value(JuicerParams::kFilmGammaFactor, 1.0);
                for (int policy : {0, 1, 2}) {
                    host.set_int_value(JuicerParams::kGrainActive, policy != 0);
                    host.set_int_value(JuicerParams::kGrainSublayersActive, policy != 1);
                    host.set_double_value(JuicerParams::kGrainAmplitude, -1.25);
                    host.set_double_value(JuicerParams::kGrainChroma, 0.375);
                    host.dispatch_event(effect, JuicerParams::kGrainActive, OFX::eChangeUserEdit);
                    const auto observed = JuicerTestSupport::EffectTraceObserver::snapshot(effect);
                    const auto& grain = observed.pendingGrain;
                    require(observed.pendingValid && grain.active == (policy != 0) &&
                                grain.sublayersActive == (policy != 1) &&
                                grain.amplitude == static_cast<float>(std::exp2(-1.25)) &&
                                grain.chromaMix == 0.375f &&
                                grain.chromaSharedWeight == std::sqrt(0.625f) &&
                                grain.chromaIndependentWeight == std::sqrt(0.375f),
                            "OFX grain policy or resolved amount/chroma capture changed");
                    const auto admitted = JuicerTestSupport::EffectTraceObserver::admit(effect);
                    require(print ? bool(admitted.printState) : bool(admitted.directState),
                            "grain capture did not admit a complete state");
                    const auto* recipe = print ? &admitted.printState->recipe : &admitted.directState->recipe;
                    require(recipe->filmDevelop.densityCurvesLayersRequired == (policy == 2) &&
                                (policy == 0 || recipe->visualGrain.amplitude == grain.amplitude),
                            "development repeated or changed resolved grain policy");
                }
            }
            std::cout << "PASS adapter/film-f32-print-f64-grain-capture-and-recovery\n";
            std::ostringstream grainCapture;

            // Current OFX getters own these resolved controls until S6.B/C.
            for (bool print : {false, true}) {
                host.set_str_choice_value(JuicerParams::kParamScanRoute,
                                          Spektrafilm::scan_route_key(print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan));
                for (int presetIndex : {0, 1, 2}) {
                    const std::array<double, 3> presetEv{-1.396, -1.20, -0.57};
                    const std::array<double, 3> presetChroma{0.0, 0.3, 0.5};
                    host.set_int_value(JuicerParams::kGrainPreset, presetIndex);
                    const auto nested = host.nested_event_count();
                    host.dispatch_event(effect, JuicerParams::kGrainPreset, OFX::eChangeUserEdit);
                    require(host.nested_event_count() - nested == 22, "each preset suppresses its 22 nested callbacks");
                    for (int policy : {0, 1, 2}) {
                        const std::array<double, 11> amounts{-3.0, 0.0, 3.0, -1.25, 1.234567890123, -9.0, 9.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), std::nextafter(1.234567890123, 2.0)};
                        const double threshold = static_cast<double>(0.999f);
                        const std::array<double, 11> chromas{0.0, 0.375, 1.0, std::nextafter(threshold, 0.0), std::nextafter(threshold, 1.0), static_cast<double>(std::nextafter(0.999f, 0.0f)), static_cast<double>(std::nextafter(0.999f, 1.0f)), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.0, 2.0};
                        for (std::size_t index = 0; index < amounts.size(); ++index) {
                            host.set_int_value(JuicerParams::kGrainActive, policy != 0);
                            host.set_int_value(JuicerParams::kGrainSublayersActive, policy != 1);
                            host.set_double_value(JuicerParams::kGrainAmplitude, amounts[index]);
                            host.set_double_value(JuicerParams::kGrainChroma, chromas[index]);
                            const auto start = host.trace_size();
                            const auto timed = host.at_time_get_count();
                            host.dispatch_event(effect, JuicerParams::kGrainActive, OFX::eChangeTime);
                            const auto observed = JuicerTestSupport::EffectTraceObserver::snapshot(effect);
                            const auto& g = observed.pendingGrain;
                            const double ev = std::isfinite(amounts[index]) ? std::clamp(amounts[index], -3.0, 3.0) : presetEv[presetIndex];
                            const float chroma = static_cast<float>(std::isfinite(chromas[index]) ? std::clamp(chromas[index], 0.0, 1.0) : presetChroma[presetIndex]);
                            require(observed.pendingValid && host.at_time_get_count() == timed &&
                                        g.amplitude == static_cast<float>(std::exp2(ev)) && g.chromaMix == chroma &&
                                        g.chromaSharedWeight == std::sqrt(std::max(0.0f, 1.0f - chroma)) &&
                                        g.chromaIndependentWeight == std::sqrt(std::max(0.0f, chroma)),
                                    "actual resolved grain capture words");
                            const std::array<const char*, 22> getters{
                                JuicerParams::kGrainActive, JuicerParams::kGrainPreset, JuicerParams::kGrainSublayersActive, JuicerParams::kGrainAmplitude, JuicerParams::kGrainBlur, JuicerParams::kGrainSharpness, JuicerParams::kGrainChroma, JuicerParams::kGrainTexture, JuicerParams::kGrainParticleAreaUm2, JuicerParams::kGrainSizeMixScale, JuicerParams::kGrainSizeMixWeight, JuicerParams::kGrainSizeMixWeightMid, JuicerParams::kGrainBlurDyeCloudsUm, JuicerParams::kGrainParticleScale, JuicerParams::kGrainParticleScaleLayers, JuicerParams::kGrainDensityMin, JuicerParams::kGrainUniformity, JuicerParams::kGrainClumpTemporalMix, JuicerParams::kGrainClumpMorphPeriodSec, JuicerParams::kGrainMicroStructure, JuicerParams::kGrainDebugView, nullptr};
                            bool ordered = false;
                            for (std::size_t i = start; i < host.trace().size(); ++i) {
                                std::size_t j = 0;
                                while (j + 1 < getters.size() && i + j < host.trace().size() &&
                                       host.trace()[i + j].starts_with("GET_CURRENT name=" + std::string(getters[j]) + " "))
                                    ++j;
                                ordered = ordered || j + 1 == getters.size();
                            }
                            require(ordered, "all grain current getters retain exact order even disabled");
                            const auto admitted = JuicerTestSupport::EffectTraceObserver::admit(effect);
                            require(print ? bool(admitted.printState) : bool(admitted.directState), "grain capture completes route");
                            const auto& recipe = print ? admitted.printState->recipe : admitted.directState->recipe;
                            require(recipe.filmDevelop.densityCurvesLayersRequired == (policy == 2) &&
                                        (policy == 0 || (recipe.visualGrain.amplitude == g.amplitude && recipe.visualGrain.chromaSharedWeight == g.chromaSharedWeight &&
                                                         recipe.visualGrain.chromaIndependentWeight == g.chromaIndependentWeight)),
                                    "resolved grain is consumed once");
                            grainCapture << "GRAIN_CAPTURE " << print << ' ' << presetIndex << ' ' << policy << ' ' << index
                                         << ' ' << std::bit_cast<std::uint64_t>(amounts[index]) << ' ' << std::bit_cast<std::uint64_t>(chromas[index]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleAreaUm2);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleScaleCmy[0]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleScaleCmy[1]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleScaleCmy[2]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleScaleLayers[0]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleScaleLayers[1]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleScaleLayers[2]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.visualParticleDensityMinCmy[0]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.visualParticleDensityMinCmy[1]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.visualParticleDensityMinCmy[2]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.uniformityCmy[0]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.uniformityCmy[1]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.uniformityCmy[2]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.correlationSigmaPx);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.dyeCloudBlurUm);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.microStructure[0]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.microStructure[1]);
                            grainCapture << ' ' << g.nSubLayers;
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.amplitude);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.chromaMix);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.chromaSharedWeight);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.chromaIndependentWeight);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.coarseWeight);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.midWeight);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.sizeMixScale);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.clumpTemporalMix);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.clumpMorphPeriodSec);
                            grainCapture << ' ' << g.debugView;
                            grainCapture << ' ' << recipe.visualGrain.hash << ' ' << recipe.visualGrain.densityCurvesLayersHash << ' ' << recipe.hash << '\n';
                        }
                    }
                }
            }
            host.set_int_value(JuicerParams::kGrainActive, 0);
            host.set_double_value(JuicerParams::kGrainAmplitude, -1.25);
            host.set_double_value(JuicerParams::kGrainChroma, 0.375);


            for (bool print : {false, true}) {
                host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan));
                host.set_int_value(JuicerParams::kGrainActive, 1);
                host.set_int_value(JuicerParams::kGrainSublayersActive, 1);
                const auto record = [&](std::string_view label) {
                    const auto observed = JuicerTestSupport::EffectTraceObserver::snapshot(effect);
                    require(observed.pendingValid, "grain edit capture valid");
                    const auto& g = observed.pendingGrain;
                    const auto admitted = JuicerTestSupport::EffectTraceObserver::admit(effect);
                    require(print ? bool(admitted.printState) : bool(admitted.directState), "grain edit completes publication");
                    const auto& recipe = print ? admitted.printState->recipe : admitted.directState->recipe;
                    grainCapture << "GRAIN_EDIT_CAPTURE " << print << ' ' << label;
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleAreaUm2);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleScaleCmy[0]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleScaleCmy[1]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleScaleCmy[2]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleScaleLayers[0]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleScaleLayers[1]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particleScaleLayers[2]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.visualParticleDensityMinCmy[0]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.visualParticleDensityMinCmy[1]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.visualParticleDensityMinCmy[2]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.uniformityCmy[0]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.uniformityCmy[1]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.uniformityCmy[2]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.correlationSigmaPx);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.dyeCloudBlurUm);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.microStructure[0]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.microStructure[1]);
                    grainCapture << ' ' << g.nSubLayers;
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.amplitude);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.chromaMix);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.chromaSharedWeight);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.chromaIndependentWeight);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.coarseWeight);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.midWeight);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.sizeMixScale);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.clumpTemporalMix);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.clumpMorphPeriodSec);
                    grainCapture << ' ' << g.debugView;
                    grainCapture << ' ' << recipe.visualGrain.hash << ' ' << recipe.visualGrain.densityCurvesLayersHash << ' ' << recipe.hash << '\n';
                    const auto repeated = JuicerTestSupport::EffectTraceObserver::admit(effect);
                    require(print ? repeated.printState == admitted.printState : repeated.directState == admitted.directState, "same-time capture repeats reuse immutable publication");
                };
                for (int choice : {-8, 0, 1, 2, 8}) {
                    host.set_int_value(JuicerParams::kGrainPreset, choice);
                    const auto before = host.nested_event_count();
                    host.dispatch_event(effect, JuicerParams::kGrainPreset, OFX::eChangeUserEdit);
                    require(host.nested_event_count() - before == 22, "invalid choice clamps before preset edits");
                    record("preset-" + std::to_string(choice));
                    const auto unchanged = host.nested_event_count();
                    host.dispatch_event(effect, JuicerParams::kGrainPreset, OFX::eChangeTime);
                    host.dispatch_event(effect, JuicerParams::kGrainPreset, OFX::eChangePluginEdit);
                    require(host.nested_event_count() == unchanged, "non-user preset events do not repeat setters");
                    const double nan = std::numeric_limits<double>::quiet_NaN();
                    for (const char* name : {JuicerParams::kGrainAmplitude, JuicerParams::kGrainBlur, JuicerParams::kGrainSharpness, JuicerParams::kGrainChroma, JuicerParams::kGrainTexture, JuicerParams::kGrainParticleAreaUm2, JuicerParams::kGrainSizeMixScale, JuicerParams::kGrainSizeMixWeight, JuicerParams::kGrainSizeMixWeightMid, JuicerParams::kGrainBlurDyeCloudsUm, JuicerParams::kGrainClumpTemporalMix, JuicerParams::kGrainClumpMorphPeriodSec})
                        host.set_double_value(name, nan);
                    for (const char* name : {JuicerParams::kGrainParticleScale, JuicerParams::kGrainParticleScaleLayers, JuicerParams::kGrainDensityMin, JuicerParams::kGrainUniformity})
                        host.set_double3_value(name, {nan, nan, nan});
                    host.set_double2_value(JuicerParams::kGrainMicroStructure, {nan, nan});
                    host.set_int_value(JuicerParams::kGrainDebugView, choice);
                    host.dispatch_event(effect, JuicerParams::kGrainActive, OFX::eChangeTime);
                    const auto fallback = JuicerTestSupport::EffectTraceObserver::snapshot(effect).pendingGrain;
                    require(fallback.particleScaleCmy == std::array<float, 3>{1.48f, 1.48f, 1.48f}, "triplet fallback repeats preset master without ratios");
                    record("fallback-" + std::to_string(choice));
                }
                host.set_int_value(JuicerParams::kGrainPreset, 1);
                host.dispatch_event(effect, JuicerParams::kGrainPreset, OFX::eChangeUserEdit);
                host.set_double3_value(JuicerParams::kGrainParticleScale, {0.5, 1.0, 1.5});
                host.set_double_value(JuicerParams::kGrainParticleScaleMaster, 1.25);
                host.dispatch_event(effect, JuicerParams::kGrainParticleScaleMaster, OFX::eChangeUserEdit);
                require(JuicerTestSupport::EffectTraceObserver::snapshot(effect).pendingGrain.particleScaleCmy == std::array<float, 3>{0.625f, 1.25f, 1.875f}, "master edit writes normalized current ratios once");
                record("master-ratios");
                host.set_double3_value(JuicerParams::kGrainParticleScale, {2.0, 3.0, 4.0});
                host.dispatch_event(effect, JuicerParams::kGrainParticleScale, OFX::eChangeUserEdit);
                require(JuicerTestSupport::EffectTraceObserver::snapshot(effect).pendingGrain.particleScaleCmy == std::array<float, 3>{2.0f, 3.0f, 4.0f}, "manual triplets are captured without multiplying master");
                record("manual-custom");
                host.set_double_value(JuicerParams::kGrainSharpness, 0.25);
                host.dispatch_event(effect, JuicerParams::kGrainSharpness, OFX::eChangeUserEdit);
                host.set_double_value(JuicerParams::kGrainTexture, 0.75);
                host.dispatch_event(effect, JuicerParams::kGrainTexture, OFX::eChangeUserEdit);
                record("linked-sharpness-texture");
                host.dispatch_event(effect, JuicerParams::kGrainResetAdvanced, OFX::eChangeUserEdit);
                record("reset-advanced");
                host.set_int_value(JuicerParams::kGrainActive, 0);
                for (double time : {2.0, 0.0, 2.0, 2.0}) {
                    host.set_time(time);
                    host.dispatch_event(effect, JuicerParams::kGrainActive, OFX::eChangeTime);
                    record(time == 0.0 ? "disabled-backward" : "disabled-repeat");
                }
            }
            host.set_int_value(JuicerParams::kGrainActive, 0);
            host.set_double_value(JuicerParams::kGrainAmplitude, -1.25);
            host.set_double_value(JuicerParams::kGrainChroma, 0.375);

            const auto grainWords = grainCapture.str();
            require(std::count(grainWords.begin(), grainWords.end(), '\n') == 234, "complete 198 capture and 36 edit native witnesses");
            if (argc == 6) {
                const std::filesystem::path manifestPath = argv[4];
                std::ifstream manifestStream(manifestPath);
                require(bool(manifestStream), "grain capture manifest open");
                const auto manifest = nlohmann::json::parse(manifestStream);
                const auto expectedPath = manifestPath.parent_path() / manifest.at("presets").at(argv[5]).at("ofx").get<std::string>();
                std::ifstream expected(expectedPath);
                require(bool(expected), "independent native OFX grain words open");
                std::istringstream actual(grainWords);
                std::string expectedWord, actualWord;
                while (expected >> expectedWord)
                    require(bool(actual >> actualWord) && expectedWord == actualWord, "exact native authored/captured/recipe/final grain words");
                require(!(actual >> actualWord), "no extra grain capture records");
            }
            std::cout << grainWords;

            for (bool positive : {false, true})
                for (bool print : {false, true}) {
                    host.set_str_choice_value(JuicerParams::kFilmProfileKey, positive ? "fujifilm_provia_100f" : "kodak_portra_400");
                    host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan));
                    host.set_int_value(JuicerParams::kGrainActive, 0);
                    const auto gets = [&](std::size_t start, const char* name) {
                        const std::string prefix = "GET_CURRENT name=" + std::string(name) + " ";
                        const auto begin = host.trace().begin() + static_cast<std::ptrdiff_t>(start);
                        const std::string stock = "GET_CURRENT name=" + std::string(JuicerParams::kDirCouplersGammaUseStock) + " ";
                        const auto marker = std::find_if(begin, host.trace().end(), [&](const std::string& row) {
                            return row.starts_with(stock);
                        });
                        require(marker != host.trace().end() && marker != begin && (marker - 1)->starts_with("GET_CURRENT name=" + std::string(JuicerParams::kDirCouplersActive) + " "), "DIR capture begins with active then stock gamma");
                        // UI enablement reads core/tail before snapshot capture.
                        return std::count_if(marker - 1, host.trace().end(), [&](const std::string& row) {
                            return row.starts_with(prefix);
                        });
                    };
                    const auto capture = [&]() {
                        host.dispatch_event(effect, JuicerParams::kDirCouplersActive, OFX::eChangeUserEdit);
                        return JuicerTestSupport::EffectTraceObserver::snapshot(effect);
                    };
                    host.set_int_value(JuicerParams::kDirCouplersActive, 0);
                    host.set_int_value(JuicerParams::kDirCouplersGammaUseStock, 1);
                    host.set_double_value(JuicerParams::kDirCouplersAmount, std::numeric_limits<double>::quiet_NaN());
                    auto start = host.trace_size();
                    auto observed = capture();
                    require(observed.pendingValid && !observed.pendingDir.active && gets(start, JuicerParams::kDirCouplersAmount) == 0, "disabled OFX DIR skips amount capture");
                    host.set_int_value(JuicerParams::kDirCouplersActive, 1);
                    host.set_double_value(JuicerParams::kDirCouplersAmount, 0);
                    start = host.trace_size();
                    observed = capture();
                    require(observed.pendingValid && gets(start, JuicerParams::kDirCouplersInhibitionSameLayer) == 0 && gets(start, JuicerParams::kDirCouplersLangmuirDonorKRgb) == 0 && gets(start, JuicerParams::kDirCouplersLangmuirReceiverKRgb) == 0, "zero-amount OFX DIR skips chemistry");
                    const double authored = 1.234567890123;
                    host.set_double_value(JuicerParams::kDirCouplersAmount, authored);
                    host.set_double_value(JuicerParams::kDirCouplersDiffusionSizeUm, 0);
                    host.set_double_value(JuicerParams::kDirCouplersDiffusionTailWeight, std::numeric_limits<double>::quiet_NaN());
                    host.set_double3_value(JuicerParams::kDirCouplersGammaSameLayerRgb, {std::numeric_limits<double>::quiet_NaN(), 0, 0});
                    const char* consumed = positive ? JuicerParams::kDirCouplersLangmuirReceiverKRgb : JuicerParams::kDirCouplersLangmuirDonorKRgb;
                    const char* unused = positive ? JuicerParams::kDirCouplersLangmuirDonorKRgb : JuicerParams::kDirCouplersLangmuirReceiverKRgb;
                    host.set_double3_value(consumed, {authored, authored, authored});
                    host.set_double3_value(unused, {std::numeric_limits<double>::quiet_NaN(), 0, 0});
                    start = host.trace_size();
                    observed = capture();
                    require(observed.pendingValid, "stock DIR capture invalid: " + observed.pendingDiagnostic);
                    require(observed.pendingDir.amount == static_cast<float>(authored), "stock DIR amount narrowing");
                    require(gets(start, consumed) == 1, "consumed DIR K getter count=" + std::to_string(gets(start, consumed)));
                    require(gets(start, unused) == 0, "unused DIR K getter count=" + std::to_string(gets(start, unused)));
                    require(gets(start, JuicerParams::kDirCouplersGammaSameLayerRgb) == 0, "stock DIR gamma bypass");
                    require(gets(start, JuicerParams::kDirCouplersDiffusionTailWeight) == 0, "core-zero DIR tail bypass");
                    host.set_int_value(JuicerParams::kDirCouplersGammaUseStock, 0);
                    host.set_double3_value(JuicerParams::kDirCouplersGammaSameLayerRgb, {authored, authored, authored});
                    host.set_double2_value(JuicerParams::kDirCouplersGammaInterlayerRToGb, {0.123456789012, 0.234567890123});
                    host.set_double2_value(JuicerParams::kDirCouplersGammaInterlayerGToRb, {0.345678901234, 0.456789012345});
                    host.set_double2_value(JuicerParams::kDirCouplersGammaInterlayerBToRg, {0.567890123456, 0.678901234567});
                    host.set_double_value(JuicerParams::kDirCouplersDiffusionSizeUm, 20);
                    host.set_double_value(JuicerParams::kDirCouplersDiffusionTailWeight, 0);
                    host.set_double_value(JuicerParams::kDirCouplersDiffusionTailUm, std::numeric_limits<double>::quiet_NaN());
                    start = host.trace_size();
                    observed = capture();
                    const auto& dir = observed.pendingDir;
                    require(observed.pendingValid && dir.gammaSameLayerRgb[0] == static_cast<float>(authored) && dir.gammaInterlayerRToGb[1] == static_cast<float>(0.234567890123) && gets(start, JuicerParams::kDirCouplersDiffusionTailUm) == 0, "custom RGB f64-to-f32 narrowing and zero-tail bypass");
                    host.set_double3_value(consumed, {0.09, 1, 1});
                    observed = capture();
                    require(observed.pendingInvalid, "invalid consumed K blocks current snapshot");
                    host.set_double3_value(consumed, {authored, authored, authored});
                    observed = capture();
                    require(observed.pendingValid, "OFX DIR controls recover");
                    const auto admitted = JuicerTestSupport::EffectTraceObserver::admit(effect);
                    require(print ? bool(admitted.printState) : bool(admitted.directState), "captured DIR controls admit complete route");
                    host.set_int_value(JuicerParams::kDirCouplersActive, 0);
                    host.set_int_value(JuicerParams::kDirCouplersGammaUseStock, 1);
                    host.set_double3_value(JuicerParams::kDirCouplersLangmuirDonorKRgb, {1, 1, 1});
                    host.set_double3_value(JuicerParams::kDirCouplersLangmuirReceiverKRgb, {1, 1, 1});
                    host.set_double_value(JuicerParams::kDirCouplersDiffusionTailWeight, 0.03);
                    host.set_double_value(JuicerParams::kDirCouplersDiffusionTailUm, 200);
                }
            std::cout << "PASS adapter/DIR-conditional-capture-narrowing-polarity-and-recovery\n";

            require(host.current_get_count() != 0 &&
                        host.at_time_get_count() == 0,
                    "adapter getter selection was not current-value-only");
        }
        require(fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr).category == FJ_STATUS_SUCCESS,
                "adapter terminal shutdown failed");
        host.write_trace(tracePath);
        std::cout << "PASS adapter/first-render-current-value\n"
                  << "PASS adapter/two-authored-times\n"
                  << "PASS adapter/backward-seek-repeat\n"
                  << "PASS adapter/preset-reset-suppression\n"
                  << "PASS adapter/undo-redo-identity\n"
                  << "PASS adapter/invalid-valid-recovery\n"
                  << "PASS adapter/image-release-obligation\n"
                  << "TRACE " << tracePath.string() << " entries="
                  << host.trace_size() << " current_gets="
                  << host.current_get_count() << " at_time_gets="
                  << host.at_time_get_count() << " nested_events="
                  << host.nested_event_count() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL adapter trace: " << error.what() << '\n';
        fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
        return 1;
    } catch (...) {
        std::cerr << "FAIL adapter trace: unknown exception\n";
        fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
        return 1;
    }
}
