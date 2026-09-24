#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "JuicerEffect.h"
#include "JuicerState.h"
#include "ParamNames.h"
#include "ProcessRoot.h"
#include "SpectralProcessing.h"
#include "ofxsSupportPrivate.h"

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray&) {}

namespace JuicerTestSupport {

    class EffectTraceObserver final {
    public:
        struct Snapshot {
            bool pendingValid = false;
            bool pendingInvalid = false;
            bool pendingPrintRoute = false;
            double pendingExposureEv = 0.0;
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
                    result.pendingExposureEv =
                        valid->params.cameraExposureCompensationEv;
                    result.pendingFullHash = valid->fullHash;
                    result.pendingPrintRoute =
                        Spektrafilm::scan_route_is_print(
                            valid->params.scanRoute);
                } else if (std::holds_alternative<
                               PendingParamsState::InvalidSnapshotControls>(
                               state.pending.value)) {
                    result.pendingInvalid = true;
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
    };

} // namespace JuicerTestSupport

namespace {

    struct PropertyBag {
        std::map<std::string, std::vector<std::string>> strings;
        std::map<std::string, std::vector<int>> ints;
        std::map<std::string, std::vector<double>> doubles;
        std::map<std::string, std::vector<void*>> pointers;
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
        return handle
                   ? set_property(
                         reinterpret_cast<PropertyBag*>(handle)->ints,
                         name,
                         index,
                         value)
                   : kOfxStatErrBadHandle;
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
        auto dimension = [&](const auto& properties) -> std::size_t {
            const auto found = properties.find(name);
            return found == properties.end() ? 0u : found->second.size();
        };
        *count = static_cast<int>(std::max(
            {dimension(bag.strings),
             dimension(bag.ints),
             dimension(bag.doubles),
             dimension(bag.pointers)}));
        return *count == 0 ? kOfxStatErrUnknown : kOfxStatOK;
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
            configure_effect();
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

        void bind_effect(JuicerEffect& effect) noexcept {
            _effect = &effect;
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

        double double_value(const char* name) {
            const NativeParam& param = require_param(name, ParamKind::Double);
            return param.value_at(_time).numbers[0];
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

        void configure_clips() {
            configure_clip(_source, kOfxImageEffectSimpleSourceClipName);
            configure_clip(_output, kOfxImageEffectOutputClipName);
        }

        static void configure_clip(NativeClip& clip, const char* name) {
            clip.name = name;
            clip.properties.strings[kOfxPropType] = {kOfxTypeClip};
            clip.properties.strings[kOfxPropName] = {name};
            clip.properties.strings[kOfxPropLabel] = {name};
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
            add_boolean(JuicerParams::kScannerUseLut, true);
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
            _effectSuite.clipGetHandle = clip_get_handle;
            _effectSuite.clipGetPropertySet = clip_properties;
            _effectSuite.clipGetImage = clip_get_image;
            _effectSuite.clipReleaseImage = clip_release_image;
            _effectSuite.clipGetRegionOfDefinition = clip_region;
            _effectSuite.abort = effect_abort;

            _parameterSuite.paramSetGetPropertySet = parameter_set_properties;
            _parameterSuite.paramGetHandle = parameter_get_handle;
            _parameterSuite.paramGetPropertySet = parameter_properties;
            _parameterSuite.paramGetValue = parameter_get_value;
            _parameterSuite.paramGetValueAtTime = parameter_get_value_at_time;
            _parameterSuite.paramSetValue = parameter_set_value;
            _parameterSuite.paramSetValueAtTime = parameter_set_value_at_time;
            _messageSuite.message = message;
        }

        inline static NativeHost* s_active = nullptr;
        OfxPropertySuiteV1 _propertySuite{};
        OfxImageEffectSuiteV1 _effectSuite{};
        OfxParameterSuiteV1 _parameterSuite{};
        OfxMessageSuiteV1 _messageSuite{};
        NativeEffectHandle _effectHandle{};
        PropertyBag _paramSetProperties;
        NativeClip _source;
        NativeClip _output;
        std::map<std::string, std::unique_ptr<NativeParam>> _params;
        JuicerEffect* _effect = nullptr;
        double _time = 0.0;
        std::map<OfxPropertySetHandle, ImageLeaseRecord> _imageLeases;
        std::string _imageAuditError;
        bool _imageAuditActive = false;
        std::vector<std::string> _trace;
        std::uint64_t _currentGetCount = 0;
        std::uint64_t _atTimeGetCount = 0;
        std::uint64_t _nestedEventCount = 0;
        bool _insideNestedEvent = false;
        bool _nestedEventFailed = false;
        bool _nestedContractFailed = false;
        std::string _nestedContractDiagnostic;
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

} // namespace

int main(int argc, char** argv) { // NOLINT(bugprone-exception-escape): top-level catches map all fixture failures.
    try {
        if (argc != 3 || std::string_view(argv[1]) != "--trace-output") {
            throw std::runtime_error(
                "usage: JuicerAdapterTraceProbe --trace-output PATH");
        }
        const std::filesystem::path tracePath = argv[2];
        NativeHost host;
        if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
            throw std::runtime_error("CUDA initialization failed");
        }
        JuicerProcess::root().ensure_bootstrap();
        DeviceFrame frame;
        {
            JuicerEffect effect(host.effect_handle());
            host.bind_effect(effect);

            host.set_time(0.0);
            const auto first = host.render(effect, frame, "first-render");
            require(first.succeeded && exposure_matches(first, 0.0) &&
                        has_valid_latch(first),
                    "first render did not acquire the current zero-EV value");

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

            require(host.current_get_count() != 0 &&
                        host.at_time_get_count() == 0,
                    "adapter getter selection was not current-value-only");
        }
        JuicerProcess::root().shutdown();
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
        JuicerProcess::root().shutdown();
        return 1;
    } catch (...) {
        std::cerr << "FAIL adapter trace: unknown exception\n";
        JuicerProcess::root().shutdown();
        return 1;
    }
}
