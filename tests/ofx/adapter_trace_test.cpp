#include "frame_geometry_test_hooks.h"
#include "module_test.h"
#include "../ffi/exposure_fixture.h"
#include "../ffi/print_illuminant_fixture.h"
#include "../ffi/prepared_product_fixture.h"
#include "../ffi/preparation_fixture.h"
#include "../ffi/spatial_optics_fixture.h"
#include "../ffi/juicer_test_api.h"
#include "../ffi/case_results.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <type_traits>
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
#include <functional>
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

#include "ParamNames.h"
#include "ProcessRoot.h"
#include "RenderValues.h"
#include "juicer_cuda_owner.h"
#include "ColorTransforms.h"
#include "ofxMemory.h"
#include "ofxMultiThread.h"

namespace {
    using Json = nlohmann::json;

    constexpr const char* kDefaultFilmProfileKey = "kodak_portra_400";
    constexpr const char* kDefaultPrintProfileKey = "kodak_portra_endura";
    constexpr const char* kParamExposure = "Exposure";
    constexpr const char* kParamCameraAutoExposure = "CameraAutoExposure";
    constexpr const char* kParamSpectralMode = "SpectralUpsampling";
    constexpr const char* kParamOutputColorSpace = "OutputColorSpace";
    constexpr const char* kParamOutputCctfEncoding = "OutputCctfEncoding";

    OfxPlugin* test_plugin = nullptr;
    OfxStatus invoke_effect(OfxPlugin&, const char*, OfxImageEffectHandle, OfxPropertySetHandle = nullptr);

    template <auto Function>
    struct ContainedCallback;
    template <typename Return, typename... Args, Return (*Function)(Args...)>
    struct ContainedCallback<Function> {
        static Return call(Args... args) noexcept {
            try {
                return Function(args...);
            } catch (const std::bad_alloc&) {
                if constexpr (std::is_pointer_v<Return>)
                    return nullptr;
                else
                    return static_cast<Return>(kOfxStatErrMemory);
            } catch (...) {
                if constexpr (std::is_pointer_v<Return>)
                    return nullptr;
                else
                    return static_cast<Return>(kOfxStatFailed);
            }
        }
    };

    class StatusFailure final : public std::exception {
    public:
        explicit StatusFailure(OfxStatus status) noexcept : status_(status) {}
        OfxStatus status() const noexcept {
            return status_;
        }
        const char* what() const noexcept override {
            return "OFX callback failed";
        }

    private:
        OfxStatus status_;
    };

    enum class ChangeReason : std::uint8_t {
        UserEdit,
        PluginEdit,
        Time
    };

    const char* change_reason(ChangeReason reason) noexcept {
        switch (reason) {
            case ChangeReason::UserEdit:
                return kOfxChangeUserEdited;
            case ChangeReason::PluginEdit:
                return kOfxChangePluginEdited;
            case ChangeReason::Time:
                return kOfxChangeTime;
        }
        return nullptr;
    }

    // The plugin and host handle remain live until this instance is destroyed.
    class EffectLease final {
    public:
        EffectLease(OfxPlugin& plugin, OfxImageEffectHandle effect)
            : plugin_(&plugin), effect_(effect) {
            const auto status = invoke_effect(*plugin_, kOfxActionCreateInstance, effect_);
            if (status != kOfxStatOK)
                throw StatusFailure(status);
        }
        EffectLease(const EffectLease&) = delete;
        EffectLease& operator=(const EffectLease&) = delete;
        ~EffectLease() noexcept {
            if (effect_)
                (void)close();
        }
        OfxImageEffectHandle handle() const noexcept {
            return effect_;
        }
        OfxPlugin& plugin() const noexcept {
            return *plugin_;
        }
        OfxStatus close() noexcept {
            auto* const effect = std::exchange(effect_, nullptr);
            return effect ? invoke_effect(*plugin_, kOfxActionDestroyInstance, effect)
                          : kOfxStatErrBadHandle;
        }

    private:
        OfxPlugin* plugin_;
        OfxImageEffectHandle effect_;
    };
} // namespace

extern "C" FjStatus fj_test_ofx_instance_observe(OfxImageEffectHandle, FjInstanceObservation**, FjErrorBuffer*);


namespace JuicerTestSupport {

    PreparationInput read_ofx_host_input(OfxImageEffectHandle effect) {
        FjOfxControlRead* raw = nullptr;
        const OfxStatus status = fj_test_ofx_capture_host_input(effect, &raw, nullptr);
        const auto release = [](FjOfxControlRead* value) noexcept {
            if (value)
                (void)fj_test_ofx_control_input_release(value, nullptr);
        };
        std::unique_ptr<FjOfxControlRead, decltype(release)> owner(raw, release);
        if (status != kOfxStatOK)
            throw StatusFailure(status);
        if (!owner)
            throw std::runtime_error("successful OFX host input read returned no owner");
        FjPreparationParams value{};
        if (fj_test_ofx_control_input_view(owner.get(), &value, nullptr).category != FJ_STATUS_SUCCESS)
            throw std::runtime_error("cannot inspect OFX host input");
        return PreparationInput(value);
    }

    class OfxObservation final {
    public:
        explicit OfxObservation(OfxImageEffectHandle effect) {
            const auto status = fj_test_ofx_instance_observe(effect, &owner_, nullptr);
            if (status.category != FJ_STATUS_SUCCESS || !owner_) {
                if (owner_)
                    (void)fj_test_instance_observation_release(std::exchange(owner_, nullptr), nullptr);
                throw std::runtime_error("cannot observe OFX instance");
            }
            if (fj_test_instance_observation_view(owner_, &view, nullptr).category != FJ_STATUS_SUCCESS) {
                (void)fj_test_instance_observation_release(std::exchange(owner_, nullptr), nullptr);
                throw std::runtime_error("cannot inspect OFX instance observation");
            }
        }
        ~OfxObservation() {
            if (owner_)
                (void)fj_test_instance_observation_release(owner_, nullptr);
        }
        OfxObservation(const OfxObservation&) = delete;
        OfxObservation& operator=(const OfxObservation&) = delete;
        FjInstanceObservationView view{};

    private:
        FjInstanceObservation* owner_ = nullptr;
    };

    PublicationMetrics ofx_metrics(OfxImageEffectHandle effect) {
        const OfxObservation observation(effect);
        const auto& view = observation.view;
        return {view.pending_kind, view.input_identity, view.published_identity, view.build, view.backing_identity, view.published.route, view.published.recipe_hash, view.issued, view.write_revision, view.outstanding_writes, view.diagnostic.count ? std::string(view.diagnostic.data, view.diagnostic.count) : std::string{}};
    }

    PreparationInput ofx_captured_input(OfxImageEffectHandle effect) {
        const OfxObservation observation(effect);
        if (observation.view.pending_kind != 4)
            throw std::runtime_error("OFX instance has no captured input");
        return PreparationInput(observation.view.captured);
    }

    PublicationAdmission ofx_admission(OfxImageEffectHandle effect) {
        std::array<char, 2048> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        FjAdmission admission{};
        PublicationAdmission result;
        result.delivery = fj_test_ofx_instance_admission(effect, &admission, &error);
        const auto release = [](FjPreparedState* state) noexcept {
            if (state)
                (void)fj_test_prepared_state_release(state, nullptr);
        };
        std::unique_ptr<FjPreparedState, decltype(release)> partial(admission.state, release);
        result.kind = admission.kind;
        result.diagnostic.assign(text.data(), std::min(error.length, text.size()));
        if (admission.state) {
            if (result.delivery.category != FJ_STATUS_SUCCESS || admission.kind != FJ_ADMISSION_READY)
                throw std::runtime_error("unexpected state in failed OFX admission");
            auto held = PreparedStateHold::take_admission(admission);
            (void)partial.release();
            result.state = std::make_shared<PreparedStateHold>(std::move(held));
        }
        return result;
    }

    class EffectObservation final {
    public:
        struct Snapshot {
            bool pendingValid = false;
            bool pendingInvalid = false;
            std::string pendingDiagnostic;
            bool pendingPrintRoute = false;
            Spektrafilm::ScanRoute pendingRoute = Spektrafilm::kDefaultScanRoute;
            FjPreparationParams input{};
            double pendingExposureEv = 0.0;
            float pendingFilmGamma = 0.0f;
            double pendingPrintGamma = 0.0;
            std::uint64_t pendingFullHash = 0;
            std::uint64_t buildCounter = 0;
            std::uint64_t recipeHash = 0;
            std::uint64_t lastParamHash = 0;
            bool directPublished = false;
            bool printPublished = false;
            bool paramEventsSuppressed = false;
            bool latchValid = false;
            std::uint64_t latchSnapshotId = 0;
            std::uint64_t latchFrameToken = 0;
        };

        static Snapshot snapshot(const EffectLease& effect) {
            Snapshot result{};
            result.input = PreparationInput{}.values;
            const OfxObservation observation(effect.handle());
            const auto& view = observation.view;
            result.pendingValid = view.pending_kind == 4;
            result.pendingInvalid = view.pending_kind == 5;
            if (result.pendingValid) {
                result.input = view.captured;
                result.input.film_profile_key = {};
                result.input.print_profile_key = {};
                result.pendingExposureEv = result.input.camera_exposure_compensation_ev;
                result.pendingFilmGamma = result.input.film_gamma_factor;
                result.pendingPrintGamma = result.input.print_gamma_factor;
                result.pendingFullHash = view.input_identity;
                result.pendingRoute = static_cast<Spektrafilm::ScanRoute>(result.input.scan_route);
                result.pendingPrintRoute = (result.input.scan_route & 1u) != 0;
            } else if (result.pendingInvalid && view.diagnostic.count) {
                result.pendingDiagnostic.assign(view.diagnostic.data, view.diagnostic.count);
            }
            result.directPublished = view.build != 0 && (view.published.route & 1u) == 0;
            result.printPublished = view.build != 0 && (view.published.route & 1u) != 0;
            result.buildCounter = view.build;
            result.recipeHash = view.published.recipe_hash;
            result.lastParamHash = view.published_identity;
            const int active = fj_test_ofx_host_call_active(effect.handle());
            if (active < 0)
                throw std::runtime_error("cannot observe OFX host-call suppression");
            result.paramEventsSuppressed = active != 0;
            std::uint32_t valid = 0;
            FjSubmission submission{};
            FjCudaContext context{};
            if (fj_test_ofx_instance_submission(effect.handle(), &valid, &submission, &context, nullptr).category != FJ_STATUS_SUCCESS)
                throw std::runtime_error("cannot observe OFX submission");
            result.latchValid = valid != 0;
            if (result.latchValid) {
                result.latchSnapshotId = submission.submission_id;
                result.latchFrameToken = submission.frame_token;
            }
            return result;
        }
        static PublicationAdmission admit(const EffectLease& effect) {
            return ofx_admission(effect.handle());
        }
        static PreparationInput captured(const EffectLease& effect) {
            return ofx_captured_input(effect.handle());
        }
        static PreparationInput read_host_snapshot(const EffectLease& effect) {
            return read_ofx_host_input(effect.handle());
        }
        static bool capture_is_current(const EffectLease& effect, std::uint64_t ticket) {
            std::uint32_t current = 0;
            if (fj_test_ofx_capture_is_current(effect.handle(), ticket, &current, nullptr).category != FJ_STATUS_SUCCESS)
                throw std::runtime_error("cannot query current OFX capture");
            return current != 0;
        }
        static PublicationMetrics metrics(const EffectLease& effect) {
            return ofx_metrics(effect.handle());
        }
    };

} // namespace JuicerTestSupport

namespace {

    struct PropertyBag {
        using IntWriteObserver =
            OfxStatus (*)(void*, const char*, const char*, int);

        std::map<std::string, std::vector<std::string>> strings;
        std::map<std::string, std::vector<int>> ints;
        std::map<std::string, std::vector<double>> doubles;
        std::map<std::string, std::vector<void*>> pointers;
        std::string owningParamName;
        std::array<unsigned, 3> geometryReads{};
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
            return properties.intWriteObserver(
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
        if (handle && name) {
            auto& bag = *reinterpret_cast<PropertyBag*>(handle);
            if (std::string_view(name) == kOfxImagePropPixelAspectRatio)
                ++bag.geometryReads[0];
            if (std::string_view(name) == kOfxImageEffectPropRenderScale)
                ++bag.geometryReads[1];
        }
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
        if (handle && name) {
            auto& bag = *reinterpret_cast<PropertyBag*>(handle);
            if (std::string_view(name) == kOfxImagePropRegionOfDefinition)
                ++bag.geometryReads[2];
        }
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

        struct Dimensions {
            int width;
            int height;
        };
        DeviceFrame() : DeviceFrame(Dimensions{kWidth, kHeight}) {}
        explicit DeviceFrame(Dimensions dimensions) : _width(dimensions.width), _height(dimensions.height), _pitch(dimensions.width * kComponents + 3) {
            const std::size_t bytes = byte_count();
            require(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking), "create stream");
            require(cudaMalloc(&_source, bytes), "allocate source");
            require(cudaMalloc(&_destination, bytes), "allocate destination");
            const std::size_t elementCount =
                static_cast<std::size_t>(_pitch) *
                static_cast<std::size_t>(_height);
            _input.assign(elementCount, -7.0f);
            _output.assign(elementCount, kCanary);
            for (int y = 0; y < _height; ++y) {
                for (int x = 0; x < _width; ++x) {
                    const std::size_t offset =
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(_pitch) +
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
            for (int y = 0; y < _height; ++y) {
                for (int x = 0; x < _width; ++x) {
                    const std::size_t offset =
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(_pitch) +
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
                for (int index = _width * kComponents; index < _pitch; ++index) {
                    const std::size_t offset =
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(_pitch) +
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

        int width() const noexcept {
            return _width;
        }
        int height() const noexcept {
            return _height;
        }
        int pitch() const noexcept {
            return _pitch;
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
        std::size_t byte_count() const {
            return static_cast<std::size_t>(_pitch * _height) * sizeof(float);
        }

        static void require(cudaError_t status, const char* operation) {
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(operation) + ": " + cudaGetErrorString(status));
            }
        }

        int _width;
        int _height;
        int _pitch;
        float* _source = nullptr;
        float* _destination = nullptr;
        cudaStream_t _stream = nullptr;
        std::vector<float> _input;
        std::vector<float> _output;
    };

    struct FrameGeometryCase {
        OfxRectI source{0, 0, 7, 5};
        OfxRectI destination{0, 0, 7, 5};
        OfxRectI window{0, 0, 7, 5};
        OfxRectI definition{-10, -20, 18, 20};
        OfxRectD canonical{-50.0, -60.0, 62.0, 100.0};
        OfxPointD scale{0.25, 0.25};
        double par = 1.0;
        int sourceWidth = 7;
        int sourceHeight = 5;
        int definitionWidth = 28;
        int definitionHeight = 40;
        bool cuda = true;
        bool nonfloat = false;
        bool missingSource = false;
        int expectedLeases = 2;
    };

    class NativeHost final {
    public:
        NativeHost() {
            s_active = this;
            try {
                configure_suites();
                configure_host();
                configure_effect();
                configure_descriptor();
                configure_clips();
                configure_params();
            } catch (...) {
                restore_host_suites();
                throw;
            }
        }

        ~NativeHost() {
            restore_host_suites();
        }

        OfxStatus invoke(OfxPlugin& plugin, const char* action, OfxImageEffectHandle effect, OfxPropertySetHandle args) {
            struct ActiveScope {
                NativeHost*& slot;
                NativeHost* previous;
                ~ActiveScope() {
                    slot = previous;
                }
            } scope{s_active, s_active};
            s_active = this;
            return plugin.mainEntry(action, effect, args, nullptr);
        }

        OfxImageEffectHandle effect_handle() noexcept {
            return reinterpret_cast<OfxImageEffectHandle>(&_effectHandle);
        }

        void* instance_data() const {
            const auto found = _effectHandle.properties.pointers.find(kOfxPropInstanceData);
            return found == _effectHandle.properties.pointers.end() || found->second.empty() ? nullptr : found->second.front();
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

        void bind_effect(EffectLease& effect) noexcept {
            _ofxEffect = &effect;
        }

        void unbind_effect() noexcept {
            _ofxEffect = nullptr;
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

        void fail_scan_route_getter(unsigned occurrence) noexcept {
            _scanRouteFailure = occurrence;
        }

        using ParameterHook = std::function<OfxStatus(const std::string&)>;
        void set_getter_hook(ParameterHook hook) {
            _getterHook = std::move(hook);
        }
        void set_setter_hooks(ParameterHook before) {
            _beforeSetterHook = std::move(before);
            _afterSetterHook = {};
        }
        void set_after_setter_hook(ParameterHook after) {
            _afterSetterHook = std::move(after);
        }
        void fail_property_once(const char* param, const char* property, OfxStatus status) {
            _propertyFailureParam = param;
            _propertyFailureProperty = property;
            _propertyFailureStatus = status;
        }
        int int_value(const char* name) const {
            return find_param(name).value.integer;
        }
        std::array<double, 3> double3_value(const char* name) const {
            return find_param(name).value.numbers;
        }

        void call_event(EffectLease& effect, const char* name, ChangeReason reason) {
            PropertyBag arguments;
            arguments.strings[kOfxPropChangeReason] = {change_reason(reason)};
            arguments.strings[kOfxPropType] = {kOfxTypeParameter};
            arguments.strings[kOfxPropName] = {name};
            arguments.doubles[kOfxPropTime] = {_time};
            arguments.doubles[kOfxImageEffectPropRenderScale] = {1.0, 1.0};
            const auto status = invoke_effect(effect.plugin(), kOfxActionInstanceChanged, effect.handle(), reinterpret_cast<OfxPropertySetHandle>(&arguments));
            if (status != kOfxStatReplyDefault)
                throw StatusFailure(status);
        }

        void dispatch_event(EffectLease& effect, const char* name, ChangeReason reason) {
            std::ostringstream line;
            line << "EVENT name=" << name << " reason=" << static_cast<int>(reason)
                 << " time=" << std::setprecision(17) << _time;
            _trace.push_back(line.str());
            call_event(effect, name, reason);
        }

        void render_stream_cases(OfxPlugin& plugin) {
            if (cudaSetDevice(0) != cudaSuccess) {
                throw std::runtime_error("stream test CUDA initialization failed");
            }
            DeviceFrame frame;
            std::uint64_t signature = 0;
            const std::array<std::string, 6> intended{{"supplied-nonnull/immediate", "supplied-nonnull/delayed", "present-null/immediate", "present-null/delayed", "absent/immediate", "absent/delayed"}};
            JuicerTests::Cohort rows("stream_render", std::vector<std::string>(intended.begin(), intended.end()));
            for (int streamCase = 0; streamCase < 3; ++streamCase) {
                const bool present = streamCase != 2;
                cudaStream_t stream = streamCase == 0 ? frame.stream() : nullptr;
                for (bool delayed : {false, true}) {
                    const char* const stream_name = streamCase == 0 ? "supplied-nonnull" : streamCase == 1 ? "present-null"
                                                                                                           : "absent";
                    auto case_event = rows.case_event(std::string(stream_name) + (delayed ? "/delayed" : "/immediate"));
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
                    const auto synchronized = cudaStreamSynchronize(stream);
                    if (synchronized != cudaSuccess || result != kOfxStatOK || releases != 2) {
                        throw std::runtime_error("stream action or lease completion failed: case=" + std::to_string(streamCase) +
                                                 " sync=" + std::to_string(synchronized) + " result=" + std::to_string(result) +
                                                 " releases=" + std::to_string(releases));
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
                    case_event.passed();
                }
            }
        }

        enum class ExceptionClass : std::uint8_t {
            None,
            Allocation,
            Suite,
            Unexpected
        };
        struct RenderResult {
            ExceptionClass exceptionClass = ExceptionClass::None;
            OfxStatus suiteStatus = kOfxStatOK;
            std::size_t releasedBeforeException = 0;
            std::vector<std::string> releaseOrder;
            bool succeeded = false;
            bool rejected = false;
            std::uint64_t outputSignature = 0;
            int releasedImages = 0;
            JuicerTestSupport::EffectObservation::Snapshot state;
        };

        RenderResult render(
            EffectLease& effect,
            DeviceFrame& frame,
            std::string_view phase,
            bool expectRejection = false,
            const FrameGeometryCase* geometry = nullptr) {
            _frameGeometryCase = geometry;
            _canonicalQueries = 0;
            _canonicalTime = -1;
            JuicerFrame::FrameGeometryTest::reset();
            frame.reset();
            configure_image_properties(
                _source.imageProperties,
                frame.source(),
                "adapter-source",
                &frame);
            configure_image_properties(
                _output.imageProperties,
                frame.destination(),
                "adapter-output",
                &frame);
            if (geometry) {
                _source.imageProperties.ints[kOfxImagePropBounds] = {geometry->source.x1, geometry->source.y1, geometry->source.x2, geometry->source.y2};
                _output.imageProperties.ints[kOfxImagePropBounds] = {geometry->destination.x1, geometry->destination.y1, geometry->destination.x2, geometry->destination.y2};
                _source.imageProperties.ints[kOfxImagePropRegionOfDefinition] = {geometry->definition.x1, geometry->definition.y1, geometry->definition.x2, geometry->definition.y2};
                _source.imageProperties.doubles[kOfxImageEffectPropRenderScale] = {geometry->scale.x, geometry->scale.y};
                _source.imageProperties.doubles[kOfxImagePropPixelAspectRatio] = {geometry->par};
                _output.imageProperties.doubles[kOfxImageEffectPropRenderScale] = {0.75, 0.5};
                _output.imageProperties.doubles[kOfxImagePropPixelAspectRatio] = {2.0};
                if (geometry->nonfloat)
                    _source.imageProperties.strings[kOfxImageEffectPropPixelDepth] = {kOfxBitDepthShort};
            }
            begin_image_audit();
            _trace.push_back(
                "RENDER_BEGIN phase=" + std::string(phase) +
                " time=" + number_string(_time));
            RenderResult result{};
            PropertyBag arguments;
            arguments.doubles[kOfxPropTime] = {_time};
            arguments.doubles[kOfxImageEffectPropRenderScale] = {1.0, 1.0};
            const OfxRectI window = geometry ? geometry->window : OfxRectI{0, 0, frame.width(), frame.height()};
            arguments.ints[kOfxImageEffectPropRenderWindow] = {window.x1, window.y1, window.x2, window.y2};
            arguments.ints[kOfxImageEffectPropCudaEnabled] = {geometry ? int(geometry->cuda) : 1};
            arguments.strings[kOfxImageEffectPropFieldToRender] = {kOfxImageFieldNone};
            arguments.pointers[kOfxImageEffectPropCudaStream] = {frame.stream()};
            const auto status = invoke_effect(effect.plugin(), kOfxImageEffectActionRender, effect.handle(), reinterpret_cast<OfxPropertySetHandle>(&arguments));
            result.succeeded = status == kOfxStatOK;
            result.rejected = !result.succeeded;
            if (result.rejected) {
                result.exceptionClass = status == kOfxStatErrMemory ? ExceptionClass::Allocation : ExceptionClass::Suite;
                result.suiteStatus = status;
                result.releasedBeforeException = _imageReleaseOrder.size();
            }
            result.releasedImages = finish_image_audit(phase);
            result.releaseOrder = _imageReleaseOrder;
            const bool completeAllocation = !geometry ||
                                            (geometry->source.x1 == geometry->destination.x1 && geometry->source.y1 == geometry->destination.y1 &&
                                             geometry->source.x2 == geometry->destination.x2 && geometry->source.y2 == geometry->destination.y2 &&
                                             ((geometry->window.x1 == geometry->source.x1 && geometry->window.y1 == geometry->source.y1 &&
                                               geometry->window.x2 == geometry->source.x2 && geometry->window.y2 == geometry->source.y2) ||
                                              (geometry->window.x1 == geometry->window.x2 && geometry->window.y1 == geometry->window.y2)));
            result.outputSignature = frame.collect_signature(result.succeeded && completeAllocation && (!geometry || !geometry->missingSource));
            _frameGeometryCase = nullptr;
            result.state =
                JuicerTestSupport::EffectObservation::snapshot(effect);
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
            if (result.releasedImages != (geometry ? geometry->expectedLeases : 2)) {
                throw std::runtime_error(
                    "adapter render did not release two OFX images in " +
                    std::string(phase));
            }
            return result;
        }

        unsigned canonical_queries() const {
            return _canonicalQueries;
        }
        double canonical_time() const {
            return _canonicalTime;
        }
        const std::array<unsigned, 3>& source_geometry_reads() const {
            return _source.imageProperties.geometryReads;
        }
        const std::array<unsigned, 3>& destination_geometry_reads() const {
            return _output.imageProperties.geometryReads;
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
            _imageReleaseOrder.clear();
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
            _imageReleaseOrder.push_back(found->second.label);
            _trace.push_back("IMAGE_RELEASE clip=" + found->second.label);
        }

        int finish_image_audit(std::string_view phase) {
            if (!_imageAuditActive) {
                throw std::runtime_error("adapter image audit is not active");
            }
            int releases = 0;
            if (_imageLeases.size() != static_cast<std::size_t>(_frameGeometryCase ? _frameGeometryCase->expectedLeases : 2)) {
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
            const JuicerTestSupport::EffectObservation::Snapshot& lhs,
            const JuicerTestSupport::EffectObservation::Snapshot& rhs) {
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
            _ofxHost.fetchSuite = ContainedCallback<fetch_suite>::call;
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
            const char* identifier,
            const DeviceFrame* allocation = nullptr) {
            const int width = allocation ? allocation->width() : DeviceFrame::kWidth;
            const int height = allocation ? allocation->height() : DeviceFrame::kHeight;
            const int pitch = allocation ? allocation->pitch() : DeviceFrame::kPitch;
            properties = PropertyBag{};
            properties.ints[kOfxImagePropRowBytes] = {
                pitch * static_cast<int>(sizeof(float))};
            properties.ints[kOfxImagePropBounds] = {
                0,
                0,
                width,
                height};
            properties.ints[kOfxImagePropRegionOfDefinition] = {
                0,
                0,
                width,
                height};
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
                kDefaultFilmProfileKey);
            add_choice(kParamSpectralMode, 0);
            add_str_choice(
                JuicerParams::kPrintProfileKey,
                kDefaultPrintProfileKey);
            add_choice("ReferenceIlluminant", 0);
            add_choice("EnlargerIlluminant", 3);
            add_choice(
                JuicerParams::kInputColorSpace,
                JuicerTestSupport::PreparationInput{}.values.input_color_space);
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

        static OfxStatus observe_property_int_write(
            void* context,
            const char* owningParam,
            const char* property,
            int value) {
            auto& host = *static_cast<NativeHost*>(context);
            host._trace.push_back(
                "PROPERTY_SET_INT param=" + std::string(owningParam) +
                " property=" + property + " value=" +
                std::to_string(value));
            if (host._propertyFailureStatus != kOfxStatOK &&
                host._propertyFailureParam == owningParam && host._propertyFailureProperty == property) {
                return std::exchange(host._propertyFailureStatus, kOfxStatOK);
            }
            if (!host._propertyReentryArmed ||
                host._propertyReentryOwningParam != owningParam ||
                host._propertyReentryProperty != property) {
                return kOfxStatOK;
            }

            host._propertyReentryArmed = false;
            ++host._propertyReentryCount;
            host._trace.push_back(
                "EVENT_REENTRANT boundary=PROPERTY_SET_INT callback=" +
                host._propertyReentryCallbackParam);
            try {
                if (host._ofxEffect) {
                    host.call_event(*host._ofxEffect, host._propertyReentryCallbackParam.c_str(), ChangeReason::PluginEdit);
                    return kOfxStatOK;
                }
                throw std::runtime_error("property reentry has no bound effect");
            } catch (...) {
                host._propertyReentryFailed = true;
            }
            return kOfxStatOK;
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
            if (host._frameGeometryCase && host._frameGeometryCase->missingSource && reinterpret_cast<NativeClip*>(clip)->name == "Source") {
                *output = nullptr;
                return kOfxStatFailed;
            }
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
            OfxImageClipHandle clip,
            OfxTime time,
            OfxRectD* bounds) {
            if (!bounds) {
                return kOfxStatErrBadHandle;
            }
            auto& host = active();
            ++host._canonicalQueries;
            host._canonicalTime = time;
            host._trace.push_back("CANONICAL_QUERY clip=" + reinterpret_cast<NativeClip*>(clip)->name + " time=" + number_string(time));
            if (host._frameGeometryCase) {
                const auto prepared = JuicerFrame::FrameGeometryTest::prepared_policy_snapshot();
                // The live canonical query follows the accepted Rust policy and scalar pixel phase.
                if (prepared.count != 1 || JuicerFrame::FrameGeometryTest::snapshot().count != 2)
                    return kOfxStatFailed;
                *bounds = host._frameGeometryCase->canonical;
                return kOfxStatOK;
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
            if (param.name == JuicerParams::kParamScanRoute && host._scanRouteFailure != 0 &&
                --host._scanRouteFailure == 0)
                return kOfxStatErrBadHandle;
            const auto hook = host._getterHook;
            return hook ? hook(param.name) : kOfxStatOK;
        }

        static OfxStatus parameter_get_value(OfxParamHandle handle, ...) {
            va_list arguments;
            va_start(arguments, handle);
            try {
                const OfxStatus status = read_param_value(
                    handle,
                    active()._time,
                    false,
                    arguments);
                va_end(arguments);
                return status;
            } catch (const std::bad_alloc&) {
                va_end(arguments);
                return kOfxStatErrMemory;
            } catch (...) {
                va_end(arguments);
                return kOfxStatFailed;
            }
        }

        static OfxStatus parameter_get_value_at_time(
            OfxParamHandle handle,
            OfxTime time,
            ...) {
            va_list arguments;
            va_start(arguments, time);
            try {
                const OfxStatus status = read_param_value(
                    handle,
                    time,
                    true,
                    arguments);
                va_end(arguments);
                return status;
            } catch (const std::bad_alloc&) {
                va_end(arguments);
                return kOfxStatErrMemory;
            } catch (...) {
                va_end(arguments);
                return kOfxStatFailed;
            }
        }

        static OfxStatus write_param_value(
            OfxParamHandle handle,
            va_list arguments) {
            if (!handle) {
                return kOfxStatErrBadHandle;
            }
            NativeHost& host = active();
            NativeParam& param = *reinterpret_cast<NativeParam*>(handle);
            const auto beforeHook = host._beforeSetterHook;
            if (beforeHook) {
                const OfxStatus status = beforeHook(param.name);
                if (status != kOfxStatOK) {
                    return status;
                }
            }
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
            const auto afterHook = host._afterSetterHook;
            return afterHook ? afterHook(param.name) : kOfxStatOK;
        }

        static OfxStatus parameter_set_value(OfxParamHandle handle, ...) {
            va_list arguments;
            va_start(arguments, handle);
            try {
                const OfxStatus status = write_param_value(handle, arguments);
                va_end(arguments);
                return status;
            } catch (const std::bad_alloc&) {
                va_end(arguments);
                return kOfxStatErrMemory;
            } catch (...) {
                va_end(arguments);
                return kOfxStatFailed;
            }
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
            try {
                NativeParam& param = *reinterpret_cast<NativeParam*>(handle);
                const ParamValue previous = param.value;
                const OfxStatus status = write_param_value(handle, arguments);
                if (status == kOfxStatOK) {
                    param.keys[time] = param.value;
                    param.value = previous;
                }
                va_end(arguments);
                return status;
            } catch (const std::bad_alloc&) {
                va_end(arguments);
                return kOfxStatErrMemory;
            } catch (...) {
                va_end(arguments);
                return kOfxStatFailed;
            }
        }

        void dispatch_nested_event(const std::string& name) noexcept {
            try {
                if (_insideNestedEvent)
                    return;
                if (_ofxEffect) {
                    const auto before = JuicerTestSupport::EffectObservation::snapshot(*_ofxEffect);
                    ++_nestedEventCount;
                    _trace.push_back("EVENT_NESTED name=" + name + " reason=" +
                                     std::to_string(static_cast<int>(ChangeReason::PluginEdit)) +
                                     " time=" + number_string(_time));
                    _insideNestedEvent = true;
                    call_event(*_ofxEffect, name.c_str(), ChangeReason::PluginEdit);
                    const auto after = JuicerTestSupport::EffectObservation::snapshot(*_ofxEffect);
                    if (!before.paramEventsSuppressed || !same_effect_state(before, after)) {
                        _nestedContractFailed = true;
                        _nestedContractDiagnostic = "suppressed nested callback changed effect state";
                    }
                    _insideNestedEvent = false;
                    return;
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
            try {
                NativeHost& host = active();
                host._trace.push_back(
                    "MESSAGE type=" + std::string(messageType ? messageType : "") +
                    " id=" + std::string(messageId ? messageId : "") +
                    " text=" + std::string(format ? format : ""));
                return kOfxStatOK;
            } catch (...) {
                return kOfxStatFailed;
            }
        }

        void configure_suites() {
            _propertySuite.propSetPointer = ContainedCallback<property_set_pointer>::call;
            _propertySuite.propSetString = ContainedCallback<property_set_string>::call;
            _propertySuite.propSetDouble = ContainedCallback<property_set_double>::call;
            _propertySuite.propSetInt = ContainedCallback<property_set_int>::call;
            _propertySuite.propSetPointerN = ContainedCallback<property_set_pointer_n>::call;
            _propertySuite.propSetStringN = ContainedCallback<property_set_string_n>::call;
            _propertySuite.propSetDoubleN = ContainedCallback<property_set_double_n>::call;
            _propertySuite.propSetIntN = ContainedCallback<property_set_int_n>::call;
            _propertySuite.propGetPointer = ContainedCallback<property_get_pointer>::call;
            _propertySuite.propGetString = ContainedCallback<property_get_string>::call;
            _propertySuite.propGetDouble = ContainedCallback<property_get_double>::call;
            _propertySuite.propGetInt = ContainedCallback<property_get_int>::call;
            _propertySuite.propGetPointerN = ContainedCallback<property_get_pointer_n>::call;
            _propertySuite.propGetStringN = ContainedCallback<property_get_string_n>::call;
            _propertySuite.propGetDoubleN = ContainedCallback<property_get_double_n>::call;
            _propertySuite.propGetIntN = ContainedCallback<property_get_int_n>::call;
            _propertySuite.propReset = ContainedCallback<property_reset>::call;
            _propertySuite.propGetDimension = ContainedCallback<property_dimension>::call;

            _effectSuite.getPropertySet = ContainedCallback<effect_properties>::call;
            _effectSuite.getParamSet = ContainedCallback<effect_parameters>::call;
            _effectSuite.clipDefine = ContainedCallback<clip_define>::call;
            _effectSuite.clipGetHandle = ContainedCallback<clip_get_handle>::call;
            _effectSuite.clipGetPropertySet = ContainedCallback<clip_properties>::call;
            _effectSuite.clipGetImage = ContainedCallback<clip_get_image>::call;
            _effectSuite.clipReleaseImage = ContainedCallback<clip_release_image>::call;
            _effectSuite.clipGetRegionOfDefinition = ContainedCallback<clip_region>::call;
            _effectSuite.abort = ContainedCallback<effect_abort>::call;

            _parameterSuite.paramSetGetPropertySet = ContainedCallback<parameter_set_properties>::call;
            _parameterSuite.paramDefine = ContainedCallback<parameter_define>::call;
            _parameterSuite.paramGetHandle = ContainedCallback<parameter_get_handle>::call;
            _parameterSuite.paramGetPropertySet = ContainedCallback<parameter_properties>::call;
            _parameterSuite.paramGetValue = parameter_get_value;
            _parameterSuite.paramGetValueAtTime = parameter_get_value_at_time;
            _parameterSuite.paramSetValue = parameter_set_value;
            _parameterSuite.paramSetValueAtTime = parameter_set_value_at_time;
            _memorySuite.memoryAlloc = ContainedCallback<memory_allocate>::call;
            _memorySuite.memoryFree = ContainedCallback<memory_free>::call;
            _multiThreadSuite.multiThread = ContainedCallback<run_threads>::call;
            _multiThreadSuite.multiThreadNumCPUs = ContainedCallback<thread_cpu_count>::call;
            _multiThreadSuite.multiThreadIndex = ContainedCallback<thread_index>::call;
            _multiThreadSuite.multiThreadIsSpawnedThread = ContainedCallback<is_spawned_thread>::call;
            _multiThreadSuite.mutexCreate = ContainedCallback<mutex_create>::call;
            _multiThreadSuite.mutexDestroy = ContainedCallback<mutex_destroy>::call;
            _multiThreadSuite.mutexLock = ContainedCallback<mutex_lock>::call;
            _multiThreadSuite.mutexUnLock = ContainedCallback<mutex_unlock>::call;
            _multiThreadSuite.mutexTryLock = ContainedCallback<mutex_try_lock>::call;
            _messageSuite.message = message;
        }

        void restore_host_suites() noexcept {
            s_active = _previousHost;
        }
        inline static thread_local NativeHost* s_active = nullptr;
        NativeHost* _previousHost = s_active;
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
        EffectLease* _ofxEffect = nullptr;
        double _time = 0.0;
        const FrameGeometryCase* _frameGeometryCase = nullptr;
        unsigned _canonicalQueries = 0;
        double _canonicalTime = -1;
        std::map<OfxPropertySetHandle, ImageLeaseRecord> _imageLeases;
        std::vector<std::string> _imageReleaseOrder;
        unsigned _scanRouteFailure = 0;
        ParameterHook _getterHook;
        ParameterHook _beforeSetterHook;
        ParameterHook _afterSetterHook;
        std::string _propertyFailureParam;
        std::string _propertyFailureProperty;
        OfxStatus _propertyFailureStatus = kOfxStatOK;
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

    OfxStatus invoke_effect(OfxPlugin& plugin, const char* action, OfxImageEffectHandle effect, OfxPropertySetHandle args) {
        if (!effect)
            return kOfxStatErrBadHandle;
        auto* const native = reinterpret_cast<NativeEffectHandle*>(effect);
        if (!native->host)
            return kOfxStatErrBadHandle;
        return native->host->invoke(plugin, action, effect, args);
    }

    void require(bool condition, const std::string& message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void run_submission_retirement_rows(FjRuntimeView runtime) {
        using namespace JuicerFrame::FrameGeometryTest;
        using Observer = JuicerTestSupport::EffectObservation;
        JuicerTests::Cohort rows("submission_retirement", {"never-rendered", "live-peer", "retirement-failure", "partial-construction"});
        struct Lifetime {
            FjOfxLifetime* owner = nullptr;
            explicit Lifetime(OfxImageEffectHandle effect) {
                require(fj_test_ofx_lifetime_acquire(effect, &owner, nullptr).category == FJ_STATUS_SUCCESS && owner,
                        "retain weak instance lifetime observation");
            }
            ~Lifetime() {
                if (owner)
                    (void)fj_test_ofx_lifetime_release(owner, nullptr);
            }
            bool live() const {
                std::uint32_t value = 0;
                require(fj_test_ofx_lifetime_live(owner, &value, nullptr).category == FJ_STATUS_SUCCESS,
                        "read weak instance lifetime observation");
                return value != 0;
            }
        };
        NativeHost host;
        DeviceFrame frame;
        {
            auto row = rows.case_event("never-rendered");
            auto effect = std::make_unique<EffectLease>(*test_plugin, host.effect_handle());
            Lifetime lifetime(effect->handle());
            require(lifetime.live() && !Observer::snapshot(*effect).latchValid, "never-rendered effect owns initialized identity without a submission");
            fj_test_ofx_retirement_reset();
            require(effect->close() == kOfxStatOK, "never-rendered effect destroy status");
            effect.reset();
            FjOfxRetirementObservation retired{};
            require(fj_test_ofx_retirement_observe(&retired) != 0, "retirement observation");
            require(retired.count == 1 && retired.identity_live && retired.token != 0 &&
                        retired.status.category == FJ_STATUS_SUCCESS && !lifetime.live(),
                    "native retirement returns before Rust instance lifetime release");
            row.passed();
        }
        {
            auto row = rows.case_event("live-peer");
            auto first = std::make_unique<EffectLease>(*test_plugin, host.effect_handle());
            Lifetime lifetime(first->handle());
            host.bind_effect(*first);
            require(host.render(*first, frame, "submission-first").succeeded, "first effect render");
            NativeHost peerHost;
            auto peer = std::make_unique<EffectLease>(*test_plugin, peerHost.effect_handle());
            peerHost.bind_effect(*peer);
            require(peerHost.render(*peer, frame, "submission-peer").succeeded, "peer effect render");
            fj_test_ofx_retirement_reset();
            require(first->close() == kOfxStatOK, "first effect destroy status");
            first.reset();
            FjOfxRetirementObservation retired{};
            require(fj_test_ofx_retirement_observe(&retired) != 0, "retirement observation");
            require(retired.count == 1 && retired.identity_live && retired.status.category == FJ_STATUS_SUCCESS && !lifetime.live(),
                    "rendered effect retires membership before Rust release");
            require(peerHost.render(*peer, frame, "submission-peer-after-close").succeeded, "another live effect survives retirement");
            peerHost.unbind_effect();
            row.passed();
        }
        {
            auto row = rows.case_event("retirement-failure");
            auto effect = std::make_unique<EffectLease>(*test_plugin, host.effect_handle());
            Lifetime lifetime(effect->handle());
            require(!Observer::snapshot(*effect).latchValid, "retirement failure control has no native membership");
            fj_test_ofx_retirement_reset();
            {
                JuicerCuda::NativeCall heldGate(runtime.cuda);
                require(effect->close() == kOfxStatErrFatal, "failed retirement destroy status");
                effect.reset();
                FjOfxRetirementObservation retired{};
                require(fj_test_ofx_retirement_observe(&retired) != 0, "retirement observation");
                require(retired.count == 1 && retired.identity_live && retired.token != 0 &&
                            retired.status.category == FJ_STATUS_UNSUPPORTED_INPUT &&
                            retired.status.api == FJ_API_NONE && retired.status.native_code == 0 && !lifetime.live(),
                        "failed native retirement preserves status and releases Rust instance once");
            }
            FjOfxRetirementObservation afterGate{};
            require(fj_test_ofx_retirement_observe(&afterGate) != 0 && afterGate.count == 1, "gate release does not retry failed instance retirement");
            EffectLease next(*test_plugin, host.effect_handle());
            host.bind_effect(next);
            require(host.render(next, frame, "submission-after-retirement-failure").succeeded,
                    "new effect remains usable after retirement failure");
            host.unbind_effect();
            row.passed();
        }
        {
            auto row = rows.case_event("partial-construction");
            fj_test_ofx_retirement_reset();
            bool getterFailed = false;
            host.set_getter_hook([&](const std::string&) {
                getterFailed = true;
                return kOfxStatErrBadHandle;
            });
            bool failed = false;
            try {
                EffectLease partial(*test_plugin, host.effect_handle());
            } catch (const StatusFailure&) {
                failed = true;
            }
            host.set_getter_hook({});
            FjOfxRetirementObservation partialRetirement{};
            require(fj_test_ofx_retirement_observe(&partialRetirement) != 0 && failed && getterFailed && partialRetirement.count == 0,
                    "partial constructor before identity binding creates no membership retirement");
            row.passed();
        }
    }

    void run_instance_contract_rows() {
        using Observer = JuicerTestSupport::EffectObservation;
        const std::vector<std::string> ids{
            "older-capture-success", "older-capture-getter-failure", "capturing-admission", "superseded-before-write-grant", "late-initial-gamma-write-success", "late-initial-gamma-write-partial-failure", "late-gamma-write-retained-hold", "partial-preset-failure", "property-failure-after-edits", "current-getter-failure", "initial-getter-failure", "current-capture-allocation", "current-preparation-allocation", "current-development-allocation", "current-preparation-rejection", "obsolete-preparation-allocation", "unrelated-thread-overlap", "another-effect-chain", "overlapping-writes-old-first", "overlapping-writes-new-first", "overlapping-write-first-failure"};
        JuicerTests::Cohort rows("publication_capture", ids);
        struct CallbackPause {
            std::mutex mutex;
            std::condition_variable condition;
            bool arrived = false, released = false, timedOut = false;
            void pause() {
                std::unique_lock<std::mutex> lock(mutex);
                arrived = true;
                condition.notify_all();
                if (!condition.wait_for(lock, std::chrono::seconds(10), [&]() {
                        return released;
                    })) {
                    timedOut = true;
                }
            }
            bool await() {
                std::unique_lock<std::mutex> lock(mutex);
                return condition.wait_for(lock, std::chrono::seconds(10), [&]() {
                    return arrived;
                });
            }
            void release() {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    released = true;
                }
                condition.notify_all();
            }
        };
        const auto released_failure = [](const NativeHost::RenderResult& result, NativeHost::ExceptionClass kind, OfxStatus suite = kOfxStatOK) {
            require(result.rejected && !result.succeeded && result.exceptionClass == kind &&
                        (kind == NativeHost::ExceptionClass::Allocation ? result.suiteStatus == kOfxStatErrMemory : result.suiteStatus == suite) &&
                        result.releasedImages == 2 && result.releasedBeforeException == 2 &&
                        result.releaseOrder == std::vector<std::string>{"Output", "Source"},
                    "exact adapter exception and image release order");
        };
        for (const auto& id : ids) {
            auto row = rows.case_event(id);
            NativeHost host;
            host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(Spektrafilm::ScanRoute::NegativeDirectScan));
            host.set_int_value(JuicerParams::kDirCouplersActive, 1);
            host.set_int_value(JuicerParams::kDirCouplersGammaUseStock, 1);
            EffectLease effect(*test_plugin, host.effect_handle());
            host.bind_effect(effect);
            DeviceFrame frame;
            const bool initial = id == "superseded-before-write-grant" || id == "late-initial-gamma-write-success" ||
                                 id == "late-initial-gamma-write-partial-failure" || id == "initial-getter-failure";
            JuicerTestSupport::PublicationAdmission held;
            std::uint64_t heldHash = 0;
            if (!initial) {
                require(host.render(effect, frame, "publication-baseline").succeeded, "publication baseline render");
                held = Observer::admit(effect);
                heldHash = JuicerTestSupport::published_view(held).recipe_hash;
            }
            const auto before = Observer::metrics(effect);
            if (id == "older-capture-success" || id == "older-capture-getter-failure") {
                host.set_double_value(kParamExposure, 0.25);
                bool fired = false;
                host.set_getter_hook([&](const std::string& name) {
                    if (name != kParamExposure || fired) {
                        return kOfxStatOK;
                    }
                    fired = true;
                    host.set_getter_hook({});
                    host.set_double_value(kParamExposure, 0.75);
                    host.dispatch_event(effect, kParamExposure, ChangeReason::Time);
                    const auto current = Observer::admit(effect);
                    require(current.state && current.state->input_identity() == Observer::snapshot(effect).pendingFullHash, "newer actual callback installs B");
                    return id == "older-capture-getter-failure" ? kOfxStatErrBadHandle : kOfxStatOK;
                });
                OfxStatus actionStatus = kOfxStatOK;
                try {
                    host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
                } catch (const StatusFailure& error) {
                    actionStatus = error.status();
                }
                require(fired && actionStatus == (id == "older-capture-getter-failure" ? kOfxStatErrBadHandle : kOfxStatOK), "obsolete action retains its own result");
                const auto after = Observer::snapshot(effect);
                require(after.pendingValid && after.pendingDiagnostic.empty() && after.pendingExposureEv == 0.75 &&
                            after.buildCounter == before.build + 1 && host.render(effect, frame, id).succeeded,
                        "old completion cannot replace or invalidate B");
            } else if (id == "capturing-admission") {
                host.set_double_value(kParamExposure, 0.5);
                bool fired = false;
                host.set_getter_hook([&](const std::string& name) {
                    if (name != kParamExposure || fired) {
                        return kOfxStatOK;
                    }
                    fired = true;
                    host.set_getter_hook({});
                    const auto rejected = host.render(effect, frame, id, true);
                    released_failure(rejected, NativeHost::ExceptionClass::Suite, kOfxStatFailed);
                    require(rejected.state.latchSnapshotId == 1 && Observer::metrics(effect).build == before.build,
                            "capturing admission submits no new work or publication");
                    return kOfxStatOK;
                });
                host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
                require(fired && host.render(effect, frame, "capturing-completed").succeeded, "completed capture permits subsequent render");
            } else if (id == "superseded-before-write-grant") {
                bool fired = false;
                unsigned setters = 0;
                host.set_setter_hooks([&](const std::string& name) {
                    if (name == JuicerParams::kDirCouplersGammaSameLayerRgb) {
                        ++setters;
                    }
                    return kOfxStatOK;
                });
                host.set_getter_hook([&](const std::string& name) {
                    if (name != JuicerParams::kFilmProfileKey || fired) {
                        return kOfxStatOK;
                    }
                    fired = true;
                    host.set_getter_hook({});
                    require(Observer::capture_is_current(effect, Observer::metrics(effect).issued) && Observer::metrics(effect).pendingKind == 1 && Observer::metrics(effect).outstandingWrites == 0, "successful current query still does not grant a write");
                    host.set_double3_value(JuicerParams::kDirCouplersGammaSameLayerRgb, {0.71, 0.72, 0.73});
                    host.dispatch_event(effect, JuicerParams::kDirCouplersGammaSameLayerRgb, ChangeReason::UserEdit);
                    return kOfxStatOK;
                });
                require(host.render(effect, frame, id).succeeded && fired && setters == 0 &&
                            host.double3_value(JuicerParams::kDirCouplersGammaSameLayerRgb)[0] == 0.71 &&
                            host.int_value(JuicerParams::kDirCouplersGammaUseStock) == 0,
                        "superseded old grant never enters setter");
            } else if (id == "late-initial-gamma-write-success" || id == "late-initial-gamma-write-partial-failure" || id == "late-gamma-write-retained-hold") {
                const bool partial = id == "late-initial-gamma-write-partial-failure";
                bool fired = false;
                unsigned gammaSetters = 0;
                host.set_setter_hooks([&](const std::string& name) {
                    if (name != JuicerParams::kDirCouplersGammaSameLayerRgb) {
                        return kOfxStatOK;
                    }
                    ++gammaSetters;
                    if (fired) {
                        return kOfxStatOK;
                    }
                    fired = true;
                    const auto granted = Observer::metrics(effect);
                    require(granted.outstandingWrites == 1, "write registered before host commitment");
                    host.set_double3_value(JuicerParams::kDirCouplersGammaSameLayerRgb, {0.91, 0.92, 0.93});
                    const auto traceBegin = host.trace_size();
                    host.dispatch_event(effect, JuicerParams::kDirCouplersGammaSameLayerRgb, ChangeReason::UserEdit);
                    const auto deferred = Observer::metrics(effect);
                    require(deferred.pendingKind == 2 && deferred.outstandingWrites == 1 && deferred.build == before.build,
                            "newer action defers complete read while late write may commit");
                    for (std::size_t i = traceBegin; i < host.trace_size(); ++i) {
                        require(host.trace()[i].find("GET_CURRENT name=Exposure ") == std::string::npos, "Deferred issues no complete snapshot getter");
                    }
                    const auto rejected = Observer::admit(effect);
                    require(rejected.kind == FJ_ADMISSION_CAPTURING && !rejected.state, "late pre-commit write blocks admission");
                    return kOfxStatOK;
                });
                host.set_after_setter_hook([&](const std::string& name) {
                    return partial && name == JuicerParams::kDirCouplersGammaSameLayerRgb ? kOfxStatErrBadHandle : kOfxStatOK;
                });
                if (initial) {
                    const auto rendered = host.render(effect, frame, id, partial);
                    if (partial) {
                        released_failure(rendered, NativeHost::ExceptionClass::Suite, kOfxStatErrBadHandle);
                    } else {
                        require(rendered.succeeded, "late initial successful writer completes bounded refresh");
                    }
                } else {
                    host.dispatch_event(effect, JuicerParams::kDirCouplersGammaUseStock, ChangeReason::UserEdit);
                }
                host.set_setter_hooks({});
                require(fired && gammaSetters == 1 && host.int_value(JuicerParams::kDirCouplersGammaUseStock) == 0 &&
                            host.double3_value(JuicerParams::kDirCouplersGammaSameLayerRgb)[0] != 0.91,
                        "late write changes actual controls; old later setters stop");
                const auto after = Observer::metrics(effect);
                require(after.outstandingWrites == 0 && !Observer::snapshot(effect).paramEventsSuppressed, "host-write and suppression obligations balanced");
                if (partial) {
                    require(after.pendingKind == 5 && after.build == before.build && after.issued == 2, "late partial failure blocks without automatic refresh");
                    released_failure(host.render(effect, frame, "late-failure-blocked", true), NativeHost::ExceptionClass::Suite, kOfxStatErrFatal);
                    host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
                    require(host.render(effect, frame, "late-failure-recovered").succeeded, "fresh complete callback recovers after drainage");
                } else {
                    const auto expected = Observer::read_host_snapshot(effect);
                    const auto actual = Observer::captured(effect);
                    require(after.pendingKind == 4 && after.inputIdentity == JuicerTestSupport::hash_input_identity(expected) && JuicerTestSupport::hash_input_identity(actual) == JuicerTestSupport::hash_input_identity(expected) &&
                                actual.values.dir_couplers.gamma_same_layer_rgb[0] == static_cast<float>(host.double3_value(JuicerParams::kDirCouplersGammaSameLayerRgb)[0]) &&
                                !actual.values.dir_couplers.gamma_use_stock && after.issued == before.issued + 3,
                            "one getter-only refresh selects complete actual final host controls");
                }
            } else if (id == "partial-preset-failure" || id == "property-failure-after-edits") {
                host.set_double_value(JuicerParams::kGrainAmplitude, 3.0);
                host.set_int_value(JuicerParams::kGrainPreset, 1);
                if (id == "partial-preset-failure") {
                    host.set_setter_hooks([&](const std::string& name) {
                        return name == JuicerParams::kGrainBlur ? kOfxStatErrBadHandle : kOfxStatOK;
                    });
                } else {
                    host.fail_property_once(JuicerParams::kGrainChroma, kOfxParamPropEnabled, kOfxStatErrBadHandle);
                }
                OfxStatus actionStatus = kOfxStatOK;
                try {
                    host.dispatch_event(effect, JuicerParams::kGrainPreset, ChangeReason::UserEdit);
                } catch (const StatusFailure& error) {
                    actionStatus = error.status();
                }
                host.set_setter_hooks({});
                require(actionStatus == kOfxStatErrBadHandle && host.double_value(JuicerParams::kGrainAmplitude) != 3.0 &&
                            Observer::metrics(effect).pendingKind == 5 && Observer::metrics(effect).build == before.build &&
                            !Observer::snapshot(effect).paramEventsSuppressed,
                        "partial edit/property failure records failed capture without rollback");
                released_failure(host.render(effect, frame, id, true), NativeHost::ExceptionClass::Suite, kOfxStatErrFatal);
                host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
                require(host.render(effect, frame, "partial-edit-recovered").succeeded, "genuine change recovers without preset replay");
            } else if (id == "current-getter-failure" || id == "initial-getter-failure" || id == "current-capture-allocation") {
                bool fired = false;
                host.set_getter_hook([&](const std::string& name) -> OfxStatus {
                    if (name != kParamExposure || fired) {
                        return kOfxStatOK;
                    }
                    fired = true;
                    if (id == "current-capture-allocation") {
                        return kOfxStatErrMemory;
                    }
                    return kOfxStatErrBadHandle;
                });
                if (initial) {
                    released_failure(host.render(effect, frame, id, true), NativeHost::ExceptionClass::Suite, kOfxStatErrBadHandle);
                } else {
                    OfxStatus status = kOfxStatOK;
                    try {
                        host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
                    } catch (const StatusFailure& error) {
                        status = error.status();
                    }
                    require(status == (id == "current-capture-allocation" ? kOfxStatErrMemory : kOfxStatErrBadHandle), "current action keeps actual callback status");
                }
                host.set_getter_hook({});
                require(fired && Observer::metrics(effect).pendingKind == 5 && !Observer::snapshot(effect).paramEventsSuppressed, "current getter completes failed capture");
                const bool allocation = id == "current-capture-allocation";
                released_failure(host.render(effect, frame, "failed-capture-admission", true), allocation ? NativeHost::ExceptionClass::Allocation : NativeHost::ExceptionClass::Suite, kOfxStatErrFatal);
                host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
                require(host.render(effect, frame, "getter-failure-recovered").succeeded, "fresh action recovers getter failure");
            } else if (id == "current-preparation-allocation" || id == "current-development-allocation" || id == "current-preparation-rejection" || id == "obsolete-preparation-allocation") {
                host.set_double_value(kParamExposure, 0.25);
                host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
                if (id == "current-development-allocation") {
                    require(fj_test_development_construction_fault(1).category == FJ_STATUS_SUCCESS, "arm actual development allocation");
                } else if (id == "current-preparation-rejection") {
                    require(fj_test_development_construction_fault(2).category == FJ_STATUS_SUCCESS, "arm actual development rejection");
                } else {
                    fj_test_prepared_contents_reservation_failure();
                }
                if (id == "current-preparation-allocation") {
                    FrameGeometryCase missing;
                    missing.missingSource = true;
                    missing.expectedLeases = 1;
                    require(host.render(effect, frame, "allocation-missing-image", false, &missing).succeeded && !fj_test_prepared_contents_reservation_failure_consumed(), "missing image precedes allocation admission");
                    FrameGeometryCase empty;
                    empty.window.x2 = empty.window.x1;
                    require(host.render(effect, frame, "allocation-empty-image", false, &empty).succeeded && !fj_test_prepared_contents_reservation_failure_consumed(), "empty window precedes allocation admission");
                }
                if (id == "obsolete-preparation-allocation") {
                    struct NewerCapture {
                        NativeHost* host;
                        EffectLease* effect;
                        bool fired = false;
                        std::exception_ptr failure;
                    } newer{&host, &effect, false, {}};
                    fj_test_instance_prepare_hooks(nullptr, [](void* user) noexcept {
                        auto& newer = *static_cast<NewerCapture*>(user);
                        if (std::exchange(newer.fired, true)) {
                            return;
                        }
                        fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
                        try {
                            newer.host->set_double_value(kParamExposure, 0.75);
                            newer.host->dispatch_event(*newer.effect, kParamExposure, ChangeReason::Time);
                        } catch (...) {
                            newer.failure = std::current_exception();
                        }
                    },
                                                   &newer);
                    const auto rendered = host.render(effect, frame, id);
                    fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
                    if (newer.failure) {
                        std::rethrow_exception(newer.failure);
                    }
                    require(newer.fired && rendered.succeeded && rendered.exceptionClass == NativeHost::ExceptionClass::None &&
                                fj_test_prepared_contents_reservation_failure_consumed() && rendered.state.pendingExposureEv == 0.75 &&
                                rendered.state.buildCounter == before.build + 1,
                            "obsolete allocation retries newer input before external error mapping");
                } else {
                    const bool allocation = id != "current-preparation-rejection";
                    const auto failed = host.render(effect, frame, id, true);
                    released_failure(failed, allocation ? NativeHost::ExceptionClass::Allocation : NativeHost::ExceptionClass::Suite, kOfxStatErrFatal);
                    require(Observer::metrics(effect).build == before.build && Observer::metrics(effect).backingIdentity == before.backingIdentity &&
                                failed.state.latchSnapshotId == 1,
                            "typed preparation failure preserves committed publication and submission latch");
                    if (id == "current-preparation-allocation") {
                        require(fj_test_prepared_contents_reservation_failure_consumed(), "actual substantial reservation fault consumed");
                    }
                    require(host.render(effect, frame, "preparation-failure-recovered").succeeded, "one-shot ordinary preparation failure recovers current input");
                }
            } else if (id == "unrelated-thread-overlap") {
                CallbackPause pause;
                std::exception_ptr workerFailure;
                host.set_double_value(kParamExposure, 0.25);
                host.set_getter_hook([&](const std::string& name) {
                    if (name != kParamExposure) {
                        return kOfxStatOK;
                    }
                    host.set_getter_hook({});
                    pause.pause();
                    return kOfxStatOK;
                });
                std::jthread worker([&]() {
                    try {
                        host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
                    } catch (...) {
                        workerFailure = std::current_exception();
                    }
                });
                const bool paused = pause.await();
                std::exception_ptr mainFailure;
                try {
                    if (paused) {
                        host.set_double_value(kParamExposure, 0.75);
                        host.dispatch_event(effect, kParamExposure, ChangeReason::Time);
                    }
                } catch (...) {
                    mainFailure = std::current_exception();
                }
                pause.release();
                worker.join();
                if (mainFailure) {
                    std::rethrow_exception(mainFailure);
                }
                if (workerFailure) {
                    std::rethrow_exception(workerFailure);
                }
                require(paused && !pause.timedOut && Observer::snapshot(effect).pendingExposureEv == 0.75 && host.render(effect, frame, id).succeeded,
                        "unrelated callback on another thread supersedes old read without suppression");
            } else if (id == "another-effect-chain") {
                bool fired = false;
                bool otherStarted = false;
                host.set_setter_hooks([&](const std::string& name) {
                    if (name != JuicerParams::kDirCouplersGammaSameLayerRgb || fired) {
                        return kOfxStatOK;
                    }
                    fired = true;
                    NativeHost otherHost;
                    EffectLease other(*test_plugin, otherHost.effect_handle());
                    otherHost.bind_effect(other);
                    otherHost.dispatch_event(other, kParamExposure, ChangeReason::PluginEdit);
                    otherStarted = Observer::metrics(other).issued == 1 && Observer::snapshot(other).pendingValid;
                    return kOfxStatOK;
                });
                host.dispatch_event(effect, JuicerParams::kDirCouplersGammaUseStock, ChangeReason::UserEdit);
                require(fired && otherStarted && !Observer::snapshot(effect).paramEventsSuppressed, "another concrete effect is not suppressed by outer host-write chain");
            } else {
                CallbackPause olderPause, newerPause;
                std::exception_ptr olderFailure, newerFailure;
                host.set_double3_value(JuicerParams::kDirCouplersGammaSameLayerRgb, {0.71, 0.72, 0.73});
                const bool failOlder = id == "overlapping-write-first-failure";
                host.set_setter_hooks([&](const std::string& name) {
                    if (name == JuicerParams::kDirCouplersGammaSameLayerRgb) {
                        olderPause.pause();
                    }
                    if (name == JuicerParams::kDirCouplersGammaUseStock) {
                        newerPause.pause();
                    }
                    return kOfxStatOK;
                });
                host.set_after_setter_hook([&](const std::string& name) {
                    return failOlder && name == JuicerParams::kDirCouplersGammaSameLayerRgb ? kOfxStatErrBadHandle : kOfxStatOK;
                });
                std::jthread older([&]() {
                    try {
                        host.dispatch_event(effect, JuicerParams::kDirCouplersGammaUseStock, ChangeReason::UserEdit);
                    } catch (...) {
                        olderFailure = std::current_exception();
                    }
                });
                const bool olderArrived = olderPause.await();
                std::jthread newer([&]() {
                    try {
                        host.dispatch_event(effect, JuicerParams::kDirCouplersGammaSameLayerRgb, ChangeReason::UserEdit);
                    } catch (...) {
                        newerFailure = std::current_exception();
                    }
                });
                const bool newerArrived = newerPause.await();
                const bool registered = Observer::metrics(effect).outstandingWrites == 2;
                bool intermediateBlocked = false;
                if (id == "overlapping-writes-new-first") {
                    newerPause.release();
                    newer.join();
                    intermediateBlocked = Observer::metrics(effect).outstandingWrites == 1 &&
                                          Observer::metrics(effect).pendingKind == 2 && Observer::admit(effect).kind == FJ_ADMISSION_CAPTURING;
                    olderPause.release();
                    older.join();
                } else {
                    olderPause.release();
                    older.join();
                    const auto interim = Observer::admit(effect);
                    intermediateBlocked = Observer::metrics(effect).outstandingWrites == 1 && !interim.state &&
                                          interim.kind == (failOlder ? FJ_ADMISSION_CAPTURE_FAILED : FJ_ADMISSION_CAPTURING);
                    newerPause.release();
                    newer.join();
                }
                host.set_setter_hooks({});
                if (newerFailure) {
                    std::rethrow_exception(newerFailure);
                }
                if (!failOlder && olderFailure) {
                    std::rethrow_exception(olderFailure);
                }
                require(olderArrived && newerArrived && !olderPause.timedOut && !newerPause.timedOut && registered && intermediateBlocked &&
                            Observer::metrics(effect).outstandingWrites == 0 && !Observer::snapshot(effect).paramEventsSuppressed,
                        "both registered physical writes complete once in either order");
                if (failOlder) {
                    bool expectedError = false;
                    try {
                        if (olderFailure) {
                            std::rethrow_exception(olderFailure);
                        }
                    } catch (const StatusFailure& error) {
                        expectedError = error.status() == kOfxStatErrBadHandle;
                    }
                    require(expectedError && Observer::metrics(effect).pendingKind == 5 && host.int_value(JuicerParams::kDirCouplersGammaUseStock) == 0 &&
                                host.double3_value(JuicerParams::kDirCouplersGammaSameLayerRgb)[0] != 0.71,
                            "later successful write does not erase partial-write failure or roll back host controls");
                    released_failure(host.render(effect, frame, id, true), NativeHost::ExceptionClass::Suite, kOfxStatErrFatal);
                    host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
                } else {
                    const auto expected = Observer::read_host_snapshot(effect);
                    require(Observer::metrics(effect).inputIdentity == JuicerTestSupport::hash_input_identity(expected) && !expected.values.dir_couplers.gamma_use_stock &&
                                host.double3_value(JuicerParams::kDirCouplersGammaSameLayerRgb)[0] != 0.71 &&
                                Observer::metrics(effect).issued == before.issued + (id == "overlapping-writes-new-first" ? 3u : 2u),
                            "successful drainage captures actual final controls with one normal or Refresh completion");
                }
                require(host.render(effect, frame, "overlapping-write-complete").succeeded, "drained write sequence permits rendering");
            }
            host.set_getter_hook({});
            host.set_setter_hooks({});
            if (held.state) {
                require(JuicerTestSupport::published_view(held).recipe_hash == heldHash && held.state->build() == before.build, "already-admitted backing survives callback overlap/failure");
            }
            row.passed();
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

    bool same_float3(const float (&actual)[3], const std::array<float, 3>& expected) {
        return std::equal(std::begin(actual), std::end(actual), expected.begin());
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
    void read_prepared_view(const JuicerTestSupport::PreparedStateHold& owner, FjPreparedStateView& view) {
        std::array<char, 2048> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const FjStatus status = owner.view(view, error);
        if (status.category != FJ_STATUS_SUCCESS)
            throw std::runtime_error(std::string("prepared-state view failed: ") + text.data());
    }
    std::uint64_t prepared_recipe_hash(const JuicerTestSupport::PreparedStateHold& owner) {
        FjPreparedStateView view{};
        read_prepared_view(owner, view);
        return view.recipe_hash;
    }
    Json physical_words(const FjDust& value) {
        Json out = Json::array();
        const float fields[]{value.cell_width_mm, value.cell_height_mm, value.slot_probability, value.softness_min_mm, value.softness_max_mm, value.softness_size_cap_fraction, value.support_x_mm, value.support_y_mm, value.fiber_fraction, value.fiber_drift_fraction, value.fiber_first_knot_min, value.fiber_first_knot_max, value.fiber_second_knot_min, value.fiber_second_knot_max, value.fiber_interior_width_min_fraction, value.fiber_interior_width_max_fraction, value.diameter_min_mm, value.diameter_bulk_max_mm, value.diameter_max_mm, value.diameter_tail_fraction, value.fiber_length_min_mm, value.fiber_length_max_mm, value.fiber_width_min_mm, value.fiber_width_max_mm, value.opacity_faint_cumulative, value.opacity_intermediate_cumulative, value.compact_opacity_min, value.compact_opacity_faint_end, value.compact_opacity_intermediate_end, value.compact_opacity_max, value.fiber_opacity_min, value.fiber_opacity_faint_end, value.fiber_opacity_intermediate_end, value.fiber_opacity_max, value.compact_dominant_aspect_min, value.compact_dominant_aspect_max, value.compact_subsidiary_scale_min, value.compact_subsidiary_scale_max, value.compact_subsidiary_aspect_min, value.compact_subsidiary_aspect_max, value.compact_subsidiary_offset_max, value.compact_subsidiary_angle_max_radians};
        for (float field : fields)
            out.push_back(std::bit_cast<std::uint32_t>(field));
        return out;
    }
    Json physical_words(const FjScratch& value) {
        Json out = Json::array();
        const float fields[]{value.cell_width_mm, value.cell_height_mm, value.slot_probability, value.softness_min_mm, value.softness_max_mm, value.softness_size_cap_fraction, value.support_x_mm, value.support_y_mm, value.length_min_mm, value.length_bulk_max_mm, value.length_max_mm, value.length_tail_fraction, value.width_min_mm, value.width_bulk_max_mm, value.width_max_mm, value.width_tail_fraction, value.drift_fraction, value.first_knot_min, value.first_knot_max, value.second_knot_min, value.second_knot_max, value.interior_width_min_fraction, value.interior_width_max_fraction, value.interior_depth_min_fraction, value.interior_depth_max_fraction, value.endpoint_abrupt_probability, value.interruption_probability, value.gap_center_min, value.gap_center_max, value.gap_span_min, value.gap_span_max, value.scuff_probability, value.scuff_length_max_mm, value.scuff_angle_max_radians, value.strength_min, value.strength_max};
        for (float field : fields)
            out.push_back(std::bit_cast<std::uint32_t>(field));
        return out;
    }
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
    Json optics_project(const FjDiffusionParams& value) {
        const std::array<double, 9> fields{value.strength, value.spatial_scale, value.halo_warmth, value.core_intensity, value.core_size, value.halo_intensity, value.halo_size, value.bloom_intensity, value.bloom_size};
        return {{"active", value.active != 0}, {"family", value.family}, {"f64", optics_words(fields)}};
    }
    Json optics_project(const FjSpatialOpticsRecipe& a) {
        const auto project_diffusion = [](const FjDiffusionRecipe& v) {
            const double lanes[]{v.scatter_fraction, v.group_weights[0], v.group_weights[1], v.group_weights[2], v.group_lambda_um[0], v.group_lambda_um[1], v.group_lambda_um[2], v.warmth, v.spatial_scale};
            return Json{{"domain", v.domain}, {"active", v.active != 0}, {"family", v.family}, {"f64", optics_words(lanes)}, {"resolved_hash", v.resolved_hash}, {"hash", v.hash}};
        };
        const auto& h = a.scatter_halation;
        return {{"camera", project_diffusion(a.camera_diffusion)}, {"lens_f32", std::bit_cast<std::uint32_t>(a.camera_lens_blur.sigma_um)}, {"lens_hash", a.camera_lens_blur.hash}, {"halation", {{"scatter_active", h.scatter_active != 0}, {"back_reflection_active", h.back_reflection_active != 0}, {"f32", optics_words(std::array<float, 9>{h.scatter_amount, h.scatter_spatial_scale, h.halation_spatial_scale, h.first_sigma_um[0], h.first_sigma_um[1], h.first_sigma_um[2], h.total_strength[0], h.total_strength[1], h.total_strength[2]})}, {"hash", h.hash}}}, {"enlarger", project_diffusion(a.enlarger_diffusion)}, {"hash", a.hash}};
    }
    Json diffusion_policy_stage(
        const FjDiffusion& value,
        Spektrafilm::DiffusionLinearStage fixtureStage) {
        const auto weights = std::array{value.group_weights[0], value.group_weights[1], value.group_weights[2]};
        const auto centers = std::array{value.group_lambda_um[0], value.group_lambda_um[1], value.group_lambda_um[2]};
        const Json sample{{"family", value.family}, {"weights", optics_words(weights)}, {"centers", optics_words(centers)}, {"warmth", std::bit_cast<std::uint64_t>(value.warmth)}, {"scale", std::bit_cast<std::uint64_t>(value.spatial_scale)}, {"pixel", std::bit_cast<std::uint64_t>(value.pixel_size_um)}, {"radius", value.radius_px}, {"hash", value.sample_hash}};
        return {{"stage", static_cast<unsigned>(fixtureStage)}, {"scatter", std::bit_cast<std::uint64_t>(value.scatter_fraction)}, {"sample", sample}, {"hash", value.hash}};
    }
    Json diffusion_policy_project(const FjPreparedFramePolicy& policy, unsigned route) {
        const auto& value = policy.optics;
        const bool camera = (value.flags & FJ_OPTICS_CAMERA_DIFFUSION) != 0;
        const bool enlarger = (value.flags & FJ_OPTICS_ENLARGER_DIFFUSION) != 0;
        if (!camera && !enlarger)
            return nullptr;
        const auto& extent = value.full_frame;
        // Fixture stage tags use the native descriptor enum (Camera=0, Enlarger=1), not C ABI flags (1, 2).
        return {{"route", route}, {"extent", {extent.x, extent.y, extent.width, extent.height}}, {"camera", camera ? diffusion_policy_stage(value.camera, Spektrafilm::DiffusionLinearStage::CameraFilmLinear) : Json()}, {"enlarger", enlarger ? diffusion_policy_stage(value.enlarger, Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear) : Json()}, {"hash", value.diffusion_hash}};
    }
    Json halation_policy_gaussian(const FjHalationGaussian& value) {
        return {{"kind", value.kind}, {"radius", value.radius}, {"weights", optics_words(value.fir_weights)}, {"yvv", optics_words(std::array{value.feedforward, value.feedback[0], value.feedback[1], value.feedback[2]})}};
    }
    Json halation_policy_project(const FjScatterHalation& value) {
        Json channels = Json::array();
        for (const auto& channel : value.channels) {
            Json tail = Json::array();
            Json bounce = Json::array();
            for (const auto& gaussian : channel.tail)
                tail.push_back(halation_policy_gaussian(gaussian));
            for (const auto& gaussian : channel.bounce)
                bounce.push_back(halation_policy_gaussian(gaussian));
            channels.push_back({{"core", halation_policy_gaussian(channel.core)}, {"tail", tail}, {"bounce", bounce}, {"strength", std::bit_cast<std::uint32_t>(channel.total_strength)}});
        }
        return {{"recipe_hash", value.recipe_hash}, {"scatter_amount", std::bit_cast<std::uint32_t>(value.scatter_amount)}, {"channels", channels}};
    }
    Json optics_inputs(const FjPreparationParams& input, bool print) {
        const auto& h = input.scatter_halation;
        return {{"camera", optics_project(input.camera_diffusion)},
                {"lens_f32", std::bit_cast<std::uint32_t>(0.0f)},
                {"halation_active", h.active != 0},
                {"halation_f32", optics_words(std::array<float, 4>{h.scatter_amount, h.scatter_spatial_scale, h.halation_amount, h.halation_spatial_scale})},
                {"first_sigma_f32", optics_words(std::array<float, 3>{65.0f, 65.0f, 65.0f})},
                {"primary_f32", optics_words(std::array<float, 3>{0.08f, 0.02f, 0.0f})},
                {"enlarger", optics_project(input.enlarger_diffusion)},
                {"print", print}};
    }

    void check_optics_capture(const std::filesystem::path& grainManifest, const char* preset) {
        const auto path = grainManifest.parent_path().parent_path() / "optics-fixtures/manifest.json";
        std::ifstream manifestStream(path);
        require(bool(manifestStream), "optics manifest open");
        const auto manifest = Json::parse(manifestStream);
        const auto capturePath = path.parent_path() / manifest.at("presets").at(preset).at("ofx").get<std::string>();
        std::ifstream stream(capturePath);
        require(bool(stream), "optics capture payload open");
        const auto frozen = PrintIlluminantFixtures::apply(capturePath, Json::parse(stream)).at("ofx");
        require(frozen.size() == 96, "complete frozen optics capture membership");
        std::vector<std::string> intended;
        intended.reserve(96);
        for (unsigned route = 0; route < 4; ++route)
            for (unsigned input = 0; input < 24; ++input)
                intended.push_back("route-" + std::to_string(route) + "/input-" + std::to_string(input));
        JuicerTests::Cohort rows("optics_capture", intended);
        unsigned visits = 0;
        for (unsigned route = 0; route < 4; ++route) {
            NativeHost host;
            EffectLease effect(*test_plugin, host.effect_handle());
            host.bind_effect(effect);
            host.set_time(0);
            host.set_str_choice_value(JuicerParams::kFilmProfileKey, route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f");
            host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(route % 2 ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan));
            const std::array<const char*, 9> camera{JuicerParams::kCameraDiffusionStrength, JuicerParams::kCameraDiffusionSpatialScale, JuicerParams::kCameraDiffusionHaloWarmth, JuicerParams::kCameraDiffusionCoreIntensity, JuicerParams::kCameraDiffusionCoreSize, JuicerParams::kCameraDiffusionHaloIntensity, JuicerParams::kCameraDiffusionHaloSize, JuicerParams::kCameraDiffusionBloomIntensity, JuicerParams::kCameraDiffusionBloomSize};
            const std::array<const char*, 9> enlarger{JuicerParams::kPrintDiffusionStrength, JuicerParams::kPrintDiffusionSpatialScale, JuicerParams::kPrintDiffusionHaloWarmth, JuicerParams::kPrintDiffusionCoreIntensity, JuicerParams::kPrintDiffusionCoreSize, JuicerParams::kPrintDiffusionHaloIntensity, JuicerParams::kPrintDiffusionHaloSize, JuicerParams::kPrintDiffusionBloomIntensity, JuicerParams::kPrintDiffusionBloomSize};
            const std::array<const char*, 4> halation{JuicerParams::kHalationScatterAmount, JuicerParams::kHalationScatterSpatialScale, JuicerParams::kHalationAmount, JuicerParams::kHalationSpatialScale};
            for (int n = 0; n < 24; ++n) {
                auto case_event = rows.case_event("route-" + std::to_string(route) + "/input-" + std::to_string(n));
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
                host.dispatch_event(effect, JuicerParams::kGrainActive, ChangeReason::Time);
                auto observed = JuicerTestSupport::EffectObservation::snapshot(effect);
                require(host.at_time_get_count() == timed, "current getters");
                Json getters = Json::array();
                for (std::size_t i = start; i < host.trace_size(); ++i)
                    if (host.trace()[i].starts_with("GET_CURRENT"))
                        getters.push_back(host.trace()[i]);
                auto admission = JuicerTestSupport::EffectObservation::admit(effect);
                FjPreparedStateView preparedView{};
                const bool admitted = bool(admission.state);
                if (admission.state)
                    read_prepared_view(*admission.state, preparedView);
                const auto& expected = frozen.at(route * 24 + n);
                const Json actual{{"route", route}, {"index", n}, {"pending_valid", observed.pendingValid}, {"pending_invalid", observed.pendingInvalid}, {"pending_diagnostic", observed.pendingDiagnostic}, {"snapshot_hash", observed.pendingFullHash}, {"admitted", admitted}, {"input", optics_inputs(observed.input, route % 2 != 0)}, {"recipe", admitted ? optics_project(preparedView.optics) : Json()}, {"final_hash", admitted ? preparedView.recipe_hash : 0}, {"getters", getters}};
                require(actual == expected, "all frozen OFX optics capture/getter/publication fields route=" + std::to_string(route) + " index=" + std::to_string(n));
                case_event.passed();
                ++visits;
            }
            host.unbind_effect();
        }
        require(visits == intended.size(), "optics capture visit count matches declared route/input extent");
        std::cout << "PASS adapter/96-frozen-optics-capture-publications-four-routes\n";
    }
    void check_physical_capture(NativeHost& host, const std::filesystem::path& grainManifest, const char* preset) {
        const auto manifestPath = grainManifest.parent_path().parent_path() / "effects-fixtures/manifest.json";
        std::ifstream manifestStream(manifestPath);
        require(bool(manifestStream), "effects capture manifest open");
        const auto manifest = nlohmann::json::parse(manifestStream);
        const auto capturePath = manifestPath.parent_path() / manifest.at("presets").at(preset).at("ofx").get<std::string>();
        std::ifstream stream(capturePath);
        require(bool(stream), "effects capture fixture open");
        const auto captured = PrintIlluminantFixtures::apply(capturePath, nlohmann::json::parse(stream));
        require(captured.at("ofx").size() == 608, "complete frozen effects capture membership");
        std::vector<std::string> intended;
        intended.reserve(20);
        for (const auto& row : captured.at("ofx")) {
            const unsigned policy = row.at("policy");
            const unsigned index = row.at("index");
            if ((index == 5 && (policy == 0 || policy == 1 || policy == 4 || policy == 7)) || (index == 15 && policy == 0))
                intended.push_back("route-" + std::to_string(row.at("route").get<unsigned>()) + "/policy-" + std::to_string(policy) + "/index-" + std::to_string(index));
        }
        JuicerTests::Cohort rows("physical_capture", intended);
        EffectLease effect(*test_plugin, host.effect_handle());
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
            auto case_event = rows.case_event("route-" + std::to_string(route) + "/policy-" + std::to_string(policy) + "/index-" + std::to_string(index));
            const bool print = route % 2;
            host.set_str_choice_value(JuicerParams::kFilmProfileKey, route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f");
            host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan));
            host.set_int_value(JuicerParams::kGrainActive, policy != 0);
            host.set_int_value(JuicerParams::kGrainDebugView, policy > 1 ? static_cast<int>(policy - 1) : 0);
            for (std::size_t i = 0; i < getters.size(); ++i)
                host.set_double_value(getters[i], std::bit_cast<double>(row.at("authored_getter_order_f64")[i].get<std::uint64_t>()));
            const auto start = host.trace_size();
            const auto timed = host.at_time_get_count();
            host.dispatch_event(effect, JuicerParams::kGrainActive, ChangeReason::Time);
            const auto observed = JuicerTestSupport::EffectObservation::snapshot(effect);
            require(observed.pendingValid && host.at_time_get_count() == timed, "physical capture uses current getters even during disabled/debug grain");
            bool ordered = false;
            for (std::size_t i = start; i < host.trace_size(); ++i) {
                std::size_t j = 0;
                while (j < getters.size() && i + j < host.trace_size() && host.trace()[i + j].starts_with("GET_CURRENT name=" + std::string(getters[j]) + " "))
                    ++j;
                ordered = ordered || j == getters.size();
            }
            require(ordered, "physical getter order film dust/gate dust/film scratch/gate scratch/weave");
            const auto& controls = observed.input;
            const auto words = nlohmann::json::array({std::bit_cast<std::uint32_t>(controls.film_dust_amount), std::bit_cast<std::uint32_t>(controls.film_scratch_amount), std::bit_cast<std::uint32_t>(controls.gate_dust_amount), std::bit_cast<std::uint32_t>(controls.gate_scratch_amount)});
            require(words == row.at("controls").at("amounts") && std::bit_cast<std::uint64_t>(controls.gate_weave_amount) == row.at("controls").at("weave") && observed.pendingFullHash == row.at("snapshot_hash"), "frozen native captured f32/f64 controls and snapshot identity");
            const auto admitted = JuicerTestSupport::EffectObservation::admit(effect);
            require(bool(admitted.state), "physical capture completes actual route publication");
            FjPreparedStateView preparedView{};
            read_prepared_view(*admitted.state, preparedView);
            const auto& effects = preparedView.effects;
            const nlohmann::json projection{{"film_dust", physical_words(effects.film_dust)}, {"film_scratch", physical_words(effects.film_scratch)}, {"gate_dust", physical_words(effects.gate_dust)}, {"gate_scratch", physical_words(effects.gate_scratch)}, {"weave", std::bit_cast<std::uint64_t>(effects.gate_weave_amount)}, {"active", effects.active != 0}, {"hash", effects.hash}};
            require(projection == row.at("recipe") && preparedView.grain.hash == row.at("grain_hash") && preparedView.recipe_hash == row.at("final_hash"), "actual OFX snapshot to completed grain to exact Rust effects and final publication");
            case_event.passed();
            ++count;
        }
        require(count == 20, "bounded physical OFX witness membership");
        for (const auto* getter : getters)
            host.set_double_value(getter, 0);
        host.set_str_choice_value(JuicerParams::kFilmProfileKey, kDefaultFilmProfileKey);
        host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(Spektrafilm::kDefaultScanRoute));
        host.set_int_value(JuicerParams::kGrainActive, 0);
        host.set_int_value(JuicerParams::kGrainDebugView, 0);
        host.unbind_effect();
        std::cout << "PASS adapter/20-frozen-physical-capture-publications-four-routes\n";
    }
    struct FrameGeometryFixturePaths {
        std::filesystem::path geometry;
        std::filesystem::path halation;
        std::filesystem::path diffusion;
    };
    std::vector<std::string> frame_geometry_case_ids() {
        std::vector<std::string> ids{"shared/off",
                                     "shared/same-state-roi-scale-par-destination",
                                     "shared/both-empty-window-fallback",
                                     "shared/same-state-changed-allocation",
                                     "shared/negative-nonzero-origin-portrait",
                                     "shared/admitted-f32-format-before-f64-pixel-arithmetic"};
        ids.reserve(57);
        for (unsigned route = 0; route < 4; ++route) {
            const std::string prefix = "route-" + std::to_string(route) + "/";
            ids.push_back(prefix + "four-route-distinct-pixel-definition-rejected");
            ids.push_back(prefix + "four-route-inactive-halation");
            ids.push_back(prefix + "four-route-active-metadata");
            for (unsigned mask = 0; mask < 4; ++mask)
                ids.push_back(prefix + "diffusion-camera-enlarger-combinations/mask-" + std::to_string(mask));
            ids.push_back(prefix + "active-diffusion-current-square-domain");
            ids.push_back(prefix + "effects-range-after-canonical");
            ids.push_back(prefix + "four-route-weave-full-definition");
            if (route % 2)
                ids.push_back(prefix + "print-partial-window-rejected-before-scalar");
        }
        for (const char* name : {"rust-diffusion-failure-short-circuits-halation-effects",
                                 "descriptor-failure-before-effects-query",
                                 "halation-pixel-failure-before-effects-query",
                                 "backend-before-images-and-state",
                                 "missing-image-before-state",
                                 "format-before-state",
                                 "single-empty-axis-before-state",
                                 "full-range-before-state",
                                 "invalid-state-before-scalar"})
            ids.push_back(std::string("shared/") + name);
        return ids;
    }
    void run_frame_geometry_cases(const FrameGeometryFixturePaths& fixtures, const char* preset, const std::filesystem::path& tracePath) {
        using namespace JuicerFrame::FrameGeometryTest;
        const auto& manifestPath = fixtures.geometry;
        const auto& halationManifestPath = fixtures.halation;
        const auto manifest = nlohmann::json::parse(std::ifstream(manifestPath));
        const auto captured = load_shared_fixture(manifestPath.parent_path() / manifest.at("presets").at(preset).at("native").get<std::string>(), "manifest.json");
        const auto halationManifest = nlohmann::json::parse(std::ifstream(halationManifestPath));
        const auto halationCaptured = load_shared_fixture(halationManifestPath.parent_path() / halationManifest.at("presets").at(preset).at("native").get<std::string>(), "manifest.json");
        require(halationCaptured.at("preset") == preset && halationCaptured.at("products").size() == 20, "halation callback frozen native context/membership");
        const auto diffusionManifest = nlohmann::json::parse(std::ifstream(fixtures.diffusion));
        const auto diffusionCaptured = load_shared_fixture(fixtures.diffusion.parent_path() / diffusionManifest.at("presets").at(preset).at("native").get<std::string>(), "manifest.json");
        require(diffusionCaptured.at("preset") == preset && diffusionCaptured.at("products").size() == 92, "diffusion callback frozen native context/membership");
        // Cohort membership follows callback scenarios; pixel-size expectations
        // consume only the named landscape, square, portrait, and fractional-film rows.
        const auto intended = frame_geometry_case_ids();
        JuicerTests::Cohort rows("frame_geometry", intended);
        std::string route_prefix = "shared/";
        std::string check_id_override;
        auto expected_pixel = [&](const char* name) {
            for (const auto& row : captured.at("geometry"))
                if (row.at("name") == name)
                    return row.at("pixel_bits").get<std::uint32_t>();
            throw std::runtime_error("missing frozen frame geometry case");
        };
        NativeHost host;
        EffectLease effect(*test_plugin, host.effect_handle());
        host.bind_effect(effect);
        host.set_time(12.5);
        host.set_double_value(JuicerParams::kCameraFilmFormatMm, 36.0);
        host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(Spektrafilm::ScanRoute::NegativeDirectScan));
        host.dispatch_event(effect, JuicerParams::kCameraFilmFormatMm, ChangeReason::PluginEdit);
        DeviceFrame frame;
        FrameGeometryCase geometry;
        struct ExpectedPhases {
            unsigned completed = 0;
            unsigned canonicalQueries = 0;
        };
        auto check = [&](const char* name, DeviceFrame& allocation, const FrameGeometryCase& facts, bool failed, ExpectedPhases expected, const char* pixelCase = "landscape", const std::function<void(const NativeHost::RenderResult&)>& after = {}) {
            const std::string membership_id = route_prefix + (check_id_override.empty() ? name : check_id_override);
            check_id_override.clear();
            auto case_event = rows.case_event(membership_id);
            const auto phases = expected.completed;
            const auto queries = expected.canonicalQueries;
            const auto traceBegin = host.trace_size();
            NativeHost::RenderResult result;
            try {
                result = host.render(effect, allocation, name, failed, &facts);
            } catch (...) {
                host.write_trace(tracePath);
                throw std::runtime_error(std::string("frame geometry callback case failed: ") + name);
            }
            const auto observation = snapshot();
            const auto policy = prepared_policy_snapshot();
            require(policy.count == (phases >= 2 ? 1u : 0u), std::string(name) + " accepted prepared-policy invocation count");
            require(diffusion_snapshot().count == 0 && halation_snapshot().count == 0,
                    std::string(name) + " production entry does not call retired native frame builders");
            if (policy.count) {
                const auto admission = JuicerTestSupport::EffectObservation::admit(effect);
                FjPreparedStateView published{};
                read_prepared_view(*admission.state, published);
                require(policy.route == published.route && policy.policy.recipe_hash == published.recipe_hash,
                        "frame policy comes from the currently admitted route publication");
                const auto& extent = policy.policy.geometry.full_frame;
                require(extent.x == facts.source.x1 && extent.y == facts.source.y1 && extent.width == facts.sourceWidth && extent.height == facts.sourceHeight,
                        "Rust prepared policy binds current full callback extent");
                require(std::bit_cast<std::uint32_t>(policy.policy.geometry.pixel_size_um) == expected_pixel(pixelCase),
                        "Rust prepared policy supplies frozen f32 pixel size");
                const auto& policyOptics = policy.policy.optics;
                constexpr std::uint32_t kDiffusionFlags =
                    FJ_OPTICS_CAMERA_DIFFUSION | FJ_OPTICS_ENLARGER_DIFFUSION;
                const std::uint32_t activeDiffusionFlags = policyOptics.flags & kDiffusionFlags;
                const auto diffusionComponentIsZero = [](const FjDiffusion& value) {
                    const auto zeroDoubles = [](const double* fields, std::size_t count) {
                        return std::all_of(fields, fields + count, [](double field) {
                            return std::bit_cast<std::uint64_t>(field) == 0;
                        });
                    };
                    return value.family == 0 &&
                           std::bit_cast<std::uint64_t>(value.scatter_fraction) == 0 &&
                           zeroDoubles(value.group_weights, 3) &&
                           zeroDoubles(value.group_lambda_um, 3) &&
                           std::bit_cast<std::uint64_t>(value.warmth) == 0 &&
                           std::bit_cast<std::uint64_t>(value.spatial_scale) == 0 &&
                           std::bit_cast<std::uint64_t>(value.pixel_size_um) == 0 &&
                           value.radius_px == 0 && value.sample_hash == 0 && value.hash == 0;
                };
                if (activeDiffusionFlags == 0) {
                    require(policyOptics.diffusion_hash == 0,
                            std::string("Rust prepared inactive diffusion has nonzero hash case=") + name +
                                " diffusion_hash=" + std::to_string(policyOptics.diffusion_hash));
                    require(diffusionComponentIsZero(policyOptics.camera),
                            std::string("Rust prepared inactive camera diffusion is nonzero case=") + name);
                    require(diffusionComponentIsZero(policyOptics.enlarger),
                            std::string("Rust prepared inactive enlarger diffusion is nonzero case=") + name);
                    const auto inactiveDiffusionRow = std::find_if(
                        diffusionCaptured.at("products").begin(),
                        diffusionCaptured.at("products").end(),
                        [&](const Json& row) {
                            return row.at("frame").is_null() &&
                                   row.at("input").at("route").get<unsigned>() == policy.route;
                        });
                    if (inactiveDiffusionRow == diffusionCaptured.at("products").end()) {
                        throw std::runtime_error(
                            std::string("frozen inactive diffusion row missing case=") + name +
                            " route=" + std::to_string(policy.route));
                    }
                    require(inactiveDiffusionRow->at("frame").is_null(),
                            std::string("frozen inactive diffusion row is not null case=") + name +
                                " fixture_case=" + inactiveDiffusionRow->at("input").at("name").get<std::string>());
                } else {
                    require(policyOptics.diffusion_hash != 0,
                            std::string("Rust prepared active diffusion has zero hash case=") + name +
                                " flags=" + std::to_string(activeDiffusionFlags));
                    const auto& frameExtent = policy.policy.geometry.full_frame;
                    const auto pixelBits = std::bit_cast<std::uint64_t>(
                        static_cast<double>(policy.policy.geometry.pixel_size_um));
                    const auto diffusionRow = std::find_if(
                        diffusionCaptured.at("products").begin(),
                        diffusionCaptured.at("products").end(),
                        [&](const Json& row) {
                            if (row.at("frame").is_null()) {
                                return false;
                            }
                            const auto& input = row.at("input");
                            return input.at("route").get<unsigned>() == policy.route &&
                                   input.at("pixel").get<std::uint64_t>() == pixelBits &&
                                   input.at("extent") == Json::array({frameExtent.x, frameExtent.y, frameExtent.width, frameExtent.height}) &&
                                   row.at("frame").at("hash").get<std::uint64_t>() == policyOptics.diffusion_hash;
                        });
                    if (diffusionRow == diffusionCaptured.at("products").end()) {
                        throw std::runtime_error(
                            std::string("Rust prepared active diffusion fixture row missing case=") + name +
                            " diffusion_hash=" + std::to_string(policyOptics.diffusion_hash) +
                            " pixel_bits=" + std::to_string(pixelBits) +
                            " extent=" + Json::array({frameExtent.x, frameExtent.y, frameExtent.width, frameExtent.height}).dump());
                    }
                    const Json projectedDiffusion = diffusion_policy_project(policy.policy, policy.route);
                    const Json& frozenDiffusion = diffusionRow->at("frame");
                    require(projectedDiffusion == frozenDiffusion,
                            std::string("Rust prepared diffusion differs from frozen row case=") + name +
                                " fixture_case=" + diffusionRow->at("input").at("name").get<std::string>() +
                                " diffusion_hash=" + std::to_string(policyOptics.diffusion_hash) +
                                " pixel_bits=" + std::to_string(pixelBits) +
                                " diff=" + Json::diff(frozenDiffusion, projectedDiffusion).dump());
                }
                const auto& policyHalation = policy.policy.optics.scatter_halation;
                const auto actualPixelBits = std::bit_cast<std::uint32_t>(policy.policy.geometry.pixel_size_um);
                if (policyHalation.recipe_hash == 0) {
                    const auto gaussianIsZero = [](const FjHalationGaussian& gaussian) {
                        return gaussian.kind == 0 && gaussian.radius == 0 &&
                               std::all_of(std::begin(gaussian.fir_weights), std::end(gaussian.fir_weights), [](float value) {
                                   return std::bit_cast<std::uint32_t>(value) == 0;
                               }) &&
                               std::bit_cast<std::uint32_t>(gaussian.feedforward) == 0 && std::all_of(std::begin(gaussian.feedback), std::end(gaussian.feedback), [](float value) {
                                   return std::bit_cast<std::uint32_t>(value) == 0;
                               });
                    };
                    const bool emptyMath =
                        std::bit_cast<std::uint32_t>(policyHalation.scatter_amount) == 0 &&
                        std::all_of(std::begin(policyHalation.channels), std::end(policyHalation.channels), [&](const FjHalationChannel& channel) {
                            return std::bit_cast<std::uint32_t>(channel.total_strength) == 0 &&
                                   gaussianIsZero(channel.core) &&
                                   std::all_of(std::begin(channel.tail), std::end(channel.tail), gaussianIsZero) &&
                                   std::all_of(std::begin(channel.bounce), std::end(channel.bounce), gaussianIsZero);
                        });
                    require(emptyMath,
                            std::string("Rust prepared inactive halation contains nonzero math case=") + name +
                                " recipe_hash=0 pixel_size_um_bits=" + std::to_string(actualPixelBits));
                    const auto inactiveRow = std::find_if(halationCaptured.at("products").begin(), halationCaptured.at("products").end(), [](const Json& row) {
                        return row.at("input").at("recipe_hash").get<std::uint64_t>() == 0;
                    });
                    require(inactiveRow != halationCaptured.at("products").end() &&
                                inactiveRow->at("descriptor").is_null(),
                            std::string("frozen inactive halation row missing/null mismatch case=") + name);
                } else {
                    const auto halationRow = std::find_if(halationCaptured.at("products").begin(), halationCaptured.at("products").end(), [&](const Json& row) {
                        return row.at("input").at("recipe_hash").get<std::uint64_t>() == policyHalation.recipe_hash &&
                               row.at("input").at("pixel_size_um").get<std::uint32_t>() == actualPixelBits;
                    });
                    if (halationRow == halationCaptured.at("products").end()) {
                        throw std::runtime_error(
                            std::string("Rust prepared active halation fixture row missing case=") + name +
                            " recipe_hash=" + std::to_string(policyHalation.recipe_hash) +
                            " pixel_size_um_bits=" + std::to_string(actualPixelBits));
                    }
                    const Json projectedHalation = halation_policy_project(policyHalation);
                    const Json& frozenHalation = halationRow->at("descriptor");
                    require(projectedHalation == frozenHalation,
                            std::string("Rust prepared halation descriptor differs from frozen row case=") + name +
                                " recipe_hash=" + std::to_string(policyHalation.recipe_hash) +
                                " pixel_size_um_bits=" + std::to_string(actualPixelBits) +
                                " fixture_case=" + halationRow->at("input").at("name").get<std::string>() +
                                " diff=" + Json::diff(frozenHalation, projectedHalation).dump());
                }
            }
            require(observation.count == phases, std::string(name) + " exact completed phase count");
            for (unsigned i = 0; i < phases; ++i)
                require(observation.phases[i] == i + 1, "full/scalar/effects phase order");
            require(host.canonical_queries() == queries, "conditional live canonical query count");
            if (queries) {
                require(host.canonical_time() == 12.5, "live canonical query uses args.time");
                require(std::count_if(host.trace().begin() + static_cast<std::ptrdiff_t>(traceBegin), host.trace().end(), [](const std::string& line) {
                            return line.starts_with("CANONICAL_QUERY clip=Source time=");
                        }) == queries,
                        "live canonical query belongs to the source clip");
            }
            if (facts.cuda && !facts.missingSource) {
                for (auto reads : {host.source_geometry_reads(), host.destination_geometry_reads()})
                    require(reads[0] > 0 && reads[1] > 0 && reads[2] > 0, "acquisition reads cached PAR/scale/pixel RoD even with effects off");
            }
            if (phases >= 1) {
                const auto& d = observation.fullDomain;
                require(d.originX == facts.source.x1 && d.originY == facts.source.y1 && d.width == facts.sourceWidth && d.height == facts.sourceHeight, "source allocation origin is current");
            }
            if (phases >= 2)
                require(std::bit_cast<std::uint32_t>(observation.pixelSizeUm) == expected_pixel(pixelCase), "actual callback consumes frozen final f32 pixel size");
            if (phases == 3) {
                const auto& e = observation.effects;
                require(e.pixelDefinition.x == facts.definition.x1 && e.pixelDefinition.y == facts.definition.y1 && e.pixelDefinition.width == facts.definitionWidth && e.pixelDefinition.height == facts.definitionHeight, "effects consumes source cached pixel RoD, distinct from allocation/window");
                require(e.canonicalX == facts.canonical.x1 && e.canonicalY == facts.canonical.y1 && e.canonicalWidth == 112.0 && e.canonicalHeight == 160.0 && e.scaleX == facts.scale.x && e.scaleY == facts.scale.y && e.pixelAspectRatio == facts.par, "effects consumes live canonical and cached source scale/PAR, distinct from destination and args scale");
            }
            if (after)
                after(result);
            case_event.passed();
            return result;
        };
        const auto first = check("off", frame, geometry, false, {2, 0});
        geometry.scale = {0.5, 0.25};
        geometry.par = 1.25;
        geometry.destination = {-1, -1, 5, 3};
        geometry.window = {1, 1, 5, 3};
        check("same-state-roi-scale-par-destination", frame, geometry, false, {2, 0}, "landscape", [&](const auto& result) {
            require(result.state.recipeHash == first.state.recipeHash && result.state.buildCounter == first.state.buildCounter, "callback geometry changes preserve immutable parameter state");
        });
        geometry.destination = geometry.source;
        geometry.window = {99, 88, 99, 88};
        check("both-empty-window-fallback", frame, geometry, false, {2, 0});
        DeviceFrame square({9, 9});
        auto squareFacts = geometry;
        squareFacts.source = {3, -4, 12, 5};
        squareFacts.destination = squareFacts.source;
        squareFacts.window = squareFacts.source;
        squareFacts.sourceWidth = squareFacts.sourceHeight = 9;
        check("same-state-changed-allocation", square, squareFacts, false, {2, 0}, "square", [&](const auto& result) {
            require(result.state.recipeHash == first.state.recipeHash && result.state.buildCounter == first.state.buildCounter, "new allocation derives new geometry without rebuilding state");
        });
        DeviceFrame portrait({5, 7});
        auto portraitFacts = geometry;
        portraitFacts.source = {-11, 17, -6, 24};
        portraitFacts.destination = portraitFacts.source;
        portraitFacts.window = portraitFacts.source;
        portraitFacts.sourceWidth = 5;
        portraitFacts.sourceHeight = 7;
        check("negative-nonzero-origin-portrait", portrait, portraitFacts, false, {2, 0}, "portrait");
        geometry.window = geometry.source;
        host.set_double_value(JuicerParams::kCameraFilmFormatMm, 35.749999);
        host.dispatch_event(effect, JuicerParams::kCameraFilmFormatMm, ChangeReason::PluginEdit);
        check("admitted-f32-format-before-f64-pixel-arithmetic", frame, geometry, false, {2, 0}, "film-fractional");
        host.set_double_value(JuicerParams::kCameraFilmFormatMm, 36.0);
        host.dispatch_event(effect, JuicerParams::kCameraFilmFormatMm, ChangeReason::PluginEdit);
        for (unsigned route = 0; route < 4; ++route) {
            route_prefix = "route-" + std::to_string(route) + "/";
            host.set_str_choice_value(JuicerParams::kFilmProfileKey, route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f");
            host.dispatch_event(effect, JuicerParams::kFilmProfileKey, ChangeReason::PluginEdit);
            host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(route % 2 ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan));
            host.set_int_value(kParamSpectralMode, static_cast<int>(route % 3));
            host.set_int_value(JuicerParams::kGrainActive, route % 2 == 0);
            host.set_double_value(JuicerParams::kFilmDustAmount, 0.1);
            host.set_double_value(JuicerParams::kGateDustAmount, 0.1);
            host.set_double_value(JuicerParams::kGateWeaveAmount, 0.0);
            host.set_double_value(JuicerParams::kFilmScratchAmount, 0.1);
            host.set_double_value(JuicerParams::kGateScratchAmount, 0.1);
            host.set_int_value(JuicerParams::kCameraDiffusionEnabled, route < 2);
            host.set_int_value(JuicerParams::kPrintDiffusionEnabled, static_cast<int>(route % 2));
            host.set_int_value(JuicerParams::kHalationActive, route >= 2);
            host.dispatch_event(effect, JuicerParams::kFilmDustAmount, ChangeReason::PluginEdit);
            check("four-route-distinct-pixel-definition-rejected", frame, geometry, true, {3, 1});
            auto active = geometry;
            active.definition = geometry.source;
            active.definitionWidth = 7;
            active.definitionHeight = 5;
            host.set_int_value(JuicerParams::kHalationActive, 0);
            host.dispatch_event(effect, JuicerParams::kHalationActive, ChangeReason::PluginEdit);
            check("four-route-inactive-halation", frame, active, false, {3, 1}, "landscape", [&](const auto&) {
                require((prepared_policy_snapshot().policy.optics.flags & FJ_OPTICS_SCATTER_HALATION) == 0,
                        "inactive halation is absent from accepted Rust frame policy");
            });
            host.set_int_value(JuicerParams::kHalationActive, 1);
            host.dispatch_event(effect, JuicerParams::kHalationActive, ChangeReason::PluginEdit);
            check("four-route-active-metadata", frame, active, false, {3, 1}, "landscape", [&](const auto&) {
                require((prepared_policy_snapshot().policy.optics.flags & FJ_OPTICS_SCATTER_HALATION) != 0,
                        "active halation reaches accepted Rust frame policy on each route/polarity");
            });
            for (unsigned mask = 0; mask < 4; ++mask) {
                check_id_override = "diffusion-camera-enlarger-combinations/mask-" + std::to_string(mask);
                host.set_int_value(JuicerParams::kCameraDiffusionEnabled, (mask & 1) != 0);
                host.set_int_value(JuicerParams::kPrintDiffusionEnabled, (mask & 2) != 0);
                host.dispatch_event(effect, JuicerParams::kCameraDiffusionEnabled, ChangeReason::PluginEdit);
                const unsigned requestedStages = (mask & 1u ? FJ_OPTICS_CAMERA_DIFFUSION : 0u) |
                                                 (route % 2 ? (mask & 2u ? FJ_OPTICS_ENLARGER_DIFFUSION : 0u) : 0u);
                check("diffusion-camera-enlarger-combinations", frame, active, false, {3, 1}, "landscape", [&](const auto&) {
                    const unsigned actualStages = prepared_policy_snapshot().policy.optics.flags &
                                                  (FJ_OPTICS_CAMERA_DIFFUSION | FJ_OPTICS_ENLARGER_DIFFUSION);
                    require(actualStages == requestedStages, "accepted Rust frame policy preserves camera/enlarger stage selection on all routes");
                });
            }
            auto diffusionSquare = squareFacts;
            diffusionSquare.definition = diffusionSquare.source;
            diffusionSquare.definitionWidth = diffusionSquare.definitionHeight = 9;
            check("active-diffusion-current-square-domain", square, diffusionSquare, false, {3, 1}, "square");
            auto badDefinition = geometry;
            badDefinition.definition = {std::numeric_limits<int>::min(), 0, std::numeric_limits<int>::max(), 5};
            check("effects-range-after-canonical", frame, badDefinition, true, {2, 1});
            auto weave = geometry;
            weave.definition = geometry.source;
            weave.definitionWidth = 7;
            weave.definitionHeight = 5;
            host.set_double_value(JuicerParams::kGateWeaveAmount, 0.1);
            host.dispatch_event(effect, JuicerParams::kGateWeaveAmount, ChangeReason::PluginEdit);
            check("four-route-weave-full-definition", frame, weave, false, {3, 1});
            host.set_double_value(JuicerParams::kGateWeaveAmount, 0.0);
            host.dispatch_event(effect, JuicerParams::kGateWeaveAmount, ChangeReason::PluginEdit);
            if (route % 2) {
                auto partial = geometry;
                partial.window = {1, 1, 5, 3};
                check("print-partial-window-rejected-before-scalar", frame, partial, true, {1, 0});
            }
        }
        route_prefix = "shared/";
        host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(Spektrafilm::ScanRoute::NegativeDirectScan));
        host.set_int_value(JuicerParams::kCameraDiffusionEnabled, 1);
        host.dispatch_event(effect, JuicerParams::kCameraDiffusionEnabled, ChangeReason::PluginEdit);
        auto diffusionFailure = geometry;
        diffusionFailure.definition = geometry.source;
        diffusionFailure.definitionWidth = 7;
        diffusionFailure.definitionHeight = 5;
        require(fj_test_diffusion_frame_fault(3, 1).category == FJ_STATUS_SUCCESS, "arm bounded diffusion callback failure");
        check("rust-diffusion-failure-short-circuits-halation-effects", frame, diffusionFailure, true, {1, 0});
        auto invalidPhysical = geometry;
        invalidPhysical.source = {7, 5, 0, 0};
        invalidPhysical.sourceWidth = -7;
        invalidPhysical.sourceHeight = -5;
        check("descriptor-failure-before-effects-query", frame, invalidPhysical, true, {1, 0}, "negative");
        host.set_int_value(JuicerParams::kCameraDiffusionEnabled, 0);
        host.set_int_value(JuicerParams::kHalationActive, 1);
        host.dispatch_event(effect, JuicerParams::kCameraDiffusionEnabled, ChangeReason::PluginEdit);
        check("halation-pixel-failure-before-effects-query", frame, invalidPhysical, true, {1, 0}, "negative");
        host.set_double_value(JuicerParams::kCameraFilmFormatMm, -1);
        host.dispatch_event(effect, JuicerParams::kCameraFilmFormatMm, ChangeReason::PluginEdit);
        require(JuicerTestSupport::EffectObservation::snapshot(effect).pendingInvalid, "pending controls invalid for priority cases");
        auto invalid = geometry;
        invalid.cuda = false;
        invalid.expectedLeases = 0;
        check("backend-before-images-and-state", frame, invalid, true, {0, 0});
        invalid = geometry;
        invalid.missingSource = true;
        invalid.expectedLeases = 1;
        check("missing-image-before-state", frame, invalid, false, {0, 0});
        invalid = geometry;
        invalid.nonfloat = true;
        check("format-before-state", frame, invalid, true, {0, 0});
        invalid = geometry;
        invalid.window = {1, 0, 1, 4};
        check("single-empty-axis-before-state", frame, invalid, false, {0, 0});
        invalid = geometry;
        invalid.source = {std::numeric_limits<int>::min(), 0, std::numeric_limits<int>::max(), 5};
        check("full-range-before-state", frame, invalid, true, {0, 0});
        check("invalid-state-before-scalar", frame, geometry, true, {1, 0});
        host.unbind_effect();
        host.write_trace(tracePath);
        std::cout << "PASS prepared Rust frame policy, three phases, four routes, failure priority and image leases\n";
    }

} // namespace

int main(int argc, char** argv) try { // NOLINT(bugprone-exception-escape): top-level catches map all fixture failures.
    if (JuicerTests::list_requested(argc, argv)) {
        if (argc == 8 && std::string_view(argv[1]) == "--frame-geometry") {
            JuicerTests::list_case("adapter_operation", "frame_geometry");
            for (const auto& id : frame_geometry_case_ids())
                JuicerTests::list_case("frame_geometry", id);
            return 0;
        }
        if (argc == 6 && (std::string_view(argv[1]) == "--loaded-module" || std::string_view(argv[1]) == "--stream-contract")) {
            JuicerTests::list_case("adapter_operation", std::string(std::string_view(argv[1]) == "--stream-contract" ? "stream_contract" : "loaded_module"));
            if (std::string_view(argv[1]) == "--stream-contract")
                for (const char* stream : {"supplied-nonnull", "present-null", "absent"})
                    for (const char* delay : {"immediate", "delayed"})
                        JuicerTests::list_case("stream_render", std::string(stream) + "/" + delay);
            return 0;
        }
        if ((argc == 4 || argc == 7) && std::string_view(argv[1]) == "--trace-output") {
            JuicerTests::list_case("adapter_operation", "adapter_trace");
            for (const char* id : {"never-rendered", "live-peer", "retirement-failure", "partial-construction"})
                JuicerTests::list_case("submission_retirement", id);
            for (int failed_call : {1, 2})
                JuicerTests::list_case("route_constructor_failure", std::to_string(failed_call));
            for (unsigned failure = 0; failure < 3; ++failure)
                JuicerTests::list_case("route_failure_recovery", std::to_string(failure + 1));
            for (const char* route : {"direct", "print"}) {
                for (unsigned film = 0; film < 3; ++film)
                    for (unsigned print = 0; print < 3; ++print)
                        JuicerTests::list_case("gamma_capture", std::string(route) + "/film/" + std::to_string(film) + "/print/" + std::to_string(print));
                JuicerTests::list_case("gamma_capture", std::string(route) + "/print-range");
                for (unsigned invalid = 0; invalid < 3; ++invalid)
                    JuicerTests::list_case("gamma_capture", std::string(route) + "/invalid-film/" + std::to_string(invalid));
                for (unsigned policy = 0; policy < 3; ++policy)
                    JuicerTests::list_case("gamma_capture", std::string(route) + "/grain-policy/" + std::to_string(policy));
                for (unsigned preset = 0; preset < 3; ++preset)
                    for (unsigned policy = 0; policy < 3; ++policy)
                        for (unsigned sample = 0; sample < 11; ++sample)
                            JuicerTests::list_case("grain_capture", std::string(route) + "/preset/" + std::to_string(preset) + "/policy/" + std::to_string(policy) + "/sample/" + std::to_string(sample));
                for (int choice : {-8, 0, 1, 2, 8}) {
                    JuicerTests::list_case("grain_edit", std::string(route) + "/choice/" + std::to_string(choice));
                    JuicerTests::list_case("grain_edit", std::string(route) + "/fallback/" + std::to_string(choice));
                }
                JuicerTests::list_case("grain_edit", std::string(route) + "/master-ratios");
                JuicerTests::list_case("grain_edit", std::string(route) + "/manual-custom");
                JuicerTests::list_case("grain_edit", std::string(route) + "/linked-sharpness-texture");
                JuicerTests::list_case("grain_edit", std::string(route) + "/reset-advanced");
                for (unsigned time = 0; time < 4; ++time)
                    JuicerTests::list_case("grain_edit", std::string(route) + "/time/" + std::to_string(time));
            }
            for (const char* route : {"negative-direct", "negative-print", "positive-direct", "positive-print"}) {
                JuicerTests::list_case("dir_capture", route);
                for (unsigned flags = 0; flags < 4; ++flags)
                    JuicerTests::list_case("scanner_capture", std::string(route) + "/flags/" + std::to_string(flags));
                JuicerTests::list_case("scanner_capture", std::string(route) + "/sanitation-and-recovery");
            }
            if (argc == 7 && std::string_view(argv[3]) == "--grain-fixtures") {
                const std::filesystem::path grain_manifest(argv[4]);
                const std::string preset(argv[5]);
                (void)preset;
                for (unsigned route = 0; route < 4; ++route)
                    for (unsigned input = 0; input < 24; ++input)
                        JuicerTests::list_case("optics_capture", "route-" + std::to_string(route) + "/input-" + std::to_string(input));
                const auto effects_path = grain_manifest.parent_path().parent_path() / "effects-fixtures/manifest.json";
                const auto effects_manifest = nlohmann::json::parse(std::ifstream(effects_path));
                const auto effects_path_data = effects_path.parent_path() / effects_manifest.at("presets").at(preset).at("ofx").get<std::string>();
                const auto effects_capture = PrintIlluminantFixtures::apply(effects_path_data, nlohmann::json::parse(std::ifstream(effects_path_data))).at("ofx");
                for (const auto& row : effects_capture) {
                    const unsigned policy = row.at("policy");
                    const unsigned index = row.at("index");
                    if ((index == 5 && (policy == 0 || policy == 1 || policy == 4 || policy == 7)) || (index == 15 && policy == 0))
                        JuicerTests::list_case("physical_capture", "route-" + std::to_string(row.at("route").get<unsigned>()) + "/policy-" + std::to_string(policy) + "/index-" + std::to_string(index));
                }
            }
            return 0;
        }
        return 2;
    }
    JuicerTests::CaseLog case_log;
    std::string operation_id = "adapter_trace";
    if (argc > 1) {
        const std::string_view mode(argv[1]);
        if (mode == "--frame-geometry")
            operation_id = "frame_geometry";
        else if (mode == "--stream-contract")
            operation_id = "stream_contract";
        else if (mode == "--loaded-module")
            operation_id = "loaded_module";
    }
    JuicerTests::CaseEvent operation("adapter_operation", operation_id);
    try {
        if (argc == 5 && (std::string_view(argv[1]) == "--loaded-module" || std::string_view(argv[1]) == "--stream-contract") &&
            std::string_view(argv[3]) == "--trace-output") {
            NativeHost host;
            run_loaded_module_callback(host, argv[2], argv[4], std::string_view(argv[1]) == "--stream-contract");
            operation.passed();
            case_log.finish();
            return 0;
        }
        NativeHost binding;
        test_plugin = OfxGetPlugin(0);
        require(test_plugin && test_plugin->setHost && test_plugin->mainEntry, "standard OFX descriptor");
        test_plugin->setHost(binding.ofx_host());
        require(test_plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr) == kOfxStatOK, "standard OFX load");
        struct LoadedGuard {
            OfxPlugin* plugin;
            ~LoadedGuard() {
                if (plugin)
                    (void)plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr);
            }
            OfxStatus close() {
                auto* current = std::exchange(plugin, nullptr);
                return current->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr);
            }
        } loaded{test_plugin};
        require(binding.invoke(*test_plugin, kOfxActionDescribe, binding.descriptor_handle(), nullptr) == kOfxStatOK,
                "standard OFX describe");
        require(binding.invoke(*test_plugin, kOfxImageEffectActionDescribeInContext, binding.descriptor_handle(), binding.describe_context_handle()) == kOfxStatOK,
                "standard OFX filter context");
        FjRuntimeView runtime{};
        runtime.cuda = fj_test_ofx_native();
        runtime.assets = static_cast<const FjAssets*>(fj_test_ofx_assets());
        require(runtime.cuda && runtime.assets, "loaded OFX Runtime backing");
        if (argc == 7 && std::string_view(argv[1]) == "--frame-geometry") {
            if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess)
                throw std::runtime_error("CUDA initialization failed");
            run_frame_geometry_cases({.geometry = argv[2], .halation = argv[5], .diffusion = argv[6]}, argv[3], argv[4]);
            require(loaded.close() == kOfxStatOK, "standard OFX unload");
            operation.passed();
            case_log.finish();
            return 0;
        }
        if ((argc != 3 && !(argc == 6 && std::string_view(argv[3]) == "--grain-fixtures")) || std::string_view(argv[1]) != "--trace-output") {
            throw std::runtime_error(
                "usage: JuicerAdapterTraceProbe --trace-output PATH or "
                "--loaded-module MODULE --trace-output PATH");
        }
        const std::filesystem::path tracePath = argv[2];
        if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
            throw std::runtime_error("CUDA initialization failed");
        }
        if (argc == 6)
            check_optics_capture(argv[4], argv[5]);
        run_submission_retirement_rows(runtime);
        run_instance_contract_rows();
        NativeHost host;
        if (argc == 6)
            check_physical_capture(host, argv[4], argv[5]);
        JuicerTestSupport::PreparationInput routeSnapshot;
        routeSnapshot.values.scan_route = static_cast<std::uint32_t>(Spektrafilm::ScanRoute::NegativeDirectScan);
        std::optional<JuicerTestSupport::PreparedStateHold> routeProduct;
        std::string routeDiagnostic;
        const auto prepareRoute = [&] {
            std::array<char, 2048> text{};
            FjErrorBuffer error{text.data(), text.size(), 0};
            const FjStatus status = prepare_state(runtime.assets,
                                                  routeSnapshot,
                                                  routeProduct,
                                                  error);
            routeDiagnostic.assign(text.data(), std::min(error.length, text.size()));
            require(status.api == FJ_API_NONE && status.native_code == 0 &&
                        bool(routeProduct) == (status.category == FJ_STATUS_SUCCESS),
                    "complete route construction returned inconsistent ownership");
            if (!routeProduct)
                return false;
            FjPreparedStateView view{};
            require(routeProduct->view(view, error).category == FJ_STATUS_SUCCESS &&
                        view.route == routeSnapshot.values.scan_route,
                    "complete route construction changed selected route");
            return true;
        };
        require(
            prepareRoute(),
            "matching negative profile route did not build: " + routeDiagnostic);
        routeSnapshot.values.scan_route = static_cast<std::uint32_t>(Spektrafilm::ScanRoute::PositiveDirectScan);
        routeDiagnostic.clear();
        require(
            !prepareRoute() &&
                routeDiagnostic.find("route mismatch") != std::string::npos,
            "mismatched negative profile route was not rejected");
        routeSnapshot.film_key = "fujifilm_provia_100f";
        routeDiagnostic.clear();
        require(
            prepareRoute(),
            "matching positive profile route did not build: " + routeDiagnostic);
        routeSnapshot.values.scan_route = static_cast<std::uint32_t>(Spektrafilm::ScanRoute::NegativeDirectScan);
        routeDiagnostic.clear();
        require(
            !prepareRoute() &&
                routeDiagnostic.find("route mismatch") != std::string::npos,
            "mismatched positive profile route was not rejected");
        {
            EffectLease routeGuard(*test_plugin, host.effect_handle());
            host.bind_effect(routeGuard);
            host.fail_scan_route_getter(1);
            routeSnapshot.values.scan_route = static_cast<std::uint32_t>(Spektrafilm::ScanRoute::PositiveDirectScan);
            routeDiagnostic.clear();
            require(prepareRoute(), "complete positive route preparation must not consult the host route getter");
            const auto traceStart = host.trace_size();
            OfxStatus status = kOfxStatOK;
            try {
                host.dispatch_event(routeGuard, kParamExposure, ChangeReason::UserEdit);
            } catch (const StatusFailure& error) {
                status = error.status();
            }
            const auto observed = JuicerTestSupport::EffectObservation::snapshot(routeGuard);
            require(status == kOfxStatErrBadHandle && observed.pendingInvalid &&
                        observed.pendingDiagnostic.find("host parameter action failed") != std::string::npos &&
                        failure_ui_trace_matches(summarize_ui_trace(host.trace(), traceStart), 1),
                    "independent preparation leaves the actual host route getter fault armed");
            host.unbind_effect();
        }
        routeProduct.reset();
        DeviceFrame frame;
        const std::array<std::string, 2> constructor_failure_ids{{"1", "2"}};
        JuicerTests::Cohort constructor_failure_rows("route_constructor_failure", std::vector<std::string>(constructor_failure_ids.begin(), constructor_failure_ids.end()));
        for (int failedCall : {1, 2}) {
            auto row = constructor_failure_rows.case_event(std::to_string(failedCall));
            const std::size_t traceStart = host.trace_size();
            host.fail_scan_route_getter(static_cast<unsigned>(failedCall));
            OfxStatus creationStatus = kOfxStatOK;
            try {
                EffectLease failedEffect(*test_plugin, host.effect_handle());
            } catch (const StatusFailure& error) {
                creationStatus = error.status();
            }
            const auto trace = summarize_ui_trace(host.trace(), traceStart);
            require(creationStatus == kOfxStatErrBadHandle &&
                        failure_ui_trace_matches(trace, failedCall) &&
                        host.instance_data() == nullptr,
                    "failed route getter must stop creation before route writes and detach instance data");
            row.passed();
        }
        constexpr std::array<double, 3> recoveryExposureEv{{0.125, -0.25, 0.375}};
        const std::array<std::string, 3> recovery_ids{{"1", "2", "3"}};
        JuicerTests::Cohort route_recovery_rows("route_failure_recovery", std::vector<std::string>(recovery_ids.begin(), recovery_ids.end()));
        {
            EffectLease effect(*test_plugin, host.effect_handle());
            host.bind_effect(effect);

            host.set_time(0.0);
            host.fail_scan_route_getter(1);
            const auto initialRejected =
                host.render(effect, frame, "initial-route-failure", true);
            require(initialRejected.rejected && initialRejected.suiteStatus == kOfxStatErrBadHandle && initialRejected.state.pendingInvalid &&
                        initialRejected.state.pendingDiagnostic.find("host parameter capture failed") !=
                            std::string::npos,
                    "initial bridge failure did not block admission");
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
            const auto first = host.render(effect, frame, "first-render");
            require(first.succeeded && exposure_matches(first, 0.0) &&
                        has_valid_latch(first),
                    "first render did not acquire the current zero-EV value");

            for (std::size_t failureIndex = 0;
                 failureIndex < recoveryExposureEv.size();
                 ++failureIndex) {
                auto row = route_recovery_rows.case_event(std::to_string(failureIndex + 1));
                const int failedCall = static_cast<int>(failureIndex) + 1;
                const auto retainedSnapshot = JuicerTestSupport::EffectObservation::captured(effect);
                const JuicerTestSupport::PublicationAdmission retained =
                    JuicerTestSupport::EffectObservation::admit(effect);
                const bool retainedPrint = (JuicerTestSupport::published_view(retained).route & 1u) != 0;
                require(retained.kind == FJ_ADMISSION_READY && retained.state,
                        "could not retain an admitted route state before failure");
                const auto retainedMetrics = JuicerTestSupport::EffectObservation::metrics(effect);
                const std::size_t retainedStateIdentity = retainedMetrics.backingIdentity;
                const std::uint64_t retainedBuildCounter = retained.state->build();
                const std::uint64_t retainedRecipeHash = prepared_recipe_hash(*retained.state);
                const std::uint32_t retainedRoute = retainedSnapshot.values.scan_route;
                const double retainedExposureEv = retainedSnapshot.values.camera_exposure_compensation_ev;
                const auto retained_state_unchanged = [&]() {
                    const auto view = JuicerTestSupport::published_view(retained);
                    return retained.kind == FJ_ADMISSION_READY && retained.state &&
                           view.route == static_cast<std::uint32_t>(retainedRoute) &&
                           retained.state->build() == retainedBuildCounter &&
                           prepared_recipe_hash(*retained.state) == retainedRecipeHash &&
                           double_matches(retainedSnapshot.values.camera_exposure_compensation_ev, retainedExposureEv);
                };

                host.set_double_value(
                    kParamExposure,
                    recoveryExposureEv[failureIndex]);
                const std::size_t traceStart = host.trace_size();
                host.fail_scan_route_getter(static_cast<unsigned>(failedCall));
                OfxStatus failedStatus = kOfxStatOK;
                try {
                    host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
                } catch (const StatusFailure& error) {
                    failedStatus = error.status();
                }
                const auto invalid =
                    JuicerTestSupport::EffectObservation::snapshot(effect);
                const JuicerTestSupport::PublicationAdmission blocked =
                    JuicerTestSupport::EffectObservation::admit(effect);
                const auto rejected = host.render(
                    effect, frame, "route-bridge-failure", true);
                const UiTraceSummary failureTrace =
                    summarize_ui_trace(host.trace(), traceStart);
                require(failedStatus == kOfxStatErrBadHandle && invalid.pendingInvalid &&
                            invalid.pendingDiagnostic.find("host parameter action failed") !=
                                std::string::npos &&
                            blocked.kind == FJ_ADMISSION_CAPTURE_FAILED &&
                            !blocked.state &&
                            rejected.rejected && rejected.suiteStatus == kOfxStatErrFatal && rejected.state.pendingInvalid &&
                            failure_ui_trace_matches(failureTrace, failedCall) &&
                            retained_state_unchanged(),
                        "route bridge failure did not block new admission while retaining the admitted state");
                host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
                const auto recoveredSnapshot = JuicerTestSupport::EffectObservation::captured(effect);
                const JuicerTestSupport::PublicationAdmission recoveredAdmission =
                    JuicerTestSupport::EffectObservation::admit(effect);
                const auto recoveredRoute =
                    host.render(effect, frame, "route-bridge-recovery");
                require(recoveredAdmission.kind == FJ_ADMISSION_READY && recoveredAdmission.state &&
                            ((JuicerTestSupport::published_view(recoveredAdmission).route & 1u) != 0) == retainedPrint,
                        "valid route recovery did not return the expected admission type");
                const std::size_t recoveredStateIdentity = JuicerTestSupport::EffectObservation::metrics(effect).backingIdentity;
                const std::uint64_t recoveredBuildCounter = recoveredAdmission.state->build();
                const std::uint64_t recoveredRecipeHash = prepared_recipe_hash(*recoveredAdmission.state);
                require(recoveredRoute.succeeded &&
                            !recoveredRoute.state.pendingInvalid &&
                            exposure_matches(
                                recoveredRoute,
                                recoveryExposureEv[failureIndex]) &&
                            recoveredSnapshot.values.scan_route == retainedRoute &&
                            double_matches(
                                recoveredSnapshot.values
                                    .camera_exposure_compensation_ev,
                                recoveryExposureEv[failureIndex]) &&
                            recoveredStateIdentity != retainedStateIdentity &&
                            recoveredBuildCounter > retainedBuildCounter &&
                            recoveredRecipeHash != retainedRecipeHash &&
                            recoveredRoute.state.recipeHash ==
                                recoveredRecipeHash &&
                            retained_state_unchanged(),
                        "route bridge failure did not replace publication while preserving the retained admission");
                row.passed();
            }

            host.set_str_choice_value(
                JuicerParams::kParamScanRoute,
                Spektrafilm::scan_route_key(
                    Spektrafilm::ScanRoute::NegativeDirectScan));
            std::size_t uiTraceStart = host.trace_size();
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
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
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
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
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
            const auto reentryState =
                JuicerTestSupport::EffectObservation::snapshot(effect);
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
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);

            host.set_str_choice_value(JuicerParams::kFilmProfileKey, "missing-profile");
            host.set_str_choice_value(JuicerParams::kParamScanRoute, "missing-route");
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
            require(
                JuicerTestSupport::EffectObservation::snapshot(effect).pendingRoute ==
                    Spektrafilm::ScanRoute::NegativePrintScan,
                "unknown profile and route keys did not use the negative print default");

            host.set_str_choice_value(JuicerParams::kFilmProfileKey, "fujifilm_provia_100f");
            host.set_str_choice_value(JuicerParams::kParamScanRoute, "");
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
            require(
                JuicerTestSupport::EffectObservation::snapshot(effect).pendingRoute ==
                    Spektrafilm::ScanRoute::PositiveDirectScan,
                "empty route key did not use the positive direct default");

            host.set_str_choice_value(
                JuicerParams::kFilmProfileKey, kDefaultFilmProfileKey);
            host.set_str_choice_value(
                JuicerParams::kParamScanRoute,
                Spektrafilm::scan_route_key(Spektrafilm::kDefaultScanRoute));
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
            const auto restoredRoute = host.render(effect, frame, "route-fallback-restored");
            require(restoredRoute.succeeded &&
                        restoredRoute.state.recipeHash == first.state.recipeHash,
                    "route fallback test did not restore the original route");

            host.set_time(10.0);
            host.set_double_key(kParamExposure, 10.0, 0.25);
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
            const auto timeTen = host.render(effect, frame, "authored-time-10");
            require(timeTen.succeeded && exposure_matches(timeTen, 0.25) &&
                        has_valid_latch(timeTen) &&
                        timeTen.state.latchSnapshotId !=
                            first.state.latchSnapshotId,
                    "time-10 event did not admit its authored value");

            host.set_time(20.0);
            host.set_double_key(kParamExposure, 20.0, -0.5);
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
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
            host.dispatch_event(effect, kParamExposure, ChangeReason::Time);
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
                ChangeReason::UserEdit);
            const auto presetState =
                JuicerTestSupport::EffectObservation::snapshot(effect);
            require(
                host.nested_event_count() - nestedBeforePreset == 22 &&
                    !host.nested_event_failed() &&
                    !host.nested_contract_failed() &&
                    !presetState.paramEventsSuppressed &&
                    presetState.pendingValid &&
                    presetState.input.grain.amplitude == static_cast<float>(std::exp2(-0.57)) &&
                    presetState.input.grain.chroma_mix == 0.50f &&
                    presetState.input.grain.chroma_shared_weight == std::sqrt(0.50f) &&
                    presetState.input.grain.chroma_independent_weight == std::sqrt(0.50f) &&
                    presetState.input.grain.particle_area_um2 == 0.33f &&
                    presetState.input.grain.correlation_sigma_px == 0.56f &&
                    presetState.input.grain.size_mix_scale == 16.0f &&
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
                ChangeReason::UserEdit);
            const auto resetState =
                JuicerTestSupport::EffectObservation::snapshot(effect);
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
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
            const auto edit = host.render(effect, frame, "edit-ev-0.5");
            host.set_double_key(kParamExposure, 10.0, 0.25);
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
            const auto undo = host.render(effect, frame, "undo-ev-0.25");
            host.set_double_key(kParamExposure, 10.0, 0.5);
            host.dispatch_event(effect, kParamExposure, ChangeReason::UserEdit);
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
                ChangeReason::UserEdit);
            const auto invalidBeforeRender =
                JuicerTestSupport::EffectObservation::snapshot(effect);
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
                ChangeReason::UserEdit);
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
            std::vector<std::string> gamma_intended;
            gamma_intended.reserve(50);
            for (const char* route : {"direct", "print"}) {
                for (unsigned film = 0; film < 3; ++film)
                    for (unsigned print = 0; print < 3; ++print)
                        gamma_intended.push_back(std::string(route) + "/film/" + std::to_string(film) + "/print/" + std::to_string(print));
                gamma_intended.push_back(std::string(route) + "/print-range");
                for (unsigned invalid = 0; invalid < 3; ++invalid)
                    gamma_intended.push_back(std::string(route) + "/invalid-film/" + std::to_string(invalid));
                for (unsigned policy = 0; policy < 3; ++policy)
                    gamma_intended.push_back(std::string(route) + "/grain-policy/" + std::to_string(policy));
            }
            JuicerTests::Cohort gamma_rows("gamma_capture", gamma_intended);
            for (bool print : {false, true}) {
                const std::string routeName = print ? "print" : "direct";
                host.set_str_choice_value(JuicerParams::kParamScanRoute,
                                          Spektrafilm::scan_route_key(print ? Spektrafilm::ScanRoute::NegativePrintScan
                                                                            : Spektrafilm::ScanRoute::NegativeDirectScan));
                const std::array<double, 3> film_gammas{0.05, 1.234567890123, 4.0};
                const std::array<double, 3> print_gammas{0.5, 1.234567890123, 2.0};
                for (std::size_t filmIndex = 0; filmIndex < film_gammas.size(); ++filmIndex) {
                    for (std::size_t printIndex = 0; printIndex < print_gammas.size(); ++printIndex) {
                        auto row = gamma_rows.case_event(routeName + "/film/" + std::to_string(filmIndex) + "/print/" + std::to_string(printIndex));
                        const double filmGamma = film_gammas[filmIndex];
                        const double printGamma = print_gammas[printIndex];
                        host.set_double_value(JuicerParams::kFilmGammaFactor, filmGamma);
                        host.set_double_value(JuicerParams::kPrintGammaFactor, printGamma);
                        host.dispatch_event(effect, JuicerParams::kFilmGammaFactor, ChangeReason::UserEdit);
                        const auto observed = JuicerTestSupport::EffectObservation::snapshot(effect);
                        require(observed.pendingValid &&
                                    std::bit_cast<std::uint32_t>(observed.pendingFilmGamma) ==
                                        std::bit_cast<std::uint32_t>(static_cast<float>(filmGamma)) &&
                                    std::bit_cast<std::uint64_t>(observed.pendingPrintGamma) ==
                                        std::bit_cast<std::uint64_t>(printGamma),
                                "OFX film-f32/print-f64 capture bits changed");
                        row.passed();
                    }
                }
                host.set_double_value(JuicerParams::kFilmGammaFactor, 1.0);
                host.set_double_value(JuicerParams::kPrintGammaFactor, 0.25);
                host.dispatch_event(effect, JuicerParams::kPrintGammaFactor, ChangeReason::UserEdit);
                const auto printInvalid = JuicerTestSupport::EffectObservation::snapshot(effect);
                auto print_range_row = gamma_rows.case_event(routeName + "/print-range");
                require(print ? printInvalid.pendingInvalid : printInvalid.pendingValid,
                        "OFX print gamma validation lost its print-only range");
                host.set_double_value(JuicerParams::kPrintGammaFactor, 1.0);
                host.dispatch_event(effect, JuicerParams::kPrintGammaFactor, ChangeReason::UserEdit);
                const auto retained = JuicerTestSupport::EffectObservation::admit(effect);
                require(bool(retained.state),
                        "gamma recovery failed to admit a complete state");
                print_range_row.passed();
                const std::array<double, 3> invalid_gammas{std::nextafter(0.05, 0.0), std::nextafter(4.0, 5.0), std::numeric_limits<double>::quiet_NaN()};
                for (std::size_t invalidIndex = 0; invalidIndex < invalid_gammas.size(); ++invalidIndex) {
                    auto row = gamma_rows.case_event(routeName + "/invalid-film/" + std::to_string(invalidIndex));
                    const double invalid = invalid_gammas[invalidIndex];
                    host.set_double_value(JuicerParams::kFilmGammaFactor, invalid);
                    host.dispatch_event(effect, JuicerParams::kFilmGammaFactor, ChangeReason::UserEdit);
                    require(JuicerTestSupport::EffectObservation::snapshot(effect).pendingInvalid &&
                                JuicerTestSupport::EffectObservation::admit(effect).kind == FJ_ADMISSION_CAPTURE_FAILED,
                            "invalid authored film gamma admitted an older publication");
                    row.passed();
                }
                host.set_double_value(JuicerParams::kFilmGammaFactor, 1.0);
                for (int policy : {0, 1, 2}) {
                    auto row = gamma_rows.case_event(routeName + "/grain-policy/" + std::to_string(policy));
                    host.set_int_value(JuicerParams::kGrainActive, policy != 0);
                    host.set_int_value(JuicerParams::kGrainSublayersActive, policy != 1);
                    host.set_double_value(JuicerParams::kGrainAmplitude, -1.25);
                    host.set_double_value(JuicerParams::kGrainChroma, 0.375);
                    host.dispatch_event(effect, JuicerParams::kGrainActive, ChangeReason::UserEdit);
                    const auto observed = JuicerTestSupport::EffectObservation::snapshot(effect);
                    const auto& grain = observed.input.grain;
                    require(observed.pendingValid && grain.active == (policy != 0) &&
                                grain.sublayers_active == (policy != 1) &&
                                grain.amplitude == static_cast<float>(std::exp2(-1.25)) &&
                                grain.chroma_mix == 0.375f &&
                                grain.chroma_shared_weight == std::sqrt(0.625f) &&
                                grain.chroma_independent_weight == std::sqrt(0.375f),
                            "OFX grain policy or resolved amount/chroma capture changed");
                    const auto admitted = JuicerTestSupport::EffectObservation::admit(effect);
                    require(bool(admitted.state),
                            "grain capture did not admit a complete state");
                    FjPreparedStateView view{};
                    read_prepared_view(*admitted.state, view);
                    require((view.film_development.density_layers[0][0].count != 0) == (policy == 2) &&
                                (policy == 0 || view.grain.amplitude == grain.amplitude),
                            "development repeated or changed resolved grain policy");
                    row.passed();
                }
            }
            std::cout << "PASS adapter/film-f32-print-f64-grain-capture-and-recovery\n";
            std::ostringstream grainCapture;
            std::vector<std::unique_ptr<JuicerTests::CaseEvent>> grain_events;
            grain_events.reserve(234);

            // Current OFX getters own these resolved controls until S6.B/C.
            std::vector<std::string> grain_capture_intended;
            grain_capture_intended.reserve(198);
            for (const char* route : {"direct", "print"})
                for (unsigned preset = 0; preset < 3; ++preset)
                    for (unsigned policy = 0; policy < 3; ++policy)
                        for (unsigned sample = 0; sample < 11; ++sample)
                            grain_capture_intended.push_back(std::string(route) + "/preset/" + std::to_string(preset) + "/policy/" + std::to_string(policy) + "/sample/" + std::to_string(sample));
            JuicerTests::Cohort grain_capture_rows("grain_capture", grain_capture_intended);
            for (bool print : {false, true}) {
                const std::string routeName = print ? "print" : "direct";
                host.set_str_choice_value(JuicerParams::kParamScanRoute,
                                          Spektrafilm::scan_route_key(print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan));
                for (int presetIndex : {0, 1, 2}) {
                    const std::array<double, 3> presetEv{-1.396, -1.20, -0.57};
                    const std::array<double, 3> presetChroma{0.0, 0.3, 0.5};
                    host.set_int_value(JuicerParams::kGrainPreset, presetIndex);
                    const auto nested = host.nested_event_count();
                    host.dispatch_event(effect, JuicerParams::kGrainPreset, ChangeReason::UserEdit);
                    require(host.nested_event_count() - nested == 22, "each preset suppresses its 22 nested callbacks");
                    for (int policy : {0, 1, 2}) {
                        const std::array<double, 11> amounts{-3.0, 0.0, 3.0, -1.25, 1.234567890123, -9.0, 9.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), std::nextafter(1.234567890123, 2.0)};
                        const double threshold = static_cast<double>(0.999f);
                        const std::array<double, 11> chromas{0.0, 0.375, 1.0, std::nextafter(threshold, 0.0), std::nextafter(threshold, 1.0), static_cast<double>(std::nextafter(0.999f, 0.0f)), static_cast<double>(std::nextafter(0.999f, 1.0f)), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.0, 2.0};
                        for (std::size_t index = 0; index < amounts.size(); ++index) {
                            grain_events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_capture", routeName + "/preset/" + std::to_string(presetIndex) + "/policy/" + std::to_string(policy) + "/sample/" + std::to_string(index)));
                            host.set_int_value(JuicerParams::kGrainActive, policy != 0);
                            host.set_int_value(JuicerParams::kGrainSublayersActive, policy != 1);
                            host.set_double_value(JuicerParams::kGrainAmplitude, amounts[index]);
                            host.set_double_value(JuicerParams::kGrainChroma, chromas[index]);
                            const auto start = host.trace_size();
                            const auto timed = host.at_time_get_count();
                            host.dispatch_event(effect, JuicerParams::kGrainActive, ChangeReason::Time);
                            const auto observed = JuicerTestSupport::EffectObservation::snapshot(effect);
                            const auto& g = observed.input.grain;
                            const double ev = std::isfinite(amounts[index]) ? std::clamp(amounts[index], -3.0, 3.0) : presetEv[presetIndex];
                            const float chroma = static_cast<float>(std::isfinite(chromas[index]) ? std::clamp(chromas[index], 0.0, 1.0) : presetChroma[presetIndex]);
                            require(observed.pendingValid && host.at_time_get_count() == timed &&
                                        g.amplitude == static_cast<float>(std::exp2(ev)) && g.chroma_mix == chroma &&
                                        g.chroma_shared_weight == std::sqrt(std::max(0.0f, 1.0f - chroma)) &&
                                        g.chroma_independent_weight == std::sqrt(std::max(0.0f, chroma)),
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
                            const auto admitted = JuicerTestSupport::EffectObservation::admit(effect);
                            require(bool(admitted.state), "grain capture completes route");
                            FjPreparedStateView view{};
                            read_prepared_view(*admitted.state, view);
                            const auto& recipe = view.grain;
                            const bool hasDensityLayers = view.film_development.density_layers[0][0].count != 0;
                            require(hasDensityLayers == (policy == 2) &&
                                        (policy == 0 || (recipe.amplitude == g.amplitude && recipe.chroma_shared_weight == g.chroma_shared_weight &&
                                                         recipe.chroma_independent_weight == g.chroma_independent_weight)),
                                    "resolved grain is consumed once");
                            grainCapture << "GRAIN_CAPTURE " << print << ' ' << presetIndex << ' ' << policy << ' ' << index
                                         << ' ' << std::bit_cast<std::uint64_t>(amounts[index]) << ' ' << std::bit_cast<std::uint64_t>(chromas[index]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_area_um2);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_scale_cmy[0]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_scale_cmy[1]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_scale_cmy[2]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_scale_layers[0]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_scale_layers[1]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_scale_layers[2]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_density_min_cmy[0]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_density_min_cmy[1]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_density_min_cmy[2]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.uniformity_cmy[0]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.uniformity_cmy[1]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.uniformity_cmy[2]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.correlation_sigma_px);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.dye_cloud_blur_um);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.micro_structure[0]);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.micro_structure[1]);
                            grainCapture << ' ' << g.sublayer_count;
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.amplitude);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.chroma_mix);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.chroma_shared_weight);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.chroma_independent_weight);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.coarse_weight);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.mid_weight);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.size_mix_scale);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.clump_temporal_mix);
                            grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.clump_morph_period_sec);
                            grainCapture << ' ' << g.debug_view;
                            grainCapture << ' ' << recipe.hash << ' ' << recipe.layers_hash << ' ' << view.recipe_hash << '\n';
                        }
                    }
                }
            }
            host.set_int_value(JuicerParams::kGrainActive, 0);
            host.set_double_value(JuicerParams::kGrainAmplitude, -1.25);
            host.set_double_value(JuicerParams::kGrainChroma, 0.375);


            std::vector<std::string> grain_edit_intended;
            grain_edit_intended.reserve(36);
            for (const char* route : {"direct", "print"}) {
                for (int choice : {-8, 0, 1, 2, 8}) {
                    grain_edit_intended.push_back(std::string(route) + "/choice/" + std::to_string(choice));
                    grain_edit_intended.push_back(std::string(route) + "/fallback/" + std::to_string(choice));
                }
                grain_edit_intended.push_back(std::string(route) + "/master-ratios");
                grain_edit_intended.push_back(std::string(route) + "/manual-custom");
                grain_edit_intended.push_back(std::string(route) + "/linked-sharpness-texture");
                grain_edit_intended.push_back(std::string(route) + "/reset-advanced");
                for (unsigned time = 0; time < 4; ++time)
                    grain_edit_intended.push_back(std::string(route) + "/time/" + std::to_string(time));
            }
            JuicerTests::Cohort grain_edit_rows("grain_edit", grain_edit_intended);
            for (bool print : {false, true}) {
                const std::string routeName = print ? "print" : "direct";
                host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan));
                host.set_int_value(JuicerParams::kGrainActive, 1);
                host.set_int_value(JuicerParams::kGrainSublayersActive, 1);
                const auto record = [&](std::string_view label) {
                    const auto observed = JuicerTestSupport::EffectObservation::snapshot(effect);
                    require(observed.pendingValid, "grain edit capture valid");
                    const auto& g = observed.input.grain;
                    const auto admitted = JuicerTestSupport::EffectObservation::admit(effect);
                    require(bool(admitted.state), "grain edit completes publication");
                    FjPreparedStateView view{};
                    read_prepared_view(*admitted.state, view);
                    const auto& recipe = view.grain;
                    grainCapture << "GRAIN_EDIT_CAPTURE " << print << ' ' << label;
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_area_um2);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_scale_cmy[0]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_scale_cmy[1]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_scale_cmy[2]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_scale_layers[0]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_scale_layers[1]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_scale_layers[2]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_density_min_cmy[0]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_density_min_cmy[1]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.particle_density_min_cmy[2]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.uniformity_cmy[0]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.uniformity_cmy[1]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.uniformity_cmy[2]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.correlation_sigma_px);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.dye_cloud_blur_um);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.micro_structure[0]);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.micro_structure[1]);
                    grainCapture << ' ' << g.sublayer_count;
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.amplitude);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.chroma_mix);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.chroma_shared_weight);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.chroma_independent_weight);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.coarse_weight);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.mid_weight);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.size_mix_scale);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.clump_temporal_mix);
                    grainCapture << ' ' << std::bit_cast<std::uint32_t>(g.clump_morph_period_sec);
                    grainCapture << ' ' << g.debug_view;
                    grainCapture << ' ' << recipe.hash << ' ' << recipe.layers_hash << ' ' << view.recipe_hash << '\n';
                    const auto repeated = JuicerTestSupport::EffectObservation::admit(effect);
                    require(repeated.kind == FJ_ADMISSION_READY && repeated.state &&
                                repeated.state->build() == admitted.state->build() &&
                                JuicerTestSupport::PublicationObserver::backing(*repeated.state) == JuicerTestSupport::PublicationObserver::backing(*admitted.state),
                            "same-time capture repeats reuse immutable publication");
                };
                for (int choice : {-8, 0, 1, 2, 8}) {
                    grain_events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", routeName + "/choice/" + std::to_string(choice)));
                    host.set_int_value(JuicerParams::kGrainPreset, choice);
                    const auto before = host.nested_event_count();
                    host.dispatch_event(effect, JuicerParams::kGrainPreset, ChangeReason::UserEdit);
                    require(host.nested_event_count() - before == 22, "invalid choice clamps before preset edits");
                    record("preset-" + std::to_string(choice));
                    const auto unchanged = host.nested_event_count();
                    host.dispatch_event(effect, JuicerParams::kGrainPreset, ChangeReason::Time);
                    host.dispatch_event(effect, JuicerParams::kGrainPreset, ChangeReason::PluginEdit);
                    grain_events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", routeName + "/fallback/" + std::to_string(choice)));
                    require(host.nested_event_count() == unchanged, "non-user preset events do not repeat setters");
                    const double nan = std::numeric_limits<double>::quiet_NaN();
                    for (const char* name : {JuicerParams::kGrainAmplitude, JuicerParams::kGrainBlur, JuicerParams::kGrainSharpness, JuicerParams::kGrainChroma, JuicerParams::kGrainTexture, JuicerParams::kGrainParticleAreaUm2, JuicerParams::kGrainSizeMixScale, JuicerParams::kGrainSizeMixWeight, JuicerParams::kGrainSizeMixWeightMid, JuicerParams::kGrainBlurDyeCloudsUm, JuicerParams::kGrainClumpTemporalMix, JuicerParams::kGrainClumpMorphPeriodSec})
                        host.set_double_value(name, nan);
                    for (const char* name : {JuicerParams::kGrainParticleScale, JuicerParams::kGrainParticleScaleLayers, JuicerParams::kGrainDensityMin, JuicerParams::kGrainUniformity})
                        host.set_double3_value(name, {nan, nan, nan});
                    host.set_double2_value(JuicerParams::kGrainMicroStructure, {nan, nan});
                    host.set_int_value(JuicerParams::kGrainDebugView, choice);
                    host.dispatch_event(effect, JuicerParams::kGrainActive, ChangeReason::Time);
                    const auto fallback = JuicerTestSupport::EffectObservation::snapshot(effect).input.grain;
                    require(same_float3(fallback.particle_scale_cmy, {1.48f, 1.48f, 1.48f}), "triplet fallback repeats preset master without ratios");
                    record("fallback-" + std::to_string(choice));
                }
                grain_events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", routeName + "/master-ratios"));
                host.set_int_value(JuicerParams::kGrainPreset, 1);
                host.dispatch_event(effect, JuicerParams::kGrainPreset, ChangeReason::UserEdit);
                host.set_double3_value(JuicerParams::kGrainParticleScale, {0.5, 1.0, 1.5});
                host.set_double_value(JuicerParams::kGrainParticleScaleMaster, 1.25);
                host.dispatch_event(effect, JuicerParams::kGrainParticleScaleMaster, ChangeReason::UserEdit);
                require(same_float3(JuicerTestSupport::EffectObservation::snapshot(effect).input.grain.particle_scale_cmy, {0.625f, 1.25f, 1.875f}), "master edit writes normalized current ratios once");
                record("master-ratios");
                grain_events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", routeName + "/manual-custom"));
                host.set_double3_value(JuicerParams::kGrainParticleScale, {2.0, 3.0, 4.0});
                host.dispatch_event(effect, JuicerParams::kGrainParticleScale, ChangeReason::UserEdit);
                require(same_float3(JuicerTestSupport::EffectObservation::snapshot(effect).input.grain.particle_scale_cmy, {2.0f, 3.0f, 4.0f}), "manual triplets are captured without multiplying master");
                record("manual-custom");
                grain_events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", routeName + "/linked-sharpness-texture"));
                host.set_double_value(JuicerParams::kGrainSharpness, 0.25);
                host.dispatch_event(effect, JuicerParams::kGrainSharpness, ChangeReason::UserEdit);
                host.set_double_value(JuicerParams::kGrainTexture, 0.75);
                host.dispatch_event(effect, JuicerParams::kGrainTexture, ChangeReason::UserEdit);
                record("linked-sharpness-texture");
                grain_events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", routeName + "/reset-advanced"));
                host.dispatch_event(effect, JuicerParams::kGrainResetAdvanced, ChangeReason::UserEdit);
                record("reset-advanced");
                host.set_int_value(JuicerParams::kGrainActive, 0);
                const std::array<double, 4> grain_times{2.0, 0.0, 2.0, 2.0};
                for (std::size_t timeIndex = 0; timeIndex < grain_times.size(); ++timeIndex) {
                    grain_events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", routeName + "/time/" + std::to_string(timeIndex)));
                    const double time = grain_times[timeIndex];
                    host.set_time(time);
                    host.dispatch_event(effect, JuicerParams::kGrainActive, ChangeReason::Time);
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
                std::vector<std::string> originalWords;
                originalWords.reserve(234);
                for (std::string word; expected >> word;)
                    originalWords.push_back(word);
                const auto selectedWords = PrintIlluminantFixtures::apply(expectedPath, originalWords);
                std::string actualWord;
                for (const auto& word : selectedWords)
                    require(bool(actual >> actualWord) && word.get<std::string>() == actualWord, "exact native authored/captured/recipe/final grain words");
                require(!(actual >> actualWord), "no extra grain capture records");
            }
            require(grain_events.size() == 234, "complete grain case event membership");
            for (auto& event : grain_events)
                event->passed();
            std::cout << grainWords;

            std::vector<std::string> dir_intended;
            std::vector<std::string> scanner_intended;
            dir_intended.reserve(4);
            scanner_intended.reserve(20);
            for (const char* route : {"negative-direct", "negative-print", "positive-direct", "positive-print"}) {
                dir_intended.emplace_back(route);
                for (unsigned flags = 0; flags < 4; ++flags)
                    scanner_intended.push_back(std::string(route) + "/flags/" + std::to_string(flags));
                scanner_intended.push_back(std::string(route) + "/sanitation-and-recovery");
            }
            JuicerTests::Cohort dir_rows("dir_capture", dir_intended);
            JuicerTests::Cohort scanner_rows("scanner_capture", scanner_intended);
            for (bool positive : {false, true})
                for (bool print : {false, true}) {
                    const std::string routeName = std::string(positive ? "positive-" : "negative-") + (print ? "print" : "direct");
                    auto dir_row = dir_rows.case_event(routeName);
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
                        host.dispatch_event(effect, JuicerParams::kDirCouplersActive, ChangeReason::UserEdit);
                        return JuicerTestSupport::EffectObservation::snapshot(effect);
                    };
                    host.set_int_value(JuicerParams::kDirCouplersActive, 0);
                    host.set_int_value(JuicerParams::kDirCouplersGammaUseStock, 1);
                    host.set_double_value(JuicerParams::kDirCouplersAmount, std::numeric_limits<double>::quiet_NaN());
                    auto start = host.trace_size();
                    auto observed = capture();
                    require(observed.pendingValid && !observed.input.dir_couplers.active && gets(start, JuicerParams::kDirCouplersAmount) == 0, "disabled OFX DIR skips amount capture");
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
                    require(observed.input.dir_couplers.amount == static_cast<float>(authored), "stock DIR amount narrowing");
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
                    const auto& dir = observed.input.dir_couplers;
                    require(observed.pendingValid && dir.gamma_same_layer_rgb[0] == static_cast<float>(authored) && dir.gamma_interlayer_r_to_gb[1] == static_cast<float>(0.234567890123) && gets(start, JuicerParams::kDirCouplersDiffusionTailUm) == 0, "custom RGB f64-to-f32 narrowing and zero-tail bypass");
                    host.set_double3_value(consumed, {0.09, 1, 1});
                    observed = capture();
                    require(observed.pendingInvalid, "invalid consumed K blocks current snapshot");
                    host.set_double3_value(consumed, {authored, authored, authored});
                    observed = capture();
                    require(observed.pendingValid, "OFX DIR controls recover");
                    const auto admitted = JuicerTestSupport::EffectObservation::admit(effect);
                    require(bool(admitted.state), "captured DIR controls admit complete route");
                    host.set_int_value(JuicerParams::kDirCouplersActive, 0);
                    host.set_int_value(JuicerParams::kDirCouplersGammaUseStock, 1);
                    host.set_double3_value(JuicerParams::kDirCouplersLangmuirDonorKRgb, {1, 1, 1});
                    host.set_double3_value(JuicerParams::kDirCouplersLangmuirReceiverKRgb, {1, 1, 1});
                    host.set_double_value(JuicerParams::kDirCouplersDiffusionTailWeight, 0.03);
                    host.set_double_value(JuicerParams::kDirCouplersDiffusionTailUm, 200);
                    dir_row.passed();
                }
            std::cout << "PASS adapter/DIR-conditional-capture-narrowing-polarity-and-recovery\n";

            for (bool positive : {false, true})
                for (bool print : {false, true}) {
                    const std::string routeName = std::string(positive ? "positive-" : "negative-") + (print ? "print" : "direct");
                    host.set_str_choice_value(JuicerParams::kFilmProfileKey, positive ? "fujifilm_provia_100f" : "kodak_portra_400");
                    host.set_str_choice_value(JuicerParams::kParamScanRoute, Spektrafilm::scan_route_key(print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan));
                    host.set_double_value(JuicerParams::kFilmGammaFactor, 1.0);
                    host.set_double_value(JuicerParams::kPrintGammaFactor, 1.0);
                    const auto capture = [&](bool expectPublication = true) {
                        const auto start = host.trace_size();
                        host.dispatch_event(effect, JuicerParams::kScannerBlackCorrection, ChangeReason::UserEdit);
                        for (const char* name : {JuicerParams::kScannerBlackCorrection, JuicerParams::kScannerWhiteCorrection, JuicerParams::kScannerBlackLevel, JuicerParams::kScannerWhiteLevel}) {
                            const std::string prefix = "GET_CURRENT name=" + std::string(name) + " ";
                            require(std::count_if(host.trace().begin() + static_cast<std::ptrdiff_t>(start), host.trace().end(), [&](const std::string& row) {
                                        return row.starts_with(prefix);
                                    }) == 1,
                                    "scanner control uses exactly one current-value getter");
                        }
                        const auto admitted = JuicerTestSupport::EffectObservation::admit(effect);
                        if (expectPublication) {
                            require(bool(admitted.state),
                                    "scanner controls admit complete route: positive=" + std::to_string(positive) +
                                        " print=" + std::to_string(print) + " status=" +
                                        std::to_string(static_cast<unsigned>(admitted.kind)) + " diagnostic=" + admitted.diagnostic);
                        } else {
                            require(admitted.kind == FJ_ADMISSION_PREPARATION_FAILED &&
                                        !admitted.state,
                                    "scanner constructor failure blocks route admission: positive=" +
                                        std::to_string(positive) + " print=" + std::to_string(print) +
                                        " status=" + std::to_string(static_cast<unsigned>(admitted.kind)) +
                                        " diagnostic=" + admitted.diagnostic);
                        }
                        return admitted;
                    };
                    const auto output = [&](const auto& admitted) {
                        FjPreparedStateView view{};
                        read_prepared_view(*admitted.state, view);
                        return view.scanner_policy;
                    };
                    constexpr double black = 0.01234567890123;
                    constexpr double white = 0.98765432109876;
                    for (unsigned flags = 0; flags < 4; ++flags) {
                        auto flags_row = scanner_rows.case_event(routeName + "/flags/" + std::to_string(flags));
                        host.set_int_value(JuicerParams::kScannerBlackCorrection, int(flags & 1u));
                        host.set_int_value(JuicerParams::kScannerWhiteCorrection, int((flags >> 1u) & 1u));
                        host.set_double_value(JuicerParams::kScannerBlackLevel, black);
                        host.set_double_value(JuicerParams::kScannerWhiteLevel, white);
                        const auto admitted = capture();
                        const auto& scanner = output(admitted);
                        require(scanner.black_correction == std::uint32_t(flags & 1u) && scanner.white_correction == std::uint32_t((flags >> 1u) & 1u) &&
                                    scanner.black_level == static_cast<float>(black) && scanner.white_level == static_cast<float>(white),
                                "scanner bools and f64-to-f32 narrowing survive actual OFX capture");
                        flags_row.passed();
                    }
                    auto recovery_row = scanner_rows.case_event(routeName + "/sanitation-and-recovery");
                    const auto held = capture();
                    const auto heldHash = prepared_recipe_hash(*held.state);
                    const auto active = host.render(effect, frame, "scanner-active");
                    require(active.succeeded, "active scanner correction renders for selected polarity/route");

                    host.set_double_value(JuicerParams::kScannerBlackLevel, -1.0);
                    host.set_double_value(JuicerParams::kScannerWhiteLevel, 2.0);
                    const auto clamped = capture();
                    require(output(clamped).black_level == 0.0f && output(clamped).white_level == 1.0f, "scanner encoded levels clamp to unit range");
                    host.set_double_value(JuicerParams::kScannerBlackLevel, std::numeric_limits<double>::quiet_NaN());
                    host.set_double_value(JuicerParams::kScannerWhiteLevel, std::numeric_limits<double>::infinity());
                    const auto sanitized = capture();
                    require(output(sanitized).black_level == 0.01f && output(sanitized).white_level == 0.98f, "scanner nonfinite levels use control defaults");
                    const auto beforeFailure = JuicerTestSupport::EffectObservation::snapshot(effect);

                    if (positive || print) {
                        require((print ? beforeFailure.printPublished : beforeFailure.directPublished) &&
                                    !(print ? beforeFailure.directPublished : beforeFailure.printPublished),
                                "scanner route has one selected publication before constructor failure");
                        host.set_double_value(JuicerParams::kScannerBlackLevel, 0.0);
                        host.set_double_value(JuicerParams::kScannerWhiteLevel, 0.0);
                        const char* const expectedDiagnostic = print
                                                                   ? "MalformedRequiredProfileData phase=8B print corrected midgray"
                                                                   : "MalformedRequiredProfileData phase=8B direct corrected midgray";
                        const auto failedAdmission = capture(false);
                        const auto failedSnapshot = JuicerTestSupport::EffectObservation::captured(effect);
                        require(failedAdmission.diagnostic == expectedDiagnostic,
                                "scanner constructor failure preserves owning corrected-midgray diagnostic: " +
                                    failedAdmission.diagnostic);

                        std::optional<JuicerTestSupport::PreparedStateHold> failedConstructor;
                        std::array<char, 4096> constructorText{};
                        FjErrorBuffer constructorError{
                            constructorText.data(), constructorText.size(), 0};
                        const FjStatus constructorStatus = prepare_state(
                            runtime.assets,
                            failedSnapshot,
                            failedConstructor,
                            constructorError);
                        const std::string constructorDiagnostic(
                            constructorText.data(),
                            std::min(constructorError.length, constructorText.size()));
                        require(constructorStatus.category == FJ_STATUS_PREPARATION_FAILURE &&
                                    constructorStatus.api == FJ_API_NONE && constructorStatus.native_code == 0 &&
                                    !failedConstructor && constructorDiagnostic == expectedDiagnostic,
                                "scanner complete constructor reports the typed preparation failure: " +
                                    constructorDiagnostic);

                        const auto failedState = JuicerTestSupport::EffectObservation::snapshot(effect);
                        require(failedState.pendingValid &&
                                    failedState.pendingFullHash == JuicerTestSupport::hash_input_identity(failedSnapshot) &&
                                    failedState.pendingFullHash != beforeFailure.pendingFullHash &&
                                    failedState.lastParamHash == beforeFailure.lastParamHash &&
                                    failedState.buildCounter == beforeFailure.buildCounter &&
                                    failedState.directPublished == beforeFailure.directPublished && failedState.printPublished == beforeFailure.printPublished,
                                "failed scanner construction keeps controls valid, retains the old publication privately, and does not publish a new hash or counter");
                        require(output(held).black_level == static_cast<float>(black) &&
                                    output(held).white_level == static_cast<float>(white) &&
                                    (prepared_recipe_hash(*held.state)) == heldHash,
                                "held immutable scanner recipe survives constructor failure");

                        const auto failed = host.render(effect, frame, "scanner-constructor-midgray-failure", true);
                        require(failed.rejected && failed.state.pendingValid && failed.releasedImages == 2,
                                "OFX admission rejects the current constructor failure and releases both image leases");
                        require(failed.state.pendingFullHash == failedState.pendingFullHash &&
                                    failed.state.lastParamHash == beforeFailure.lastParamHash &&
                                    failed.state.buildCounter == beforeFailure.buildCounter &&
                                    failed.state.directPublished == beforeFailure.directPublished && failed.state.printPublished == beforeFailure.printPublished &&
                                    failed.state.latchValid == beforeFailure.latchValid &&
                                    failed.state.latchSnapshotId == beforeFailure.latchSnapshotId &&
                                    failed.state.latchFrameToken == beforeFailure.latchFrameToken,
                                "failed OFX admission does not select an older publication or alter the submission latch");
                        require(output(held).black_level == static_cast<float>(black) && output(held).white_level == static_cast<float>(white) &&
                                    (prepared_recipe_hash(*held.state)) == heldHash,
                                "held immutable scanner recipe remains readable after failed OFX admission");
                    }
                    host.set_double_value(JuicerParams::kScannerBlackLevel, black);
                    host.set_double_value(JuicerParams::kScannerWhiteLevel, white);
                    const auto restored = capture();
                    const auto recoveredScanner = host.render(effect, frame, "scanner-recovery");
                    require(recoveredScanner.succeeded && recoveredScanner.outputSignature == active.outputSignature &&
                                (prepared_recipe_hash(*restored.state)) == heldHash,
                            "scanner valid recovery restores recipe and output identity");
                    if (positive || print) {
                        const auto recoveredState = JuicerTestSupport::EffectObservation::snapshot(effect);
                        require(recoveredState.pendingValid &&
                                    (print ? recoveredState.printPublished : recoveredState.directPublished) &&
                                    recoveredState.lastParamHash == recoveredState.pendingFullHash &&
                                    recoveredState.buildCounter == beforeFailure.buildCounter + 1,
                                "scanner constructor failure recovers through one new complete publication");
                    }
                    recovery_row.passed();
                }
            std::cout << "PASS adapter/scanner-current-capture-flags-sanitation-narrowing-active-routes-and-frame-recovery\n";

            require(host.current_get_count() != 0 &&
                        host.at_time_get_count() == 0,
                    "adapter getter selection was not current-value-only");
        }
        require(loaded.close() == kOfxStatOK, "adapter terminal unload failed");
        host.write_trace(tracePath);
        operation.passed();
        case_log.finish();
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
        return 1;
    } catch (...) {
        std::cerr << "FAIL adapter trace: unknown exception\n";
        return 1;
    }
} catch (const std::exception& error) {
    std::cerr << "FAIL adapter trace setup: " << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "FAIL adapter trace setup: unknown exception\n";
    return 1;
}
