#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <dlfcn.h>
#endif

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxParam.h"
#include "ofxParamExt.h"
#include "production_images.h"
#include "observation.h"

namespace {
    using Json = nlohmann::json;
    using Clock = std::chrono::steady_clock;
    using ofx_test::InstanceHost;
    constexpr float kDestinationCanary = -8.0f;

    void require(bool condition, std::string_view message) {
        if (!condition)
            throw std::runtime_error(std::string(message));
    }
    void cuda_check(cudaError_t status, const char* operation) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
    void driver_check(CUresult status, const char* operation) {
        require(status == CUDA_SUCCESS, operation);
    }
    std::uint64_t nanoseconds(Clock::time_point before, Clock::time_point after) {
        const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        require(count >= 0, "negative steady-clock interval");
        return static_cast<std::uint64_t>(count);
    }
    template <class F>
    void timed(Json& destination, F&& operation) {
        const auto start = Clock::now();
        try {
            operation();
        } catch (...) {
            destination = nanoseconds(start, Clock::now());
            throw;
        }
        destination = nanoseconds(start, Clock::now());
    }
    const char* parameter_type(ofx_test::InstanceParamKind kind) {
        using ofx_test::InstanceParamKind;
        switch (kind) {
            case InstanceParamKind::Double:
                return kOfxParamTypeDouble;
            case InstanceParamKind::Boolean:
                return kOfxParamTypeBoolean;
            case InstanceParamKind::Choice:
                return kOfxParamTypeChoice;
            case InstanceParamKind::StringChoice:
                return kOfxParamTypeStrChoice;
            case InstanceParamKind::Integer:
                return kOfxParamTypeInteger;
            case InstanceParamKind::Double2D:
                return kOfxParamTypeDouble2D;
            case InstanceParamKind::Double3D:
                return kOfxParamTypeDouble3D;
            case InstanceParamKind::PushButton:
                return kOfxParamTypePushButton;
        }
        throw std::runtime_error("unknown consumed parameter kind");
    }
    class Module final {
    public:
        explicit Module(const std::filesystem::path& path) {
#ifdef _WIN32
            handle_ = LoadLibraryW(path.c_str());
#else
            handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
            require(handle_ != nullptr, "cannot load supplied OFX module");
        }
        ~Module() {
            if (handle_)
                close_unchecked();
        }
        Module(const Module&) = delete;
        Module& operator=(const Module&) = delete;
        template <class T>
        T symbol(const char* name) const {
#ifdef _WIN32
            const auto address = GetProcAddress(handle_, name);
#else
            auto* address = dlsym(handle_, name);
#endif
            require(address != nullptr, std::string("missing module symbol: ") + name);
            return reinterpret_cast<T>(address);
        }
        void close() {
#ifdef _WIN32
            const bool ok = FreeLibrary(handle_) != 0;
#else
            const bool ok = dlclose(handle_) == 0;
#endif
            if (ok)
                handle_ = nullptr;
            require(ok, "module library closure failed");
        }

    private:
        void close_unchecked() noexcept {
#ifdef _WIN32
            (void)FreeLibrary(handle_);
#else
            (void)dlclose(handle_);
#endif
        }
#ifdef _WIN32
        HMODULE handle_ = nullptr;
#else
        void* handle_ = nullptr;
#endif
    };
    struct ContextRestore final {
        CUcontext original = nullptr;
        ~ContextRestore() {
            (void)cuCtxSetCurrent(original);
        }
    };
    struct Buffers final : ofx_test::ImageBuffers {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        void close() {
            cuda_check(cudaStreamSynchronize(stream), "drain driver stream");
            cuda_check(cudaEventDestroy(stop), "destroy stop event");
            stop = nullptr;
            cuda_check(cudaEventDestroy(start), "destroy start event");
            start = nullptr;
            cuda_check(cudaFree(destination), "free driver destination");
            destination = nullptr;
            cuda_check(cudaFree(source), "free driver source");
            source = nullptr;
            cuda_check(cudaStreamDestroy(stream), "destroy driver stream");
            stream = nullptr;
        }
        ~Buffers() {
            if (stream)
                (void)cudaStreamSynchronize(stream);
            if (stop)
                (void)cudaEventDestroy(stop);
            if (start)
                (void)cudaEventDestroy(start);
            if (destination)
                (void)cudaFree(destination);
            if (source)
                (void)cudaFree(source);
            if (stream)
                (void)cudaStreamDestroy(stream);
        }
    };
    class Lifecycle final {
    public:
        Lifecycle(OfxPlugin* plugin, InstanceHost& host) : plugin_(plugin), host_(host) {}
        ~Lifecycle() {
            if (effect_)
                (void)call(kOfxActionDestroyInstance, effect_);
            if (loaded_)
                (void)call(kOfxActionUnload);
        }
        OfxStatus call(const char* action, const void* effect = nullptr, OfxPropertySetHandle args = nullptr) const {
            return plugin_->mainEntry(action, effect, args, nullptr);
        }
        OfxStatus load() {
            plugin_->setHost(host_.binding());
            const auto status = call(kOfxActionLoad);
            loaded_ = status == kOfxStatOK;
            return status;
        }
        OfxStatus create(OfxImageEffectHandle effect) {
            const auto status = call(kOfxActionCreateInstance, effect);
            if (status == kOfxStatOK)
                effect_ = effect;
            return status;
        }
        OfxStatus destroy() {
            const auto status = call(kOfxActionDestroyInstance, effect_);
            effect_ = nullptr;
            return status;
        }
        OfxStatus unload() {
            require(effect_ == nullptr, "Unload before DestroyInstance");
            const auto status = call(kOfxActionUnload);
            loaded_ = false;
            return status;
        }

    private:
        OfxPlugin* plugin_;
        InstanceHost& host_;
        OfxImageEffectHandle effect_ = nullptr;
        bool loaded_ = false;
    };
    struct Observation final {
        using Entry = FjMeasurementStatus (*)(FjMeasurementRecord*);
        Entry begin = nullptr;
        Entry read = nullptr;
        Entry end = nullptr;
        bool active = false;
        FjMeasurementStatus status = FJ_MEASUREMENT_OK;
        FjMeasurementRecord last{};
        explicit Observation(const Module& module, bool enabled) {
            last.version = FJ_MEASUREMENT_VERSION;
            last.size = sizeof(last);
            if (enabled) {
                begin = module.symbol<Entry>("fj_test_ofx_measurement_begin");
                read = module.symbol<Entry>("fj_test_ofx_measurement_read");
                end = module.symbol<Entry>("fj_test_ofx_measurement_end");
                require(begin(&last) == FJ_MEASUREMENT_OK, "observation begin failed");
                active = true;
            }
        }
        ~Observation() {
            if (active)
                (void)end(&last);
        }
        FjMeasurementRecord snapshot() {
            if (read)
                status = read(&last);
            return last;
        }
        void finish() {
            if (active) {
                const auto status = end(&last);
                active = false;
                require(status == FJ_MEASUREMENT_OK, "observation end failed");
                require(read(&last) == FJ_MEASUREMENT_OK, "final observation unreadable");
            }
        }
    };
    Json observation_json(const FjMeasurementRecord& r) {
        return {{"device_id", r.device_id}, {"context_opaque", r.context_opaque}, {"context_epoch", r.context_epoch}, {"construction_starts", r.construction_starts}, {"construction_pairs", r.construction_completed}, {"construction_incomplete", r.construction_incomplete}, {"construction_ns", r.construction_nanoseconds}, {"device_reserved_bytes", r.device_reserved_bytes}, {"device_committed_bytes", r.device_committed_bytes}, {"device_retiring_bytes", r.device_retiring_bytes}, {"device_charged_bytes", r.device_charged_bytes}, {"device_records", r.device_record_count}, {"device_peak_reserved_bytes", r.device_peak_reserved_bytes}, {"device_peak_committed_bytes", r.device_peak_committed_bytes}, {"device_peak_retiring_bytes", r.device_peak_retiring_bytes}, {"device_peak_charged_bytes", r.device_peak_charged_bytes}, {"device_peak_records", r.device_peak_record_count}, {"pinned_upload_pool_physical_bytes", r.upload_live_bytes}, {"pinned_upload_pool_peak_bytes", r.upload_peak_bytes}, {"pinned_upload_pool_state_bytes", r.upload_state_bytes}, {"pinned_upload_pool_state_counts", r.upload_state_counts}};
    }
    Json process_memory() {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX counters{};
        counters.cb = sizeof(counters);
        require(GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)) != 0,
                "process working-set reading failed");
        return {{"rss_bytes", counters.WorkingSetSize}, {"peak_rss_bytes", counters.PeakWorkingSetSize}, {"private_bytes", counters.PrivateUsage}};
#else
        std::ifstream input("/proc/self/status");
        require(static_cast<bool>(input), "process RSS reading unavailable");
        std::optional<std::uint64_t> rss, peak;
        for (std::string line; std::getline(input, line);) {
            auto field = [&](std::string_view name, std::optional<std::uint64_t>& value) {
                if (line.starts_with(name)) {
                    const auto number = line.find_first_of("0123456789");
                    require(number != std::string::npos && line.ends_with(" kB"), "malformed process memory field");
                    const auto kib = std::stoull(line.substr(number));
                    require(kib <= std::numeric_limits<std::uint64_t>::max() / 1024, "process memory overflow");
                    value = kib * 1024;
                }
            };
            field("VmRSS:", rss);
            field("VmHWM:", peak);
        }
        require(rss.has_value() && peak.has_value(), "required process memory field missing");
        return {{"rss_bytes", *rss}, {"peak_rss_bytes", *peak}, {"private_bytes", nullptr}};
#endif
    }
    Json control_map(InstanceHost& host, OfxImageEffectHandle effect) {
        Json controls = Json::object();
        for (const auto& definition : ofx_test::kInstanceParameters) {
            if (definition.kind == ofx_test::InstanceParamKind::PushButton)
                continue;
            Json values = Json::array();
            for (const auto& value : host.value(effect, definition.name)) {
                std::visit([&](const auto& v) {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (!std::is_same_v<T, void*>)
                        values.push_back(v);
                    else
                        throw std::runtime_error("pointer-valued photographic control");
                },
                           value);
            }
            require(!values.empty(), "effective control value missing");
            controls[definition.name] = std::move(values);
        }
        return controls;
    }
    struct RenderStep {
        const char* phase = nullptr;
        int cycle = 0;
        int index = 0;
        bool expect_construction = false;
        bool readback = false;
    };
    struct Sample {
        const char* phase = nullptr;
        int cycle = 0;
        int index = 0;
        OfxStatus status = kOfxStatFailed;
        std::uint64_t cpuNs = 0;
        float streamMs = 0.0f;
        bool streamTimed = false;
        FjMeasurementRecord before{};
        FjMeasurementRecord after{};
        FjMeasurementStatus beforeStatus = FJ_MEASUREMENT_OK;
        FjMeasurementStatus afterStatus = FJ_MEASUREMENT_OK;
        bool checked = false;
        bool outputValid = false;
        bool releasesValid = false;
        bool contextPreserved = false;
        bool expectedConstruction = false;
    };
    Json sample_json(const Sample& sample, bool observed) {
        const bool monotonic = sample.after.construction_starts >= sample.before.construction_starts &&
                               sample.after.construction_completed >= sample.before.construction_completed &&
                               sample.after.construction_incomplete >= sample.before.construction_incomplete &&
                               sample.after.construction_nanoseconds >= sample.before.construction_nanoseconds;
        Json construction{{"starts", nullptr}, {"pairs", nullptr}, {"incomplete", nullptr}, {"duration_ns", nullptr}, {"observed_duration_ns", nullptr}, {"outcome", observed ? "invalid" : "unobserved"}, {"before_status", sample.beforeStatus}, {"after_status", sample.afterStatus}};
        if (monotonic) {
            construction["starts"] = sample.after.construction_starts - sample.before.construction_starts;
            construction["pairs"] = sample.after.construction_completed - sample.before.construction_completed;
            construction["incomplete"] = sample.after.construction_incomplete - sample.before.construction_incomplete;
            if (observed && sample.after.construction_completed > sample.before.construction_completed)
                construction["observed_duration_ns"] = sample.after.construction_nanoseconds - sample.before.construction_nanoseconds;
        }
        if (observed) {
            using JuicerMeasurement::ConstructionOutcome;
            const auto outcome = JuicerMeasurement::classify_construction(sample.before, sample.beforeStatus, sample.after, sample.afterStatus, {.expected = sample.expectedConstruction, .callback_succeeded = sample.status == kOfxStatOK, .output_valid = !sample.checked || sample.outputValid, .lifecycle_valid = sample.releasesValid && sample.contextPreserved});
            switch (outcome) {
                case ConstructionOutcome::Constructed:
                    construction["outcome"] = "success";
                    construction["duration_ns"] = construction["observed_duration_ns"];
                    break;
                case ConstructionOutcome::Reused:
                    construction["outcome"] = "reused";
                    break;
                case ConstructionOutcome::Unknown:
                    construction["outcome"] = "unknown";
                    break;
                case ConstructionOutcome::Invalid:
                    construction["outcome"] = "invalid";
                    break;
            }
        }
        return {{"phase", sample.phase}, {"cycle", sample.cycle}, {"index", sample.index}, {"callback_status", sample.status}, {"callback_ns", sample.cpuNs}, {"stream_interval_ms", sample.streamTimed ? Json(sample.streamMs) : Json(nullptr)}, {"construction", construction}, {"output_checked", sample.checked}, {"output_valid", sample.checked ? Json(sample.outputValid) : Json(nullptr)}, {"releases_valid", sample.releasesValid}, {"context_preserved", sample.contextPreserved}};
    }
    bool output_valid(const std::vector<float>& source, const std::vector<float>& initial, const std::vector<float>& output, const ofx_test::Case& test) {
        if (!ofx_test::canaries_preserved(initial, output, test) || !ofx_test::alpha_preserved(source, output, test))
            return false;
        const auto pitch = test.width * test.components + ofx_test::kProductionPitchPadding;
        for (int y = 0; y < test.height; ++y)
            for (int x = 0; x < test.width * test.components; ++x)
                if (!std::isfinite(output[static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) + static_cast<std::size_t>(x)]) ||
                    (x % test.components < 3 && output[static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) + static_cast<std::size_t>(x)] == kDestinationCanary))
                    return false;
        return true;
    }
    void verify_output_guard() {
        for (const int components : {3, 4}) {
            const ofx_test::Case test{"output-guard", "negative-direct", "NegativeDirectScan", 1, components, 2, 1, 0, 0.0};
            const auto source = ofx_test::initial_source(test);
            const std::vector<float> initial(source.size(), kDestinationCanary);
            auto output = initial;
            std::copy_n(source.begin(), test.width * components, output.begin());
            require(output_valid(source, initial, output, test), "output guard rejected written RGB");
            // A successful callback must write every RGB channel, including RGBA
            // renders that copy alpha while leaving RGB untouched.
            for (int pixel = 0; pixel < test.width; ++pixel) {
                for (int channel = 0; channel < 3; ++channel) {
                    const auto index = static_cast<std::size_t>(pixel) * static_cast<std::size_t>(components) + static_cast<std::size_t>(channel);
                    const auto written = output[index];
                    output[index] = kDestinationCanary;
                    require(!output_valid(source, initial, output, test), "output guard accepted an unwritten RGB channel");
                    output[index] = written;
                }
            }
            for (int pixel = 0; pixel < test.width; ++pixel)
                for (int channel = 0; channel < 3; ++channel)
                    output[static_cast<std::size_t>(pixel) * static_cast<std::size_t>(components) + static_cast<std::size_t>(channel)] = kDestinationCanary;
            require(!output_valid(source, initial, output, test), "output guard accepted untouched RGB");
        }
    }
    struct Arguments {
        std::filesystem::path module, resources, output;
        std::string mode, lane, workload;
    };
    Arguments arguments(int argc, char** argv) {
        require(argc == 13, "expected --mode --module --resources --lane --workload --output");
        Arguments args;
        for (int i = 1; i < argc; i += 2) {
            const std::string_view key(argv[i]);
            const std::string value(argv[i + 1]);
            if (key == "--mode")
                args.mode = value;
            else if (key == "--lane")
                args.lane = value;
            else if (key == "--workload")
                args.workload = value;
            else if (key == "--module")
                args.module = std::filesystem::absolute(value);
            else if (key == "--resources")
                args.resources = std::filesystem::absolute(value);
            else if (key == "--output")
                args.output = value;
            else
                throw std::runtime_error("unknown measurement argument");
        }
        require((args.mode == "check" || args.mode == "measure") && (args.lane == "shipping" || args.lane == "observed"), "invalid mode or lane");
        require(std::filesystem::is_regular_file(args.module) && std::filesystem::is_directory(args.resources) && !args.output.empty(), "missing measurement input/output");
        return args;
    }
    void run(const Arguments& args, Json& report) {
        verify_output_guard();
        const bool observed = args.lane == "observed";
        const bool check = args.mode == "check";
        const bool print = args.workload == "negative-print" || args.workload == "negative-print-grain";
        const bool grain = args.workload == "negative-direct-grain" || args.workload == "negative-print-grain";
        require(print || grain || args.workload == "negative-direct", "unknown workload");
        const ofx_test::Case test{args.workload.c_str(), print ? "negative-print" : "negative-direct", print ? "NegativePrintScan" : "NegativeDirectScan", print ? 1 : 0, print ? 4 : 3, check ? 7 : 6048, check ? 5 : 4032, grain ? 1 : 0, 0.0};
        report = {{"schema", 1}, {"status", "failed"}, {"mode", args.mode}, {"lane", args.lane}, {"workload", args.workload}, {"width", test.width}, {"height", test.height}, {"renders", Json::array()}, {"memory", Json::array()}, {"edits", Json::array()}, {"error", nullptr}, {"drained", nullptr}};
        const auto effectiveResources = std::filesystem::canonical(args.module.parent_path().parent_path() / "Resources");
        report["resources"] = effectiveResources.string();
        require(std::filesystem::equivalent(effectiveResources, args.resources), "declared Resources differs from module-relative Resources");
        report["output_guard_version"] = 2;
        report["upload_pool_state_order"] = {"available", "reserved", "in_flight", "quarantined"};
        report["timings"] = {{"create_ns", Json::array()}, {"author_ns", Json::array()}, {"destroy_ns", Json::array()}};
        auto& timings = report["timings"];
        report["lifecycle"] = {{"module_load_status", nullptr}, {"module_unload_status", nullptr}, {"create_statuses", Json::array()}, {"destroy_statuses", Json::array()}, {"effects_created", 0}, {"effects_destroyed", 0}, {"context_restored", false}, {"module_closed", false}, {"callback_error", false}};
        auto& life = report["lifecycle"];
        driver_check(cuInit(0), "CUDA driver initialization failed");
        ContextRestore restore;
        driver_check(cuCtxGetCurrent(&restore.original), "read original context failed");
        cuda_check(cudaSetDevice(0), "select CUDA device");
        Buffers device;
        cuda_check(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create supplied stream");
        cuda_check(cudaEventCreate(&device.start), "create start event");
        cuda_check(cudaEventCreate(&device.stop), "create stop event");
        const auto source = ofx_test::initial_source(test);
        const std::vector<float> initial(source.size(), kDestinationCanary);
        std::vector<float> output(source.size());
        const auto bytes = source.size() * sizeof(float);
        cuda_check(cudaMalloc(&device.source, bytes), "allocate source");
        cuda_check(cudaMalloc(&device.destination, bytes), "allocate destination");
        cuda_check(cudaMemcpyAsync(device.source, source.data(), bytes, cudaMemcpyHostToDevice, device.stream), "upload fixed source");
        cuda_check(cudaStreamSynchronize(device.stream), "finish source upload");
        cudaDeviceProp properties{};
        int driverVersion = 0;
        cuda_check(cudaGetDeviceProperties(&properties, 0), "read CUDA device");
        cuda_check(cudaDriverGetVersion(&driverVersion), "read CUDA driver version");
        CUcontext selected = nullptr;
        driver_check(cuCtxGetCurrent(&selected), "read selected context");
        report["device"] = {{"ordinal", 0}, {"name", properties.name}, {"driver_version", driverVersion}, {"context_opaque", reinterpret_cast<std::uintptr_t>(selected)}};
        report["driver_owned"] = {{"device_buffer_bytes", bytes * 2}, {"host_buffer_bytes", bytes * 3}, {"cuda_event_count", 2}, {"stream_count", 1}};
        auto baseline = process_memory();
        baseline.update({{"name", "before-load"}, {"cycle", -1}, {"index", 0}, {"observation", nullptr}});
        report["memory"].push_back(std::move(baseline));
        const auto openStart = Clock::now();
        Module module(args.module);
        timings["module_open_ns"] = nanoseconds(openStart, Clock::now());
        Observation observation(module, observed);
        using Count = int (*)();
        using Get = OfxPlugin* (*)(int);
        const auto count = module.symbol<Count>("OfxGetNumberOfPlugins");
        const auto get = module.symbol<Get>("OfxGetPlugin");
        require(count() == 1, "discovery membership failed");
        auto* plugin = get(0);
        require(plugin && plugin->mainEntry && plugin->setHost && plugin->pluginApi &&
                    std::string_view(plugin->pluginApi) == kOfxImageEffectPluginApi && plugin->pluginIdentifier &&
                    std::string_view(plugin->pluginIdentifier) == "com.juicer.Juicer",
                "invalid OFX descriptor or entry points");
        InstanceHost host;
        Lifecycle lifecycle(plugin, host);
        OfxStatus load = kOfxStatFailed;
        timed(timings["load_ns"], [&] {
            load = lifecycle.load();
        });
        life["module_load_status"] = load;
        require(load == kOfxStatOK, "Load failed");
        timed(timings["describe_ns"], [&] {
            const auto describe = lifecycle.call(kOfxActionDescribe, host.descriptor_handle());
            life["describe_status"] = describe;
            require(describe == kOfxStatOK, "Describe failed");
            const auto context = lifecycle.call(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args());
            life["describe_in_context_status"] = context;
            require(context == kOfxStatOK, "DescribeInContext failed");
        });
        Json controlTypes = Json::object();
        for (const auto& call : host.calls()) {
            if (call.operation == "param_define" && call.values.size() == 1) {
                controlTypes[call.property] = std::get<std::string>(call.values.front());
            }
        }
        report["control_types"] = controlTypes;
        for (const auto& definition : ofx_test::kInstanceParameters) {
            require(controlTypes.contains(definition.name) && controlTypes.at(definition.name) == parameter_type(definition.kind),
                    std::string("consumed control missing or wrong descriptor type: ") + std::string(definition.name));
        }
        std::array<Sample, 71> samples{};
        std::size_t sampleCount = 0;
        report["renders"].get_ref<Json::array_t&>().reserve(check ? 15 : 71);
        auto endpoint = [&](const char* name, int cycle, int index = 0) {
            Json memory = process_memory();
            memory.update({{"name", name}, {"cycle", cycle}, {"index", index}, {"observation", observed ? observation_json(observation.snapshot()) : Json(nullptr)}});
            report["memory"].push_back(std::move(memory));
            require(observation.status == FJ_MEASUREMENT_OK, "invalid owned-memory observation");
        };
        auto render = [&](OfxImageEffectHandle effect, RenderStep step) {
            cuda_check(cudaMemcpyAsync(device.destination, initial.data(), bytes, cudaMemcpyHostToDevice, device.stream), "reset destination");
            cuda_check(cudaStreamSynchronize(device.stream), "finish destination reset");
            const auto sourceReleases = host.image_release_count(effect, "Source");
            const auto destinationReleases = host.image_release_count(effect, "Output");
            host.clear_calls();
            require(sampleCount < samples.size(), "unexpected render beyond fixed sample storage");
            auto& sample = samples[sampleCount++];
            sample.phase = step.phase;
            sample.cycle = step.cycle;
            sample.index = step.index;
            sample.expectedConstruction = step.expect_construction;
            sample.before = observation.snapshot();
            sample.beforeStatus = observation.status;
            CUcontext before = nullptr, after = nullptr;
            driver_check(cuCtxGetCurrent(&before), "read context before Render");
            cuda_check(cudaEventRecord(device.start, device.stream), "record stream start");
            const auto start = Clock::now();
            sample.status = lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args());
            const auto stop = Clock::now();
            const auto streamStopStatus = cudaEventRecord(device.stop, device.stream);
            sample.cpuNs = nanoseconds(start, stop);
            sample.after = observation.snapshot();
            sample.afterStatus = observation.status;
            sample.checked = step.readback;
            try {
                cuda_check(streamStopStatus, "record stream stop");
                driver_check(cuCtxGetCurrent(&after), "read context after Render");
                cuda_check(cudaEventSynchronize(device.stop), "complete measured stream interval");
                cuda_check(cudaEventElapsedTime(&sample.streamMs, device.start, device.stop), "read stream interval");
                sample.streamTimed = true;
                sample.releasesValid = host.image_release_count(effect, "Source") == sourceReleases + 1 && host.image_release_count(effect, "Output") == destinationReleases + 1;
                sample.contextPreserved = before == after;
                sample.checked = step.readback;
                if (step.readback) {
                    cuda_check(cudaMemcpy(output.data(), device.destination, bytes, cudaMemcpyDeviceToHost), "readback output");
                    sample.outputValid = output_valid(source, initial, output, test);
                }
            } catch (...) {
                report["renders"].push_back(sample_json(sample, observed));
                throw;
            }
            report["renders"].push_back(sample_json(sample, observed));
            require(sample.status == kOfxStatOK && !host.callback_error(), "Render callback failed; construction outcome unknown");
            require(sample.releasesValid && sample.contextPreserved && (!step.readback || sample.outputValid), "Render output/lifecycle failure");
            if (observed) {
                const auto& outcome = report["renders"].back().at("construction").at("outcome");
                require(outcome == (step.expect_construction ? "success" : "reused"), "missing, unexpected or unpaired construction span");
            }
        };
        const int warmups = check ? 1 : 10, measured = check ? 1 : 30;
        for (int cycle = 0; cycle < 3; ++cycle) {
            auto* effect = host.create_effect();
            timings["author_ns"].push_back(nullptr);
            timed(timings["author_ns"].back(), [&] {
                ofx_test::author_parameters(host, effect, test);
            });
            OfxStatus createStatus = kOfxStatFailed;
            timings["create_ns"].push_back(nullptr);
            timed(timings["create_ns"].back(), [&] {
                createStatus = lifecycle.create(effect);
            });
            life["create_statuses"].push_back(createStatus);
            require(createStatus == kOfxStatOK, "CreateInstance failed");
            life["effects_created"] = cycle + 1;
            const auto controls = control_map(host, effect);
            if (cycle == 0)
                report["controls"] = controls;
            else
                require(controls == report.at("controls"), "effective controls changed across owners");
            ofx_test::configure(host, effect, test, device);
            endpoint("after-create", cycle);
            render(effect, {.phase = cycle == 0 ? "cold" : "cycle-cold", .cycle = cycle, .expect_construction = true, .readback = true});
            endpoint("after-cold", cycle);
            for (int i = 0; i < warmups; ++i)
                render(effect, {.phase = cycle == 0 ? "warmup" : "cycle-warmup", .cycle = cycle, .index = i});
            endpoint("after-warmups", cycle);
            if (cycle == 0) {
                for (int i = 0; i < measured; ++i)
                    render(effect, {.phase = "warm", .cycle = cycle, .index = i, .readback = i == measured - 1});
                endpoint("after-sample", cycle, measured);
                constexpr std::array<double, 3> gamma{1.1, 1.0, 1.1};
                for (std::size_t i = 0; i < gamma.size(); ++i) {
                    OfxStatus status = kOfxStatFailed;
                    report["edits"].push_back({{"gamma", gamma[i]}, {"edit_ns", nullptr}, {"status", nullptr}});
                    auto& edit = report["edits"].back();
                    timed(edit["edit_ns"], [&] {
                        host.set_value(effect, "FilmGammaFactor", {gamma[i]});
                        status = lifecycle.call(kOfxActionInstanceChanged, effect, host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "FilmGammaFactor"}));
                    });
                    edit["status"] = status;
                    require(status == kOfxStatOK || status == kOfxStatReplyDefault, "parameter change callback failed");
                    require(host.value(effect, "FilmGammaFactor") == std::vector<ofx_test::InstanceValue>{gamma[i]}, "parameter edit readback differs");
                    render(effect, {.phase = "changed", .cycle = cycle, .index = static_cast<int>(i), .expect_construction = true, .readback = true});
                    endpoint("after-edit", cycle, static_cast<int>(i));
                    render(effect, {.phase = "changed-reuse", .cycle = cycle, .index = static_cast<int>(i), .readback = true});
                }
            } else
                render(effect, {.phase = "cycle-end", .cycle = cycle, .readback = true});
            OfxStatus destroyStatus = kOfxStatFailed;
            timings["destroy_ns"].push_back(nullptr);
            timed(timings["destroy_ns"].back(), [&] {
                destroyStatus = lifecycle.destroy();
            });
            life["destroy_statuses"].push_back(destroyStatus);
            require(destroyStatus == kOfxStatOK, "DestroyInstance failed");
            life["effects_destroyed"] = cycle + 1;
            endpoint("after-destroy", cycle);
        }
        require(sampleCount == (check ? 15u : 71u), "incomplete render sample membership");
        OfxStatus unload = kOfxStatFailed;
        timed(timings["unload_ns"], [&] {
            unload = lifecycle.unload();
        });
        life["module_unload_status"] = unload;
        require(unload == kOfxStatOK && !host.callback_error(), "normal Unload failed");
        endpoint("after-unload", -1);
        observation.finish();
        if (observed) {
            const auto final = observation.snapshot();
            report["drained"] = {{"device_charged_bytes", final.device_charged_bytes}, {"device_records", final.device_record_count}, {"pinned_upload_pool_physical_bytes", final.upload_live_bytes}};
            require(final.device_charged_bytes == 0 && final.device_record_count == 0 && final.upload_live_bytes == 0, "measured owned families did not drain");
        }
        module.close();
        life["module_closed"] = true;
        device.close();
        driver_check(cuCtxSetCurrent(restore.original), "restore initial context");
        CUcontext restored = nullptr;
        driver_check(cuCtxGetCurrent(&restored), "inspect restored context");
        require(restored == restore.original, "original context not restored");
        life["context_restored"] = true;
        life["callback_error"] = host.callback_error();
        report["measurement_valid"] = true;
        report["status"] = "passed";
    }
} // namespace

namespace {
    int measurement_main(int argc, char** argv) {
        Json report = Json::object();
        std::filesystem::path output;
        int status = 1;
        try {
            const auto args = arguments(argc, argv);
            output = args.output;
            run(args, report);
            status = 0;
        } catch (const std::exception& error) {
            report["status"] = "failed";
            report["error"] = error.what();
        } catch (...) {
            report["status"] = "failed";
            report["error"] = "unknown measurement failure";
        }
        if (status != 0) {
            report["measurement_valid"] = false;
            if (report.contains("renders")) {
                for (auto& render : report["renders"]) {
                    if (render["construction"]["outcome"] == "success") {
                        render["construction"]["outcome"] = "invalid";
                        render["construction"]["duration_ns"] = nullptr;
                    }
                }
            }
        }
        try {
            require(!output.empty(), "missing measurement output path");
            std::ofstream stream(output);
            stream << report.dump(2) << '\n';
            require(static_cast<bool>(stream), "cannot write raw measurement");
        } catch (...) {
            return 1;
        }
        return status;
    }

} // namespace

int main(int argc, char** argv) {
    try {
        return measurement_main(argc, argv);
    } catch (...) {
        (void)std::fputs("measurement failed before its record could be written\n", stderr);
        return 1;
    }
}
