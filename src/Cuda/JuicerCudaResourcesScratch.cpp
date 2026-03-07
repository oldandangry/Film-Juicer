// Cuda/JuicerCudaResourcesScratch.cpp
//
// Included by JuicerCudaResources.cpp (single-TU split).
    enum class SharedGaussianKind : std::uint8_t {
        Standard = 0,
        Halation = 1,
        SpatialDir = 2
    };

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)

    struct SharedGaussianKey {
        int deviceId = -1;
        void* contextOpaque = nullptr;
        SharedGaussianKind kind = SharedGaussianKind::Standard;
        int radius = 0;
        std::uint32_t sigmaBits = 0;

        bool operator==(const SharedGaussianKey& other) const noexcept {
            return deviceId == other.deviceId &&
                contextOpaque == other.contextOpaque &&
                kind == other.kind &&
                radius == other.radius &&
                sigmaBits == other.sigmaBits;
        }
    };

    struct SharedGaussianKeyHash {
        std::size_t operator()(const SharedGaussianKey& key) const noexcept {
            const std::size_t hDevice = std::hash<int>{}(key.deviceId);
            const std::size_t hContext = std::hash<std::uintptr_t>{}(
                reinterpret_cast<std::uintptr_t>(key.contextOpaque));
            const std::size_t hKind = std::hash<std::uint8_t>{}(
                static_cast<std::uint8_t>(key.kind));
            const std::size_t hRadius = std::hash<int>{}(key.radius);
            const std::size_t hSigma = std::hash<std::uint32_t>{}(key.sigmaBits);
            std::size_t h = hDevice;
            h ^= hContext + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            h ^= hKind + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            h ^= hRadius + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            h ^= hSigma + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            return h;
        }
    };

    struct SharedGaussianEntry {
        float* weights = nullptr;
        int radius = 0;
        float sigma = 0.0f;
        int capacity = 0;
        std::uint64_t id = 0;
    };

    struct SharedGaussianCacheState {
        std::mutex mutex;
        std::unordered_map<SharedGaussianKey, SharedGaussianEntry, SharedGaussianKeyHash> byKey;
    };

    static SharedGaussianCacheState& shared_gaussian_cache_state() {
        static SharedGaussianCacheState state;
        return state;
    }

    static std::uint32_t gaussian_sigma_bits(float sigma) noexcept {
        float canonical = sigma;
        if (canonical == 0.0f) {
            canonical = 0.0f;
        }
        std::uint32_t bits = 0;
        std::memcpy(&bits, &canonical, sizeof(bits));
        return bits;
    }

    static std::uint64_t make_shared_gaussian_id(const SharedGaussianKey& key) noexcept {
        std::uint64_t h = Hash::kFnvOffset;
        Hash::hash_bytes_update(h, &key.deviceId, sizeof(key.deviceId));
        const std::uintptr_t contextBits =
            reinterpret_cast<std::uintptr_t>(key.contextOpaque);
        Hash::hash_bytes_update(h, &contextBits, sizeof(contextBits));
        const std::uint8_t kindValue = static_cast<std::uint8_t>(key.kind);
        Hash::hash_bytes_update(h, &kindValue, sizeof(kindValue));
        Hash::hash_bytes_update(h, &key.radius, sizeof(key.radius));
        Hash::hash_bytes_update(h, &key.sigmaBits, sizeof(key.sigmaBits));
        return h;
    }

    static SharedGaussianKey make_shared_gaussian_key(
        const Resources& resources,
        SharedGaussianKind kind,
        int radius,
        float sigma) noexcept {
        SharedGaussianKey key{};
        key.deviceId = resources.deviceId;
        key.contextOpaque = resources.ownerContextOpaque;
        key.kind = kind;
        key.radius = radius;
        key.sigmaBits = gaussian_sigma_bits(sigma);
        return key;
    }

    static void build_gaussian_weights_cpu(
        int radius,
        float sigma,
        std::vector<float>& outWeights) {
        outWeights.clear();
        if (radius <= 0 || !std::isfinite(sigma) || sigma <= 0.0f) {
            return;
        }
        outWeights.resize(static_cast<std::size_t>(2 * radius + 1));
        const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
        double wsum = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            const double w = std::exp(-(static_cast<double>(i * i)) / s2);
            outWeights[static_cast<std::size_t>(i + radius)] = static_cast<float>(w);
            wsum += w;
        }
        const double invW = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
        for (float& w : outWeights) {
            w = static_cast<float>(static_cast<double>(w) * invW);
        }
    }

    static bool acquire_shared_gaussian_entry(
        const SharedGaussianKey& key,
        int radius,
        float sigma,
        const std::vector<float>& cpuWeights,
        SharedGaussianEntry& outEntry,
        std::string& outError) {
        outError.clear();
        SharedGaussianCacheState& cache = shared_gaussian_cache_state();
        {
            std::lock_guard<std::mutex> lock(cache.mutex);
            const auto it = cache.byKey.find(key);
            if (it != cache.byKey.end()) {
                outEntry = it->second;
                return true;
            }
        }

        float* dWeights = nullptr;
        const std::size_t bytes = cpuWeights.size() * sizeof(float);
        const cudaError_t allocErr = cudaMalloc(reinterpret_cast<void**>(&dWeights), bytes);
        if (allocErr != cudaSuccess || !dWeights) {
            outError = std::string("cudaMalloc(shared gaussian kernel) failed: ")
                + (cudaGetErrorString(allocErr) ? cudaGetErrorString(allocErr) : "(unknown)");
            dWeights = nullptr;
            return false;
        }
        const cudaError_t copyErr = cudaMemcpy(
            dWeights,
            cpuWeights.data(),
            bytes,
            cudaMemcpyHostToDevice);
        if (copyErr != cudaSuccess) {
            outError = std::string("cudaMemcpy(shared gaussian kernel) failed: ")
                + (cudaGetErrorString(copyErr) ? cudaGetErrorString(copyErr) : "(unknown)");
            cudaFree(dWeights);
            return false;
        }

        std::lock_guard<std::mutex> lock(cache.mutex);
        const auto existingIt = cache.byKey.find(key);
        if (existingIt != cache.byKey.end()) {
            cudaFree(dWeights);
            outEntry = existingIt->second;
            return true;
        }

        SharedGaussianEntry entry{};
        entry.weights = dWeights;
        entry.radius = radius;
        entry.sigma = sigma;
        entry.capacity = static_cast<int>(cpuWeights.size());
        entry.id = make_shared_gaussian_id(key);
        cache.byKey.emplace(key, entry);
        outEntry = entry;
        return true;
    }

    static bool shared_gaussian_entry_matches_cache(
        const SharedGaussianKey& key,
        const Resources::DeviceGaussianKernel& kernel) {
        if (kernel.sharedKernelId == 0 || !kernel.weights) {
            return false;
        }
        SharedGaussianCacheState& cache = shared_gaussian_cache_state();
        std::lock_guard<std::mutex> lock(cache.mutex);
        const auto it = cache.byKey.find(key);
        if (it == cache.byKey.end()) {
            return false;
        }
        return it->second.id == kernel.sharedKernelId &&
            it->second.weights == kernel.weights;
    }

    static const char* to_cstr(ResourceManager::AllocatorBackendMode mode) noexcept {
        switch (mode) {
        case ResourceManager::AllocatorBackendMode::Legacy:
            return "legacy";
        case ResourceManager::AllocatorBackendMode::AsyncPool:
            return "async_pool";
        case ResourceManager::AllocatorBackendMode::Slab:
            return "slab";
        default:
            return "unknown";
        }
    }

    static ResourceManager::AllocatorBackendMode scratch_allocator_backend_mode_locked(
        const Resources& resources) noexcept {
        ResourceManager::DeviceContextKey key{};
        key.deviceId = resources.deviceId;
        key.contextOpaque = resources.ownerContextOpaque;
        return ResourceManager::query_allocator_backend_mode(key);
    }

    static bool is_async_device_ptr_tracked_locked(const Resources& resources, const void* ptr) noexcept {
        return ptr &&
            resources.asyncDeviceAllocPointers.find(const_cast<void*>(ptr)) !=
                resources.asyncDeviceAllocPointers.end();
    }

    static void track_async_device_ptr_locked(Resources& resources, void* ptr, bool asyncAllocated) noexcept {
        if (!ptr) {
            return;
        }
        if (asyncAllocated) {
            resources.asyncDeviceAllocPointers.insert(ptr);
        }
        else {
            resources.asyncDeviceAllocPointers.erase(ptr);
        }
    }

    static void untrack_async_device_ptr_locked(Resources& resources, void* ptr) noexcept {
        if (!ptr) {
            return;
        }
        resources.asyncDeviceAllocPointers.erase(ptr);
    }

    static cudaError_t device_free_async_compat(void* ptr, void* cudaStreamOpaque) noexcept {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        return cudaFreeAsync(ptr, stream);
#else
        (void)cudaStreamOpaque;
        return cudaErrorNotSupported;
#endif
    }

    static void free_tracked_device_ptr_locked(
        Resources& resources,
        void*& ptr,
        void* cudaStreamOpaque) noexcept {
        if (!ptr) {
            return;
        }
        const bool asyncTracked = is_async_device_ptr_tracked_locked(resources, ptr);
        if (asyncTracked) {
            const cudaError_t asyncErr = device_free_async_compat(ptr, cudaStreamOpaque);
            if (asyncErr != cudaSuccess) {
                (void)cudaFree(ptr);
            }
            else if (!cudaStreamOpaque) {
                (void)cudaStreamSynchronize(nullptr);
            }
        }
        else {
            (void)cudaFree(ptr);
        }
        untrack_async_device_ptr_locked(resources, ptr);
        ptr = nullptr;
    }

    template <typename T>
    static void free_tracked_device_ptr_locked(
        Resources& resources,
        T*& ptr,
        void* cudaStreamOpaque) noexcept {
        void* raw = reinterpret_cast<void*>(ptr);
        free_tracked_device_ptr_locked(resources, raw, cudaStreamOpaque);
        ptr = reinterpret_cast<T*>(raw);
    }

    static bool allocate_scratch_device_ptr_locked(
        Resources& resources,
        void*& outPtr,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
        outPtr = nullptr;
        if (bytes == 0) {
            outError = std::string(label ? label : "scratch") + " bytes invalid";
            return false;
        }

        const ResourceManager::AllocatorBackendMode backendMode =
            scratch_allocator_backend_mode_locked(resources);
        const bool preferAsync = (backendMode == ResourceManager::AllocatorBackendMode::AsyncPool);

        cudaError_t asyncErr = cudaSuccess;
        if (preferAsync) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
            const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
            asyncErr = cudaMallocAsync(reinterpret_cast<void**>(&outPtr), bytes, stream);
            if (asyncErr == cudaSuccess && outPtr) {
                track_async_device_ptr_locked(resources, outPtr, true);
                return true;
            }
            outPtr = nullptr;
#else
            asyncErr = cudaErrorNotSupported;
#endif
        }

        const cudaError_t allocErr = cudaMalloc(reinterpret_cast<void**>(&outPtr), bytes);
        if (allocErr == cudaSuccess && outPtr) {
            track_async_device_ptr_locked(resources, outPtr, false);
            if (preferAsync && JTRACE_ENABLED(2)) {
                std::ostringstream oss;
                oss << "event=alloc_fallback"
                    << " path=scratch"
                    << " label=" << (label ? label : "scratch")
                    << " backend_requested=" << to_cstr(backendMode)
                    << " async_error=" << (cudaGetErrorString(asyncErr) ? cudaGetErrorString(asyncErr) : "(unknown)");
                JTRACE("MSALC", oss.str());
            }
            return true;
        }

        if (preferAsync) {
            outError = std::string("scratch alloc failed (async+legacy) [")
                + (label ? label : "scratch")
                + "]: async="
                + (cudaGetErrorString(asyncErr) ? cudaGetErrorString(asyncErr) : "(unknown)")
                + ", legacy="
                + (cudaGetErrorString(allocErr) ? cudaGetErrorString(allocErr) : "(unknown)");
        }
        else {
            outError = std::string("cudaMalloc(") + (label ? label : "scratch") + ") failed: "
                + (cudaGetErrorString(allocErr) ? cudaGetErrorString(allocErr) : "(unknown)");
        }
        outPtr = nullptr;
        return false;
    }

    template <typename T>
    static bool allocate_scratch_device_ptr_locked(
        Resources& resources,
        T*& outPtr,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
        void* raw = nullptr;
        const bool ok = allocate_scratch_device_ptr_locked(
            resources,
            raw,
            bytes,
            cudaStreamOpaque,
            label,
            outError);
        outPtr = reinterpret_cast<T*>(raw);
        return ok;
    }
#endif

    static void free_auto_exposure(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.autoExposureScratch.partialsA) {
            cudaFree(resources.autoExposureScratch.partialsA);
            resources.autoExposureScratch.partialsA = nullptr;
        }
        if (resources.autoExposureScratch.partialsB) {
            cudaFree(resources.autoExposureScratch.partialsB);
            resources.autoExposureScratch.partialsB = nullptr;
        }
        if (resources.autoExposureScratch.maxYBits) {
            cudaFree(resources.autoExposureScratch.maxYBits);
            resources.autoExposureScratch.maxYBits = nullptr;
        }
        if (resources.autoExposureScratch.histogram) {
            cudaFree(resources.autoExposureScratch.histogram);
            resources.autoExposureScratch.histogram = nullptr;
        }
        if (resources.autoExposureScratch.weightsX) {
            cudaFree(resources.autoExposureScratch.weightsX);
            resources.autoExposureScratch.weightsX = nullptr;
        }
        if (resources.autoExposureScratch.weightsY) {
            cudaFree(resources.autoExposureScratch.weightsY);
            resources.autoExposureScratch.weightsY = nullptr;
        }
        if (resources.autoExposureExposureScale) {
            cudaFree(resources.autoExposureExposureScale);
            resources.autoExposureExposureScale = nullptr;
        }
        if (resources.autoExposureAutoEV) {
            cudaFree(resources.autoExposureAutoEV);
            resources.autoExposureAutoEV = nullptr;
        }
        if (resources.autoExposureValid) {
            cudaFree(resources.autoExposureValid);
            resources.autoExposureValid = nullptr;
        }
#endif
        resources.autoExposureScratch.partialCapacity = 0;
        resources.autoExposureScratch.weightsXCapacity = 0;
        resources.autoExposureScratch.weightsYCapacity = 0;
        resources.autoExposureScratch.weightsWidth = 0;
        resources.autoExposureScratch.weightsHeight = 0;
        resources.autoExposureKeyHash = 0;
        resources.autoExposureSliderEV = std::numeric_limits<double>::quiet_NaN();
    }

    static void free_tables(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.tablesAx) {
            cudaFree(resources.tablesAx);
            resources.tablesAx = nullptr;
        }
        if (resources.tablesAy) {
            cudaFree(resources.tablesAy);
            resources.tablesAy = nullptr;
        }
        if (resources.tablesAz) {
            cudaFree(resources.tablesAz);
            resources.tablesAz = nullptr;
        }
        if (resources.tablesIllum) {
            cudaFree(resources.tablesIllum);
            resources.tablesIllum = nullptr;
        }
#endif
        resources.tablesK = 0;
    }

    static bool retire_tables_locked(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t bytes = static_cast<size_t>(std::max(0, resources.tablesK)) * sizeof(float);
        if (resources.tablesAx) {
            if (!retire_ptr_locked(resources, resources.tablesAx, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.tablesAx = nullptr;
        }
        if (resources.tablesAy) {
            if (!retire_ptr_locked(resources, resources.tablesAy, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.tablesAy = nullptr;
        }
        if (resources.tablesAz) {
            if (!retire_ptr_locked(resources, resources.tablesAz, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.tablesAz = nullptr;
        }
        if (resources.tablesIllum) {
            if (!retire_ptr_locked(resources, resources.tablesIllum, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.tablesIllum = nullptr;
        }
        resources.tablesK = 0;
        return true;
#endif
    }

    static void free_spectral_tables(Resources::DeviceSpectralTables& t) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (t.epsC) { cudaFree(t.epsC); t.epsC = nullptr; }
        if (t.epsM) { cudaFree(t.epsM); t.epsM = nullptr; }
        if (t.epsY) { cudaFree(t.epsY); t.epsY = nullptr; }
        if (t.Ax) { cudaFree(t.Ax); t.Ax = nullptr; }
        if (t.Ay) { cudaFree(t.Ay); t.Ay = nullptr; }
        if (t.Az) { cudaFree(t.Az); t.Az = nullptr; }
        if (t.baseMin) { cudaFree(t.baseMin); t.baseMin = nullptr; }
#endif
        t.K = 0;
        t.hasBaseline = 0;
        t.invYn = 1.0f;
    }

    static bool retire_spectral_tables_locked(Resources& resources, Resources::DeviceSpectralTables& t, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)t;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t bytes = static_cast<size_t>(std::max(0, t.K)) * sizeof(float);
        if (t.epsC) {
            if (!retire_ptr_locked(resources, t.epsC, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.epsC = nullptr;
        }
        if (t.epsM) {
            if (!retire_ptr_locked(resources, t.epsM, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.epsM = nullptr;
        }
        if (t.epsY) {
            if (!retire_ptr_locked(resources, t.epsY, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.epsY = nullptr;
        }
        if (t.Ax) {
            if (!retire_ptr_locked(resources, t.Ax, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.Ax = nullptr;
        }
        if (t.Ay) {
            if (!retire_ptr_locked(resources, t.Ay, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.Ay = nullptr;
        }
        if (t.Az) {
            if (!retire_ptr_locked(resources, t.Az, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.Az = nullptr;
        }
        if (t.baseMin) {
            if (!retire_ptr_locked(resources, t.baseMin, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.baseMin = nullptr;
        }
        t.K = 0;
        t.hasBaseline = 0;
        t.invYn = 1.0f;
        return true;
#endif
    }

    static void free_scan_medium(Resources::DeviceScanMedium& m) noexcept {
        free_spectral_tables(m.tables);
        m.mediumIsNegative = 1;
        m.min_cmy[0] = m.min_cmy[1] = m.min_cmy[2] = 0.0f;
        m.inv_max_cmy[0] = m.inv_max_cmy[1] = m.inv_max_cmy[2] = 1.0f;
    }

    static bool retire_scan_medium_locked(Resources& resources, Resources::DeviceScanMedium& m, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)m;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!retire_spectral_tables_locked(resources, m.tables, cudaStreamOpaque, label, outError)) {
            return false;
        }
        m.mediumIsNegative = 1;
        m.min_cmy[0] = m.min_cmy[1] = m.min_cmy[2] = 0.0f;
        m.inv_max_cmy[0] = m.inv_max_cmy[1] = m.inv_max_cmy[2] = 1.0f;
        return true;
#endif
    }

    static void free_scan_lut(Resources::DeviceSpectralLut& lut) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (lut.log2XYZ) {
            cudaFree(lut.log2XYZ);
            lut.log2XYZ = nullptr;
        }
#endif
        lut.res = 0;
        lut.hash = 0;
    }

    static void free_gaussian_kernel(Resources::DeviceGaussianKernel& k) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (k.weights) {
            if (k.sharedKernelId == 0) {
                cudaFree(k.weights);
            }
            k.weights = nullptr;
        }
#endif
        k.radius = 0;
        k.sigma = 0.0f;
        k.capacity = 0;
        k.sharedKernelId = 0;
    }

    static void free_optics_scratch(Resources& resources, Resources::DeviceOpticsScratch& s, void* cudaStreamOpaque = nullptr) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        free_tracked_device_ptr_locked(resources, s.rgbR, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.rgbG, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.rgbB, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.blurred, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.aux, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.grainTmp, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.grainTmpShared, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.grainTmpMid, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.grainTmpCoarse, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.gateMask, cudaStreamOpaque);
#endif
        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
        s.gateWidth = 0;
        s.gateHeight = 0;
        s.gateMaskHash = 0;
    }

    static bool retire_optics_scratch_locked(Resources& resources, Resources::DeviceOpticsScratch& s, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)s;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t planeN = static_cast<size_t>(std::max(0, s.width)) * static_cast<size_t>(std::max(0, s.height));
        const size_t planeBytes = planeN * sizeof(float);
        if (s.rgbR) {
            if (!retire_ptr_locked(resources, s.rgbR, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.rgbR = nullptr;
        }
        if (s.rgbG) {
            if (!retire_ptr_locked(resources, s.rgbG, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.rgbG = nullptr;
        }
        if (s.rgbB) {
            if (!retire_ptr_locked(resources, s.rgbB, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.rgbB = nullptr;
        }
        if (s.blurred) {
            if (!retire_ptr_locked(resources, s.blurred, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.blurred = nullptr;
        }
        if (s.aux) {
            if (!retire_ptr_locked(resources, s.aux, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.aux = nullptr;
        }
        if (s.grainTmp) {
            if (!retire_ptr_locked(resources, s.grainTmp, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.grainTmp = nullptr;
        }
        if (s.grainTmpShared) {
            if (!retire_ptr_locked(resources, s.grainTmpShared, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.grainTmpShared = nullptr;
        }
        if (s.grainTmpMid) {
            if (!retire_ptr_locked(resources, s.grainTmpMid, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.grainTmpMid = nullptr;
        }
        if (s.grainTmpCoarse) {
            if (!retire_ptr_locked(resources, s.grainTmpCoarse, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.grainTmpCoarse = nullptr;
        }

        const size_t gateN = static_cast<size_t>(std::max(0, s.gateWidth)) * static_cast<size_t>(std::max(0, s.gateHeight));
        const size_t gateBytes = gateN * sizeof(float);
        if (s.gateMask) {
            if (!retire_ptr_locked(resources, s.gateMask, gateBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.gateMask = nullptr;
        }

        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
        s.gateWidth = 0;
        s.gateHeight = 0;
        s.gateMaskHash = 0;
        return true;
#endif
    }

    static void free_spatial_dir_scratch(Resources& resources, Resources::DeviceSpatialDirScratch& s, void* cudaStreamOpaque = nullptr) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        free_tracked_device_ptr_locked(resources, s.corrY, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.corrM, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.corrC, cudaStreamOpaque);
#endif
        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
    }

    static bool retire_spatial_dir_scratch_locked(Resources& resources, Resources::DeviceSpatialDirScratch& s, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)s;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t n = static_cast<size_t>(std::max(0, s.width)) * static_cast<size_t>(std::max(0, s.height));
        const size_t bytes = n * sizeof(float);
        if (s.corrY) {
            if (!retire_ptr_locked(resources, s.corrY, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.corrY = nullptr;
        }
        if (s.corrM) {
            if (!retire_ptr_locked(resources, s.corrM, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.corrM = nullptr;
        }
        if (s.corrC) {
            if (!retire_ptr_locked(resources, s.corrC, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            s.corrC = nullptr;
        }
        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
        return true;
#endif
    }

    static void free_shared_tmp_plane(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        free_tracked_device_ptr_locked(
            resources,
            resources.sharedTmpPlane,
            nullptr);
#endif
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;
    }

    bool ensure_auto_exposure_buffers(Resources& resources, int meterWidth, int meterHeight, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)meterWidth;
        (void)meterHeight;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (meterWidth <= 0 || meterHeight <= 0) {
            outError = "auto-exposure meter dimensions invalid";
            return false;
        }

        const int blockX = 16;
        const int blockY = 16;
        const int gridX = (meterWidth + blockX - 1) / blockX;
        const int gridY = (meterHeight + blockY - 1) / blockY;
        const int neededPartials = gridX * gridY;
        if (neededPartials <= 0) {
            outError = "auto-exposure partial count invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        if (!resources.autoExposureExposureScale) {
            const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureExposureScale), sizeof(float));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure scale) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }
        }
        if (!resources.autoExposureAutoEV) {
            const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureAutoEV), sizeof(double));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure autoEV) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }
        }
        if (!resources.autoExposureValid) {
            const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureValid), sizeof(int));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure valid) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }
        }

        if (!resources.autoExposureScratch.maxYBits) {
            const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureScratch.maxYBits), sizeof(unsigned int));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure maxYBits) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }
        }
        if (!resources.autoExposureScratch.histogram) {
            const size_t bytes = sizeof(unsigned int) * 2048u;
            const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureScratch.histogram), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure histogram) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }
        }

        const bool needWeightsX = resources.autoExposureScratch.weightsXCapacity < meterWidth || !resources.autoExposureScratch.weightsX;
        const bool needWeightsY = resources.autoExposureScratch.weightsYCapacity < meterHeight || !resources.autoExposureScratch.weightsY;
        if (needWeightsX || needWeightsY) {
            if (needWeightsX) {
                if (resources.autoExposureScratch.weightsX) {
                    const size_t oldBytes = static_cast<size_t>(std::max(0, resources.autoExposureScratch.weightsXCapacity)) * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.autoExposureScratch.weightsX, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "auto-exposure weightsX", outError)) {
                        return false;
                    }
                    resources.autoExposureScratch.weightsX = nullptr;
                }
                const size_t bytes = static_cast<size_t>(meterWidth) * sizeof(float);
                const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureScratch.weightsX), bytes);
                if (err != cudaSuccess) {
                    outError = std::string("cudaMalloc(auto-exposure weightsX) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    free_auto_exposure(resources);
                    return false;
                }
                resources.autoExposureScratch.weightsXCapacity = meterWidth;
            }
            if (needWeightsY) {
                if (resources.autoExposureScratch.weightsY) {
                    const size_t oldBytes = static_cast<size_t>(std::max(0, resources.autoExposureScratch.weightsYCapacity)) * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.autoExposureScratch.weightsY, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "auto-exposure weightsY", outError)) {
                        return false;
                    }
                    resources.autoExposureScratch.weightsY = nullptr;
                }
                const size_t bytes = static_cast<size_t>(meterHeight) * sizeof(float);
                const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureScratch.weightsY), bytes);
                if (err != cudaSuccess) {
                    outError = std::string("cudaMalloc(auto-exposure weightsY) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    free_auto_exposure(resources);
                    return false;
                }
                resources.autoExposureScratch.weightsYCapacity = meterHeight;
            }

            resources.autoExposureScratch.weightsWidth = 0;
            resources.autoExposureScratch.weightsHeight = 0;
        }

        if (resources.autoExposureScratch.partialCapacity < neededPartials) {
            if (resources.autoExposureScratch.partialsA || resources.autoExposureScratch.partialsB) {
                const size_t oldBytes = static_cast<size_t>(std::max(0, resources.autoExposureScratch.partialCapacity)) * sizeof(JuicerCudaAutoExposurePartial);
                if (resources.autoExposureScratch.partialsA) {
                    if (!retire_ptr_locked(resources, resources.autoExposureScratch.partialsA, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "auto-exposure partialsA", outError)) {
                        return false;
                    }
                    resources.autoExposureScratch.partialsA = nullptr;
                }
                if (resources.autoExposureScratch.partialsB) {
                    if (!retire_ptr_locked(resources, resources.autoExposureScratch.partialsB, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "auto-exposure partialsB", outError)) {
                        return false;
                    }
                    resources.autoExposureScratch.partialsB = nullptr;
                }
                resources.autoExposureScratch.partialCapacity = 0;
            }

            const size_t bytes = static_cast<size_t>(neededPartials) * sizeof(JuicerCudaAutoExposurePartial);
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureScratch.partialsA), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure partialsA) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }
            err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureScratch.partialsB), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure partialsB) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }

            resources.autoExposureScratch.partialCapacity = neededPartials;
            resources.autoExposureKeyHash = 0;
            resources.autoExposureSliderEV = std::numeric_limits<double>::quiet_NaN();
        }

        return true;
#endif
    }

    static bool ensure_shared_tmp_plane_locked(Resources& resources, int width, int height, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)width;
        (void)height;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (width <= 0 || height <= 0) {
            outError = std::string(label ? label : "shared tmp") + " dimensions invalid";
            return false;
        }

        if (resources.sharedTmpPlane &&
            resources.sharedTmpWidth == width &&
            resources.sharedTmpHeight == height) {
            return true;
        }

        if (resources.sharedTmpPlane) {
            const size_t oldN = static_cast<size_t>(std::max(0, resources.sharedTmpWidth)) * static_cast<size_t>(std::max(0, resources.sharedTmpHeight));
            const size_t oldBytes = oldN * sizeof(float);
            if (!retire_ptr_locked(resources, resources.sharedTmpPlane, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label ? label : "shared tmp", outError)) {
                return false;
            }
            resources.sharedTmpPlane = nullptr;
        }
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;

        const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
        const size_t bytes = n * sizeof(float);
        if (!allocate_scratch_device_ptr_locked(
                resources,
                resources.sharedTmpPlane,
                bytes,
                cudaStreamOpaque,
                label ? label : "shared tmp",
                outError)) {
            resources.sharedTmpPlane = nullptr;
            return false;
        }

        resources.sharedTmpWidth = width;
        resources.sharedTmpHeight = height;
        return true;
#endif
    }

    bool ensure_optics_scratch(Resources& resources, int width, int height, bool needBlurredScratch, bool needAuxScratch, bool needGrainScratch, bool needGrainSharedScratch, bool needGateMask, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)width;
        (void)height;
        (void)needBlurredScratch;
        (void)needAuxScratch;
        (void)needGrainScratch;
        (void)needGateMask;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (width <= 0 || height <= 0) {
            outError = "optics scratch dimensions invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        const bool dimsMatch = (resources.scannerScratch.width == width && resources.scannerScratch.height == height);
        const bool haveBase = resources.scannerScratch.rgbR && resources.scannerScratch.rgbG && resources.scannerScratch.rgbB;

        if (!dimsMatch || !haveBase) {
            if (resources.scannerScratch.rgbR || resources.scannerScratch.rgbG || resources.scannerScratch.rgbB ||
                resources.scannerScratch.blurred || resources.scannerScratch.aux || resources.scannerScratch.grainTmp ||
                resources.scannerScratch.grainTmpShared || resources.scannerScratch.grainTmpMid ||
                resources.scannerScratch.grainTmpCoarse || resources.scannerScratch.gateMask) {
                if (!retire_optics_scratch_locked(resources, resources.scannerScratch, cudaStreamOpaque, "optics scratch", outError)) {
                    return false;
                }
            }
            else {
                free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
            }

            const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
            const size_t bytes = n * sizeof(float);
            if (!allocate_scratch_device_ptr_locked(
                    resources,
                    resources.scannerScratch.rgbR,
                    bytes,
                    cudaStreamOpaque,
                    "scannerScratch.rgbR",
                    outError)) {
                free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                return false;
            }
            if (!allocate_scratch_device_ptr_locked(
                    resources,
                    resources.scannerScratch.rgbG,
                    bytes,
                    cudaStreamOpaque,
                    "scannerScratch.rgbG",
                    outError)) {
                free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                return false;
            }
            if (!allocate_scratch_device_ptr_locked(
                    resources,
                    resources.scannerScratch.rgbB,
                    bytes,
                    cudaStreamOpaque,
                    "scannerScratch.rgbB",
                    outError)) {
                free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                return false;
            }

            resources.scannerScratch.width = width;
            resources.scannerScratch.height = height;
        }

        if (!ensure_shared_tmp_plane_locked(resources, width, height, cudaStreamOpaque, "shared tmp plane", outError)) {
            free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
            return false;
        }
        resources.scannerScratch.tmp = resources.sharedTmpPlane;

        if (needBlurredScratch) {
            if (!resources.scannerScratch.blurred) {
                const size_t n = static_cast<size_t>(resources.scannerScratch.width) * static_cast<size_t>(resources.scannerScratch.height);
                const size_t bytes = n * sizeof(float);
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.blurred,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.blurred",
                        outError)) {
                    return false;
                }
            }
        }
        else {
            if (resources.scannerScratch.blurred) {
                const size_t oldN = static_cast<size_t>(std::max(0, resources.scannerScratch.width)) * static_cast<size_t>(std::max(0, resources.scannerScratch.height));
                const size_t oldBytes = oldN * sizeof(float);
                if (!retire_ptr_locked(resources, resources.scannerScratch.blurred, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "unsharp scratch", outError)) {
                    return false;
                }
                resources.scannerScratch.blurred = nullptr;
            }
        }

        if (needAuxScratch) {
            if (!resources.scannerScratch.aux) {
                const size_t n = static_cast<size_t>(resources.scannerScratch.width) * static_cast<size_t>(resources.scannerScratch.height);
                const size_t bytes = n * sizeof(float);
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.aux,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.aux",
                        outError)) {
                    return false;
                }
            }
        }
        else {
            if (resources.scannerScratch.aux) {
                const size_t oldN = static_cast<size_t>(std::max(0, resources.scannerScratch.width)) * static_cast<size_t>(std::max(0, resources.scannerScratch.height));
                const size_t oldBytes = oldN * sizeof(float);
                if (!retire_ptr_locked(resources, resources.scannerScratch.aux, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain scratch", outError)) {
                    return false;
                }
                resources.scannerScratch.aux = nullptr;
            }
        }

        if (needGrainScratch) {
            if (!resources.scannerScratch.grainTmp) {
                const size_t n = static_cast<size_t>(resources.scannerScratch.width) * static_cast<size_t>(resources.scannerScratch.height);
                const size_t bytes = n * sizeof(float);
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainTmp,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.grainTmp",
                        outError)) {
                    return false;
                }
            }
            if (!resources.scannerScratch.grainTmpMid) {
                const size_t n = static_cast<size_t>(resources.scannerScratch.width) * static_cast<size_t>(resources.scannerScratch.height);
                const size_t bytes = n * sizeof(float);
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainTmpMid,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.grainTmpMid",
                        outError)) {
                    return false;
                }
            }
            if (!resources.scannerScratch.grainTmpCoarse) {
                const size_t n = static_cast<size_t>(resources.scannerScratch.width) * static_cast<size_t>(resources.scannerScratch.height);
                const size_t bytes = n * sizeof(float);
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainTmpCoarse,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.grainTmpCoarse",
                        outError)) {
                    return false;
                }
            }
        }
        else {
            if (resources.scannerScratch.grainTmp) {
                const size_t oldN = static_cast<size_t>(std::max(0, resources.scannerScratch.width)) * static_cast<size_t>(std::max(0, resources.scannerScratch.height));
                const size_t oldBytes = oldN * sizeof(float);
                if (!retire_ptr_locked(resources, resources.scannerScratch.grainTmp, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain mix scratch", outError)) {
                    return false;
                }
                resources.scannerScratch.grainTmp = nullptr;
            }
            if (resources.scannerScratch.grainTmpMid) {
                const size_t oldN = static_cast<size_t>(std::max(0, resources.scannerScratch.width)) * static_cast<size_t>(std::max(0, resources.scannerScratch.height));
                const size_t oldBytes = oldN * sizeof(float);
                if (!retire_ptr_locked(resources, resources.scannerScratch.grainTmpMid, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain mix mid scratch", outError)) {
                    return false;
                }
                resources.scannerScratch.grainTmpMid = nullptr;
            }
            if (resources.scannerScratch.grainTmpCoarse) {
                const size_t oldN = static_cast<size_t>(std::max(0, resources.scannerScratch.width)) * static_cast<size_t>(std::max(0, resources.scannerScratch.height));
                const size_t oldBytes = oldN * sizeof(float);
                if (!retire_ptr_locked(resources, resources.scannerScratch.grainTmpCoarse, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain mix coarse scratch", outError)) {
                    return false;
                }
                resources.scannerScratch.grainTmpCoarse = nullptr;
            }
        }

        if (needGrainSharedScratch) {
            if (!resources.scannerScratch.grainTmpShared) {
                const size_t n = static_cast<size_t>(resources.scannerScratch.width) * static_cast<size_t>(resources.scannerScratch.height);
                const size_t bytes = n * sizeof(float);
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainTmpShared,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.grainTmpShared",
                        outError)) {
                    return false;
                }
            }
        }
        else {
            if (resources.scannerScratch.grainTmpShared) {
                const size_t oldN = static_cast<size_t>(std::max(0, resources.scannerScratch.width)) * static_cast<size_t>(std::max(0, resources.scannerScratch.height));
                const size_t oldBytes = oldN * sizeof(float);
                if (!retire_ptr_locked(resources, resources.scannerScratch.grainTmpShared, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain shared scratch", outError)) {
                    return false;
                }
                resources.scannerScratch.grainTmpShared = nullptr;
            }
        }

        const int gateWidth = (width + 1) / 2;
        const int gateHeight = (height + 1) / 2;
        if (needGateMask) {
            const bool gateDimsMatch = (resources.scannerScratch.gateWidth == gateWidth &&
                resources.scannerScratch.gateHeight == gateHeight);
            if (!resources.scannerScratch.gateMask || !gateDimsMatch) {
                if (resources.scannerScratch.gateMask) {
                    const size_t oldN = static_cast<size_t>(std::max(0, resources.scannerScratch.gateWidth)) * static_cast<size_t>(std::max(0, resources.scannerScratch.gateHeight));
                    const size_t oldBytes = oldN * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.scannerScratch.gateMask, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "gate defect mask", outError)) {
                        return false;
                    }
                    resources.scannerScratch.gateMask = nullptr;
                }
                const size_t n = static_cast<size_t>(gateWidth) * static_cast<size_t>(gateHeight);
                const size_t bytes = n * sizeof(float);
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.gateMask,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.gateMask",
                        outError)) {
                    return false;
                }
                resources.scannerScratch.gateWidth = gateWidth;
                resources.scannerScratch.gateHeight = gateHeight;
                resources.scannerScratch.gateMaskHash = 0;
            }
        }
        else if (resources.scannerScratch.gateMask) {
            const size_t oldN = static_cast<size_t>(std::max(0, resources.scannerScratch.gateWidth)) * static_cast<size_t>(std::max(0, resources.scannerScratch.gateHeight));
            const size_t oldBytes = oldN * sizeof(float);
            if (!retire_ptr_locked(resources, resources.scannerScratch.gateMask, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "gate defect mask", outError)) {
                return false;
            }
            resources.scannerScratch.gateMask = nullptr;
            resources.scannerScratch.gateWidth = 0;
            resources.scannerScratch.gateHeight = 0;
            resources.scannerScratch.gateMaskHash = 0;
        }

        return true;
#endif
    }

    bool ensure_spatial_dir_scratch(Resources& resources, int width, int height, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)width;
        (void)height;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (width <= 0 || height <= 0) {
            outError = "spatial DIR scratch dimensions invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        Resources::DeviceSpatialDirScratch& scratch = resources.spatialDirScratch;
        const bool haveBase = (scratch.width == width && scratch.height == height && scratch.corrY && scratch.corrM && scratch.corrC);
        if (haveBase) {
            if (!ensure_shared_tmp_plane_locked(resources, width, height, cudaStreamOpaque, "shared tmp plane", outError)) {
                return false;
            }
            scratch.tmp = resources.sharedTmpPlane;
            return true;
        }

        if (scratch.corrY || scratch.corrM || scratch.corrC) {
            if (!retire_spatial_dir_scratch_locked(resources, scratch, cudaStreamOpaque, "spatial DIR scratch", outError)) {
                return false;
            }
        }

        const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
        const size_t bytes = total * sizeof(float);
        if (!allocate_scratch_device_ptr_locked(
                resources,
                scratch.corrY,
                bytes,
                cudaStreamOpaque,
                "spatial DIR corrY",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (!allocate_scratch_device_ptr_locked(
                resources,
                scratch.corrM,
                bytes,
                cudaStreamOpaque,
                "spatial DIR corrM",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (!allocate_scratch_device_ptr_locked(
                resources,
                scratch.corrC,
                bytes,
                cudaStreamOpaque,
                "spatial DIR corrC",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (!ensure_shared_tmp_plane_locked(resources, width, height, cudaStreamOpaque, "shared tmp plane", outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        scratch.tmp = resources.sharedTmpPlane;

        scratch.width = width;
        scratch.height = height;
        return true;
#endif
    }

    static bool retire_or_clear_gaussian_kernel_locked(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)kernel;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (kernel.weights) {
            if (kernel.sharedKernelId == 0) {
                const std::size_t bytes =
                    static_cast<std::size_t>(std::max(0, kernel.capacity)) * sizeof(float);
                if (!retire_ptr_locked(
                        resources,
                        kernel.weights,
                        bytes,
                        Resources::RetireKind::DeviceFree,
                        cudaStreamOpaque,
                        label,
                        outError)) {
                    return false;
                }
            }
            kernel.weights = nullptr;
        }
        kernel.radius = 0;
        kernel.sigma = 0.0f;
        kernel.capacity = 0;
        kernel.sharedKernelId = 0;
        return true;
#endif
    }

    static bool ensure_shared_gaussian_kernel(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        float sigma,
        int radius,
        SharedGaussianKind kind,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)kernel;
        (void)sigma;
        (void)radius;
        (void)kind;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        if (!sigmaOk || radius <= 0) {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            return retire_or_clear_gaussian_kernel_locked(
                resources,
                kernel,
                cudaStreamOpaque,
                label,
                outError);
        }

        SharedGaussianKey key{};
        std::uint64_t kernelId = 0;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            key = make_shared_gaussian_key(resources, kind, radius, sigma);
            kernelId = make_shared_gaussian_id(key);
            const bool same =
                kernel.weights &&
                kernel.sharedKernelId == kernelId &&
                kernel.radius == radius &&
                std::fabs(kernel.sigma - sigma) <= 1e-6f &&
                shared_gaussian_entry_matches_cache(key, kernel);
            if (same) {
                return true;
            }
        }

        std::vector<float> cpuWeights;
        build_gaussian_weights_cpu(radius, sigma, cpuWeights);
        if (cpuWeights.empty()) {
            outError = std::string(label ? label : "gaussian kernel")
                + " weights build failed";
            return false;
        }

        SharedGaussianEntry sharedEntry{};
        if (!acquire_shared_gaussian_entry(
                key,
                radius,
                sigma,
                cpuWeights,
                sharedEntry,
                outError)) {
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        key = make_shared_gaussian_key(resources, kind, radius, sigma);
        kernelId = make_shared_gaussian_id(key);
        if (kernelId != sharedEntry.id) {
            if (!acquire_shared_gaussian_entry(
                    key,
                    radius,
                    sigma,
                    cpuWeights,
                    sharedEntry,
                    outError)) {
                return false;
            }
            kernelId = sharedEntry.id;
        }

        const bool same =
            kernel.weights &&
            kernel.sharedKernelId == kernelId &&
            kernel.radius == radius &&
            std::fabs(kernel.sigma - sigma) <= 1e-6f &&
            shared_gaussian_entry_matches_cache(key, kernel);
        if (same) {
            return true;
        }

        if (!retire_or_clear_gaussian_kernel_locked(
                resources,
                kernel,
                cudaStreamOpaque,
                label,
                outError)) {
            return false;
        }

        kernel.weights = sharedEntry.weights;
        kernel.radius = sharedEntry.radius;
        kernel.sigma = sharedEntry.sigma;
        kernel.capacity = 0;
        kernel.sharedKernelId = sharedEntry.id;
        return true;
#endif
    }

    bool ensure_spatial_dir_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
        constexpr int kMaxRadius = 75;
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        const int radiusRaw = sigmaOk
            ? std::max(1, static_cast<int>(std::ceil(3.0f * sigma)))
            : 0;
        const int radius = std::min(radiusRaw, kMaxRadius);
        return ensure_shared_gaussian_kernel(
            resources,
            kernel,
            sigma,
            radius,
            SharedGaussianKind::SpatialDir,
            cudaStreamOpaque,
            "spatial DIR kernel",
            outError);
    }

    bool ensure_gaussian_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
        constexpr int kMaxRadius = 75;
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        const int radiusRaw = sigmaOk
            ? JuicerGaussian::scipy_gaussian_radius(sigma, 4.0f)
            : 0;
        const int radius = std::min(radiusRaw, kMaxRadius);
        return ensure_shared_gaussian_kernel(
            resources,
            kernel,
            sigma,
            radius,
            SharedGaussianKind::Standard,
            cudaStreamOpaque,
            "gaussian kernel",
            outError);
    }

    bool ensure_halation_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
        constexpr int kMaxRadius = 75;
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        const int radiusRaw = sigmaOk
            ? JuicerGaussian::scipy_gaussian_radius(sigma, 7.0f)
            : 0;
        const int radius = std::min(radiusRaw, kMaxRadius);
        return ensure_shared_gaussian_kernel(
            resources,
            kernel,
            sigma,
            radius,
            SharedGaussianKind::Halation,
            cudaStreamOpaque,
            "halation kernel",
            outError);
    }

    void purge_shared_gaussian_kernels_for_context(int deviceId, void* contextOpaque) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (deviceId < 0 || contextOpaque == nullptr) {
            return;
        }

        std::vector<float*> toFree;
        {
            SharedGaussianCacheState& cache = shared_gaussian_cache_state();
            std::lock_guard<std::mutex> lock(cache.mutex);
            for (auto it = cache.byKey.begin(); it != cache.byKey.end();) {
                if (it->first.deviceId == deviceId &&
                    it->first.contextOpaque == contextOpaque) {
                    if (it->second.weights) {
                        toFree.push_back(it->second.weights);
                    }
                    it = cache.byKey.erase(it);
                    continue;
                }
                ++it;
            }
        }

        if (toFree.empty()) {
            return;
        }

        int previousDevice = -1;
        const cudaError_t prevErr = cudaGetDevice(&previousDevice);
        const bool havePreviousDevice = (prevErr == cudaSuccess && previousDevice >= 0);
        const bool needRestore = havePreviousDevice && previousDevice != deviceId;
        (void)cudaSetDevice(deviceId);

        for (float* ptr : toFree) {
            if (ptr) {
                (void)cudaFree(ptr);
            }
        }

        if (needRestore) {
            (void)cudaSetDevice(previousDevice);
        }
#else
        (void)deviceId;
        (void)contextOpaque;
#endif
    }
