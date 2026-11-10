#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>

#include "juicer_test_api.h"

namespace JuicerTestSupport {
    struct PreparationInput;
    class InstanceFixture;

    // Views and render callbacks borrow this complete owner synchronously.
    class PreparedStateHold final {
    public:
        PreparedStateHold(const PreparedStateHold&) = delete;
        PreparedStateHold& operator=(const PreparedStateHold&) = delete;
        PreparedStateHold(PreparedStateHold&& other) noexcept;
        PreparedStateHold& operator=(PreparedStateHold&& other) noexcept;
        ~PreparedStateHold();

        // Consumes the Ready state slot returned by the real admission facade.
        static PreparedStateHold take_admission(FjAdmission& admission);

        const FjPreparedState* borrow() const noexcept;
        FjStatus view(FjPreparedStateView& out, FjErrorBuffer& error) const;
        FjStatus frame_policy(const FjExtent& extent, FjPreparedFramePolicy& out, FjErrorBuffer& error) const;
        FjCallCompletionResult render(const FjAssets* assets, FjCudaCallAccess access, const FjFrame& frame, const FjPreparedFramePolicy& policy, const FjInstance* instance, std::uint64_t token, FjAbortCallback abort, FjErrorBuffer& error) const;
        FjCallCompletionResult reject_missing_instance(const FjAssets* assets, FjCudaCallAccess access, const FjFrame& frame, FjAbortCallback abort, FjErrorBuffer& error) const;
        std::uint64_t input_identity() const noexcept;
        std::uint64_t build() const noexcept;

    private:
        struct AdoptTag {};
        explicit PreparedStateHold(FjPreparedState* owner, AdoptTag) noexcept;
        explicit PreparedStateHold(const FjAdmission& admission, AdoptTag) noexcept;
        void release() noexcept;
        FjPreparedState* owner_ = nullptr;
        std::uint64_t input_identity_ = 0;
        std::uint64_t build_ = 0;
        friend class InstanceFixture;
        friend FjStatus prepare_state(const FjAssets*, const PreparationInput&, std::optional<PreparedStateHold>&, FjErrorBuffer&);
    };

    class HostWriteFixture final {
    public:
        HostWriteFixture(const HostWriteFixture&) = delete;
        HostWriteFixture& operator=(const HostWriteFixture&) = delete;
        HostWriteFixture(HostWriteFixture&& other) noexcept;
        HostWriteFixture& operator=(HostWriteFixture&& other) noexcept;
        ~HostWriteFixture();
        FjStatus finish(std::uint32_t failure, FjStringView diagnostic, std::uint32_t& refresh, FjErrorBuffer& error) noexcept;

    private:
        explicit HostWriteFixture(FjHostWrite* owner) noexcept;
        void fail_if_active() noexcept;
        FjHostWrite* owner_ = nullptr;
        friend class InstanceFixture;
    };

    // All callbacks are excluded before destruction; admitted holds may outlive it.
    class InstanceFixture final {
    public:
        struct CreationFailure {};
        InstanceFixture();
        InstanceFixture(const InstanceFixture&) = delete;
        InstanceFixture& operator=(const InstanceFixture&) = delete;
        InstanceFixture(InstanceFixture&&) = delete;
        InstanceFixture& operator=(InstanceFixture&&) = delete;
        ~InstanceFixture();
        const FjInstance* borrow() const noexcept;
        FjStatus bind_identity(std::uint64_t seed, std::uint64_t token, FjErrorBuffer& error);
        FjStatus begin_capture(std::uint32_t mode, FjCaptureStart& out, FjErrorBuffer& error);
        FjStatus capture_is_current(std::uint64_t ticket, std::uint32_t& current, FjErrorBuffer& error) const;
        FjStatus begin_snapshot(std::uint64_t ticket, FjSnapshotStart& out, FjErrorBuffer& error);
        FjStatus install(std::uint64_t ticket, std::uint64_t revision, const PreparationInput& input, std::uint64_t identity, std::uint32_t& update, FjErrorBuffer& error);
        FjStatus invalidate(std::uint64_t ticket, std::uint32_t failure, FjStringView diagnostic, std::uint32_t& update, FjErrorBuffer& error);
        FjStatus admit(const FjAssets* assets, std::uint32_t& kind, std::optional<PreparedStateHold>& out, FjErrorBuffer& error);
        FjStatus begin_host_write(std::uint64_t ticket, std::uint32_t& kind, std::optional<HostWriteFixture>& out, FjErrorBuffer& error);

    private:
        FjInstance* owner_ = nullptr;
    };

    // A refused close retains the Rust allocation and its native resources.
    // Callbacks must be excluded before destruction; borrowed views never own it.
    class RuntimeFixture final {
    public:
        RuntimeFixture() noexcept = default;
        RuntimeFixture(const RuntimeFixture&) = delete;
        RuntimeFixture& operator=(const RuntimeFixture&) = delete;
        RuntimeFixture(RuntimeFixture&& other) noexcept;
        RuntimeFixture& operator=(RuntimeFixture&& other) noexcept;
        ~RuntimeFixture() noexcept;
        void create(const std::filesystem::path& resource_root);
        const FjRuntime* raw() const noexcept;
        FjRuntimeView borrow() const;
        FjStatus borrow(FjRuntimeView& out, FjErrorBuffer* error = nullptr) const noexcept;
        FjStatus check_close(FjErrorBuffer* error = nullptr) const noexcept;
        FjStatus shutdown(FjErrorBuffer* error = nullptr) noexcept;
        FjStatus close(FjErrorBuffer* error = nullptr) noexcept;

    private:
        FjRuntime* owner_ = nullptr;
    };

    FjStatus prepare_state(const FjAssets* assets, const PreparationInput& input, std::optional<PreparedStateHold>& out, FjErrorBuffer& error);
    FjCallCompletionResult reject_missing_render_state(const FjAssets* assets, FjCudaCallAccess access, const FjFrame& frame, FjAbortCallback abort, FjErrorBuffer& error);
} // namespace JuicerTestSupport
