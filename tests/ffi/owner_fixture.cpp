#include "owner_fixture.h"

#include <array>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaFailure.h"
#include "Logging.h"
#include "native_path.h"
#include "preparation_fixture.h"

namespace JuicerTestSupport {
    namespace {
        FjStatus internal_failure() noexcept {
            return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        }

        [[noreturn]] void fail(FjStatus status, std::string_view message) {
            if (status.category == FJ_STATUS_ALLOCATION_FAILURE) {
                throw std::bad_alloc();
            }
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, message);
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }

        void report_cleanup(FjStatus result) noexcept {
            if (result.category != FJ_STATUS_SUCCESS) {
                try {
                    JTRACE("RUNTIME", "process runtime cleanup failed category=" + std::to_string(result.category));
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                }
            }
        }
    } // namespace

    PreparedStateHold::PreparedStateHold(FjPreparedState* owner, AdoptTag) noexcept : owner_(owner) {}
    PreparedStateHold::PreparedStateHold(const FjAdmission& admission, AdoptTag) noexcept
        : owner_(admission.state), input_identity_(admission.input_identity), build_(admission.build) {}
    PreparedStateHold PreparedStateHold::take_admission(FjAdmission& admission) {
        if (admission.kind != FJ_ADMISSION_READY || !admission.state)
            throw std::runtime_error("cannot retain a non-Ready admission");
        PreparedStateHold held(admission, AdoptTag{});
        admission.state = nullptr;
        return held;
    }
    PreparedStateHold::PreparedStateHold(PreparedStateHold&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), input_identity_(std::exchange(other.input_identity_, 0)),
          build_(std::exchange(other.build_, 0)) {}
    PreparedStateHold& PreparedStateHold::operator=(PreparedStateHold&& other) noexcept {
        if (this != &other) {
            release();
            owner_ = std::exchange(other.owner_, nullptr);
            input_identity_ = std::exchange(other.input_identity_, 0);
            build_ = std::exchange(other.build_, 0);
        }
        return *this;
    }
    PreparedStateHold::~PreparedStateHold() {
        release();
    }
    void PreparedStateHold::release() noexcept {
        FjPreparedState* owner = std::exchange(owner_, nullptr);
        if (owner) {
            FjErrorBuffer ignored{nullptr, 0, 0};
            (void)fj_test_prepared_state_release(owner, &ignored);
        }
    }
    const FjPreparedState* PreparedStateHold::borrow() const noexcept {
        return owner_;
    }
    FjStatus PreparedStateHold::view(FjPreparedStateView& out, FjErrorBuffer& error) const {
        return fj_test_prepared_state_view(owner_, &out, &error);
    }
    FjStatus PreparedStateHold::frame_policy(const FjExtent& extent, FjPreparedFramePolicy& out, FjErrorBuffer& error) const {
        return fj_test_prepared_state_frame_policy(owner_, &extent, &out, &error);
    }
    FjCallCompletionResult PreparedStateHold::render(const FjAssets* assets, FjCudaCallAccess access, const FjFrame& frame, const FjPreparedFramePolicy& policy, const FjInstance* instance, std::uint64_t token, FjAbortCallback abort, FjErrorBuffer& error) const {
        return fj_test_prepared_state_render(assets, owner_, access, &frame, &policy, instance, token, abort, &error);
    }
    FjCallCompletionResult PreparedStateHold::reject_missing_instance(const FjAssets* assets, FjCudaCallAccess access, const FjFrame& frame, FjAbortCallback abort, FjErrorBuffer& error) const {
        return fj_test_prepared_unavailable(assets, owner_, FJ_UNAVAILABLE_INSTANCE, access, &frame, abort, &error);
    }
    std::uint64_t PreparedStateHold::input_identity() const noexcept {
        return input_identity_;
    }
    std::uint64_t PreparedStateHold::build() const noexcept {
        return build_;
    }

    FjStatus prepare_state(const FjAssets* assets, const PreparationInput& input, std::optional<PreparedStateHold>& out, FjErrorBuffer& error) {
        out.reset();
        const FjPreparationParams record = input.record();
        FjPreparedState* owner = nullptr;
        const FjStatus status = fj_test_prepared_state_create(assets, &record, &owner, &error);
        if (status.category != FJ_STATUS_SUCCESS) {
            if (owner) {
                FjErrorBuffer ignored{nullptr, 0, 0};
                (void)fj_test_prepared_state_release(owner, &ignored);
            }
            return status;
        }
        if (!owner)
            return internal_failure();
        out.emplace(PreparedStateHold(owner, PreparedStateHold::AdoptTag{}));
        return status;
    }
    FjCallCompletionResult reject_missing_render_state(const FjAssets* assets, FjCudaCallAccess access, const FjFrame& frame, FjAbortCallback abort, FjErrorBuffer& error) {
        return fj_test_prepared_unavailable(assets, nullptr, FJ_UNAVAILABLE_STATE, access, &frame, abort, &error);
    }

    HostWriteFixture::HostWriteFixture(FjHostWrite* owner) noexcept : owner_(owner) {}
    HostWriteFixture::HostWriteFixture(HostWriteFixture&& other) noexcept : owner_(std::exchange(other.owner_, nullptr)) {}
    HostWriteFixture& HostWriteFixture::operator=(HostWriteFixture&& other) noexcept {
        if (this != &other) {
            fail_if_active();
            owner_ = std::exchange(other.owner_, nullptr);
        }
        return *this;
    }
    HostWriteFixture::~HostWriteFixture() {
        fail_if_active();
    }
    FjStatus HostWriteFixture::finish(std::uint32_t failure, FjStringView diagnostic, std::uint32_t& refresh, FjErrorBuffer& error) noexcept {
        refresh = 0;
        FjHostWrite* owner = std::exchange(owner_, nullptr);
        if (!owner)
            return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
        return fj_test_host_write_finish(owner, failure, diagnostic, &refresh, &error);
    }
    void HostWriteFixture::fail_if_active() noexcept {
        FjHostWrite* owner = std::exchange(owner_, nullptr);
        if (!owner)
            return;
        static constexpr char kDiagnostic[] = "host write abandoned before completion";
        std::uint32_t ignored_refresh = 0;
        FjErrorBuffer ignored{nullptr, 0, 0};
        (void)fj_test_host_write_finish(owner, FJ_CAPTURE_HOST_FAILURE, {kDiagnostic, sizeof(kDiagnostic) - 1}, &ignored_refresh, &ignored);
    }

    InstanceFixture::InstanceFixture() {
        FjInstance* created = nullptr;
        FjErrorBuffer error{nullptr, 0, 0};
        const FjStatus status = fj_test_instance_create(&created, &error);
        if (status.category != FJ_STATUS_SUCCESS || !created) {
            if (created) {
                FjErrorBuffer ignored{nullptr, 0, 0};
                (void)fj_test_instance_destroy(created, &ignored);
            }
            if (status.category == FJ_STATUS_ALLOCATION_FAILURE)
                throw std::bad_alloc();
            throw CreationFailure{};
        }
        owner_ = created;
    }
    InstanceFixture::~InstanceFixture() {
        FjInstance* owner = std::exchange(owner_, nullptr);
        if (owner) {
            FjErrorBuffer ignored{nullptr, 0, 0};
            (void)fj_test_instance_destroy(owner, &ignored);
        }
    }
    const FjInstance* InstanceFixture::borrow() const noexcept {
        return owner_;
    }
    FjStatus InstanceFixture::bind_identity(std::uint64_t seed, std::uint64_t token, FjErrorBuffer& error) {
        return fj_test_instance_bind_identity(owner_, seed, token, &error);
    }
    FjStatus InstanceFixture::begin_capture(std::uint32_t mode, FjCaptureStart& out, FjErrorBuffer& error) {
        out = {};
        return fj_test_instance_begin_capture(owner_, mode, &out, &error);
    }
    FjStatus InstanceFixture::capture_is_current(std::uint64_t ticket, std::uint32_t& current, FjErrorBuffer& error) const {
        current = 0;
        return fj_test_instance_capture_is_current(owner_, ticket, &current, &error);
    }
    FjStatus InstanceFixture::begin_snapshot(std::uint64_t ticket, FjSnapshotStart& out, FjErrorBuffer& error) {
        out = {};
        return fj_test_instance_begin_snapshot(owner_, ticket, &out, &error);
    }
    FjStatus InstanceFixture::install(std::uint64_t ticket, std::uint64_t revision, const PreparationInput& input, std::uint64_t identity, std::uint32_t& update, FjErrorBuffer& error) {
        update = FJ_UPDATE_NONE;
        const FjPreparationParams record = input.record();
        return fj_test_instance_install(owner_, ticket, revision, &record, identity, &update, &error);
    }
    FjStatus InstanceFixture::invalidate(std::uint64_t ticket, std::uint32_t failure, FjStringView diagnostic, std::uint32_t& update, FjErrorBuffer& error) {
        update = FJ_UPDATE_NONE;
        return fj_test_instance_invalidate(owner_, ticket, failure, diagnostic, &update, &error);
    }
    FjStatus InstanceFixture::admit(const FjAssets* assets, std::uint32_t& kind, std::optional<PreparedStateHold>& out, FjErrorBuffer& error) {
        kind = FJ_ADMISSION_NONE;
        out.reset();
        FjAdmission admission{};
        const FjStatus status = fj_test_instance_admit(owner_, assets, &admission, &error);
        kind = admission.kind;
        if (status.category == FJ_STATUS_SUCCESS && admission.kind == FJ_ADMISSION_READY && admission.state) {
            out.emplace(PreparedStateHold(admission, PreparedStateHold::AdoptTag{}));
            admission.state = nullptr;
        } else if (admission.state) {
            FjErrorBuffer ignored{nullptr, 0, 0};
            (void)fj_test_prepared_state_release(admission.state, &ignored);
        }
        if (status.category == FJ_STATUS_SUCCESS && admission.kind == FJ_ADMISSION_READY && !out)
            return internal_failure();
        return status;
    }
    FjStatus InstanceFixture::begin_host_write(std::uint64_t ticket, std::uint32_t& kind, std::optional<HostWriteFixture>& out, FjErrorBuffer& error) {
        kind = FJ_HOST_WRITE_NONE;
        out.reset();
        FjHostWrite* owner = nullptr;
        const FjStatus status = fj_test_instance_begin_host_write(owner_, ticket, &owner, &kind, &error);
        if (status.category == FJ_STATUS_SUCCESS && kind == FJ_HOST_WRITE_STARTED && owner) {
            out = HostWriteFixture(owner);
        } else if (owner) {
            static constexpr char kDiagnostic[] = "native host-write result was not Started";
            std::uint32_t ignored_refresh = 0;
            FjErrorBuffer ignored{nullptr, 0, 0};
            (void)fj_test_host_write_finish(owner, FJ_CAPTURE_HOST_FAILURE, {kDiagnostic, sizeof(kDiagnostic) - 1}, &ignored_refresh, &ignored);
        }
        if (status.category == FJ_STATUS_SUCCESS && kind == FJ_HOST_WRITE_STARTED && !out)
            return internal_failure();
        return status;
    }

    RuntimeFixture::RuntimeFixture(RuntimeFixture&& other) noexcept : owner_(std::exchange(other.owner_, nullptr)) {}
    RuntimeFixture& RuntimeFixture::operator=(RuntimeFixture&& other) noexcept {
        if (this != &other) {
            if (owner_) {
                const FjStatus status = close();
                if (status.category != FJ_STATUS_SUCCESS) {
                    report_cleanup(status);
                    return *this;
                }
            }
            owner_ = std::exchange(other.owner_, nullptr);
        }
        return *this;
    }
    RuntimeFixture::~RuntimeFixture() noexcept {
        if (owner_)
            report_cleanup(close());
    }
    void RuntimeFixture::create(const std::filesystem::path& resource_root) {
        if (owner_)
            fail({FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "process runtime is already initialized");
        JuicerAssets::NativePathArgument path(resource_root);
        std::array<char, 1024> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        FjRuntime* owner = nullptr;
        const FjStatus status = fj_test_runtime_create(path.view(), &owner, &error);
        if (status.category != FJ_STATUS_SUCCESS) {
            if (owner) {
                FjErrorBuffer ignored{nullptr, 0, 0};
                (void)fj_test_runtime_destroy(owner, &ignored);
            }
            fail(status, {message.data(), error.length});
        }
        if (!owner)
            fail(internal_failure(), "Rust runtime creation returned no owner");
        owner_ = owner;
    }
    const FjRuntime* RuntimeFixture::raw() const noexcept {
        return owner_;
    }
    FjRuntimeView RuntimeFixture::borrow() const {
        std::array<char, 1024> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        FjRuntimeView view{};
        const FjStatus status = borrow(view, &error);
        if (status.category != FJ_STATUS_SUCCESS)
            fail(status, {message.data(), error.length});
        return view;
    }
    FjStatus RuntimeFixture::borrow(FjRuntimeView& out, FjErrorBuffer* error) const noexcept {
        out = {};
        if (!owner_)
            return JuicerCuda::write_status({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "process runtime is unavailable", error);
        return fj_test_runtime_borrow(owner_, &out, error);
    }
    FjStatus RuntimeFixture::check_close(FjErrorBuffer* error) const noexcept {
        if (!owner_)
            return JuicerCuda::write_status({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}, error);
        return fj_test_runtime_check_close(owner_, error);
    }
    FjStatus RuntimeFixture::shutdown(FjErrorBuffer* error) noexcept {
        if (!owner_)
            return JuicerCuda::write_status({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}, error);
        return fj_test_runtime_shutdown(owner_, error);
    }
    FjStatus RuntimeFixture::close(FjErrorBuffer* error) noexcept {
        if (!owner_)
            return JuicerCuda::write_status({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}, error);
        std::array<char, 1024> message{};
        FjErrorBuffer precheck{message.data(), message.size(), 0};
        const FjStatus checked = fj_test_runtime_check_close(owner_, &precheck);
        if (checked.category != FJ_STATUS_SUCCESS) {
            return JuicerCuda::write_status(checked, {message.data(), precheck.length}, error);
        }
        FjRuntime* owner = std::exchange(owner_, nullptr);
        return fj_test_runtime_destroy(owner, error);
    }
} // namespace JuicerTestSupport
