#pragma once

#include "juicer_legacy_api.h"

#include <cstdint>
#include <optional>

namespace JuicerTestSupport {
    class PublicationObserver;
}

struct ParamSnapshot;
struct FjCudaCallAccess;
struct FjPreparedFramePolicy;
struct FjSubmissionAccess;
struct FjCallCompletionResult;

// Profile-key strings borrow the snapshot through the synchronous create call.
FjPreparationParams preparation_params(const ParamSnapshot&);

class PreparedState {
public:
    PreparedState(const PreparedState&) = delete;
    PreparedState& operator=(const PreparedState&) = delete;

    PreparedState(PreparedState&& other) noexcept;
    PreparedState& operator=(PreparedState&& other) noexcept;
    ~PreparedState();

    // The pointer-bearing view is valid only while this holder remains alive.
    FjStatus view(FjPreparedStateView& out, FjErrorBuffer& error) const;

    FjStatus frame_policy(const FjExtent&, FjPreparedFramePolicy&, FjErrorBuffer&) const;
    FjCallCompletionResult render(
        FjAssets*, FjCudaCallAccess, const FjFrame&, const FjPreparedFramePolicy&, FjSubmissionAccess, FjAbortCallback, FjErrorBuffer&) const;
    FjCallCompletionResult reject_missing_instance(
        FjAssets*, FjCudaCallAccess, const FjFrame&, FjAbortCallback, FjErrorBuffer&) const;
    std::uint64_t input_identity() const noexcept;
    std::uint64_t build() const noexcept;

private:
    struct AdoptTag {};

    PreparedState(FjPreparedState* owner, AdoptTag) noexcept;
    PreparedState(const FjAdmission& admission, AdoptTag) noexcept;
    void release() noexcept;

    FjPreparedState* owner_;
    std::uint64_t inputIdentity_ = 0;
    std::uint64_t build_ = 0;

    friend FjStatus prepare_state(
        FjAssets*, const ParamSnapshot&, std::optional<PreparedState>&, FjErrorBuffer&);
    friend class RenderInstance;
    friend class JuicerTestSupport::PublicationObserver;
};

class HostWrite {
public:
    HostWrite(const HostWrite&) = delete;
    HostWrite& operator=(const HostWrite&) = delete;
    HostWrite(HostWrite&& other) noexcept;
    HostWrite& operator=(HostWrite&& other) noexcept;
    ~HostWrite();

    FjStatus finish(
        std::uint32_t failureKind,
        FjStringView diagnostic,
        std::uint32_t& outRefresh,
        FjErrorBuffer& error) noexcept;

private:
    explicit HostWrite(FjHostWrite* owner) noexcept;
    void fail_if_active() noexcept;

    FjHostWrite* owner_ = nullptr;

    friend class RenderInstance;
};

// Sole native owner of one Rust publication/admission instance. Callback
// exclusion is required before destruction; admitted PreparedState holds may
// outlive this owner.
class RenderInstance {
public:
    // The OFX adapter maps creation rejection; this core holder has no OFXS link.
    struct CreationFailure {};
    RenderInstance();
    RenderInstance(const RenderInstance&) = delete;
    RenderInstance& operator=(const RenderInstance&) = delete;
    RenderInstance(RenderInstance&&) = delete;
    RenderInstance& operator=(RenderInstance&&) = delete;
    ~RenderInstance();

    FjStatus begin_capture(std::uint32_t mode, FjCaptureStart& out, FjErrorBuffer& error);
    FjStatus capture_is_current(std::uint64_t ticket, std::uint32_t& outCurrent, FjErrorBuffer& error) const;
    FjStatus begin_snapshot(std::uint64_t ticket, FjSnapshotStart& out, FjErrorBuffer& error);
    FjStatus install(
        std::uint64_t ticket,
        std::uint64_t revision,
        const ParamSnapshot& snapshot,
        std::uint64_t inputIdentity,
        std::uint32_t& outUpdate,
        FjErrorBuffer& error);
    FjStatus invalidate(
        std::uint64_t ticket,
        std::uint32_t failureKind,
        FjStringView diagnostic,
        std::uint32_t& outUpdate,
        FjErrorBuffer& error);
    FjStatus admit(
        FjAssets* assets,
        std::uint32_t& outKind,
        std::optional<PreparedState>& outState,
        FjErrorBuffer& error);
    FjStatus begin_host_write(
        std::uint64_t ticket,
        std::uint32_t& outKind,
        std::optional<HostWrite>& outWrite,
        FjErrorBuffer& error);

private:
    friend class JuicerTestSupport::PublicationObserver;
    FjInstance* owner_ = nullptr;
};

FjStatus prepare_state(
    FjAssets* assets,
    const ParamSnapshot& snapshot,
    std::optional<PreparedState>& out,
    FjErrorBuffer& error);

FjCallCompletionResult reject_missing_render_state(
    FjAssets*, FjCudaCallAccess, const FjFrame&, FjAbortCallback, FjErrorBuffer&);
