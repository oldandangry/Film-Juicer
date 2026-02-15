#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace WorkingStateSharing {

struct WorkingStateCorePayload;

struct WorkingStateCoreShared {
    std::uint64_t keyHash = 0;
    std::uint64_t identity = 0;
    std::shared_ptr<const WorkingStateCorePayload> payload;
};

struct AcquireCoreSharedResult {
    std::shared_ptr<WorkingStateCoreShared> sharedCore;
    std::uint64_t keyHash = 0;
    std::uint64_t identity = 0;
    std::uint32_t cacheEntries = 0;
    bool hit = false;
    bool inserted = false;
    bool payloadPresent = false;
    bool payloadBackfilled = false;
};

class WorkingStateCoreSharedCache final {
public:
    static WorkingStateCoreSharedCache& instance() noexcept {
        static WorkingStateCoreSharedCache cache;
        return cache;
    }

    AcquireCoreSharedResult acquire_or_create(
        std::uint64_t keyHash,
        std::shared_ptr<const WorkingStateCorePayload> insertPayload = nullptr)
    {
        AcquireCoreSharedResult out{};
        out.keyHash = keyHash;
        if (keyHash == 0) {
            return out;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        ++touchSequence_;
        prune_expired_locked();

        auto it = entries_.find(keyHash);
        if (it != entries_.end()) {
            std::shared_ptr<WorkingStateCoreShared> shared = it->second.shared.lock();
            if (shared) {
                if (!shared->payload && insertPayload) {
                    shared->payload = std::move(insertPayload);
                    out.payloadBackfilled = true;
                }
                it->second.lastTouchSequence = touchSequence_;
                out.sharedCore = std::move(shared);
                out.identity = out.sharedCore->identity;
                out.hit = true;
                out.payloadPresent = (out.sharedCore->payload != nullptr);
                out.cacheEntries = static_cast<std::uint32_t>(entries_.size());
                return out;
            }
            entries_.erase(it);
        }

        auto created = std::make_shared<WorkingStateCoreShared>();
        created->keyHash = keyHash;
        created->identity = ++identitySequence_;
        created->payload = std::move(insertPayload);

        CacheEntry entry{};
        entry.shared = created;
        entry.lastTouchSequence = touchSequence_;
        entries_[keyHash] = std::move(entry);
        trim_to_cap_locked();

        out.sharedCore = std::move(created);
        out.identity = out.sharedCore->identity;
        out.inserted = true;
        out.payloadPresent = (out.sharedCore->payload != nullptr);
        out.cacheEntries = static_cast<std::uint32_t>(entries_.size());
        return out;
    }

private:
    struct CacheEntry {
        std::weak_ptr<WorkingStateCoreShared> shared;
        std::uint64_t lastTouchSequence = 0;
    };

    static constexpr std::size_t kMaxEntries = 256;

    void prune_expired_locked() {
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->second.shared.expired()) {
                it = entries_.erase(it);
            }
            else {
                ++it;
            }
        }
    }

    void trim_to_cap_locked() {
        if (entries_.size() <= kMaxEntries) {
            return;
        }
        while (entries_.size() > kMaxEntries) {
            auto victim = std::min_element(
                entries_.begin(),
                entries_.end(),
                [](const auto& a, const auto& b) {
                    return a.second.lastTouchSequence < b.second.lastTouchSequence;
                });
            if (victim == entries_.end()) {
                break;
            }
            entries_.erase(victim);
        }
    }

    std::mutex mutex_;
    std::unordered_map<std::uint64_t, CacheEntry> entries_;
    std::uint64_t touchSequence_ = 0;
    std::uint64_t identitySequence_ = 0;
};

inline AcquireCoreSharedResult acquire_or_create_shared_core(
    std::uint64_t keyHash,
    std::shared_ptr<const WorkingStateCorePayload> insertPayload = nullptr)
{
    return WorkingStateCoreSharedCache::instance().acquire_or_create(keyHash, std::move(insertPayload));
}

} // namespace WorkingStateSharing
