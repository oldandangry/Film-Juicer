#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <set>
#include <vector>

#include <nlohmann/json.hpp>

namespace JuicerTests {
    class CaseLog;
    inline CaseLog* active_log = nullptr;

    class CaseLog {
    public:
        explicit CaseLog(bool enabled = true)
            : previous_(active_log), path_(enabled ? output_path() : std::filesystem::path{}) {
            if (!path_.empty()) {
                stream_.open(path_, std::ios::out | std::ios::trunc);
            }
            if (!path_.empty() && !stream_)
                throw std::runtime_error("native case results output open");
            active_log = this;
        }
        CaseLog(const CaseLog&) = delete;
        CaseLog& operator=(const CaseLog&) = delete;
        ~CaseLog() noexcept {
            // Normal exits call finish(); stream teardown must preserve an
            // exception already unwinding from a failed case or output write.
            active_log = previous_;
        }

        // These wire fields retain schema order; independent membership and
        // event validation reject swapped fields without adding wrapper types.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        void event(std::string_view caller, std::string_view id, std::string_view event, std::string_view outcome = {}) {
            if (path_.empty())
                return;
            nlohmann::json record{{"caller", std::string(caller)}, {"id", std::string(id)}, {"event", std::string(event)}};
            if (!outcome.empty())
                record["outcome"] = outcome;
            stream_ << record.dump() << '\n';
            if (!stream_)
                throw std::runtime_error("native case results output write");
        }

        void finish() {
            if (path_.empty())
                return;
            stream_.flush();
            if (!stream_)
                throw std::runtime_error("native case results output flush");
        }

        void membership(std::string_view caller, const std::vector<std::string>& ids) {
            if (ids.empty() || std::set<std::string>(ids.begin(), ids.end()).size() != ids.size())
                throw std::runtime_error("empty or duplicate native cohort membership");
            if (path_.empty())
                return;
            const nlohmann::json record{{"caller", caller}, {"event", "membership"}, {"ids", ids}};
            stream_ << record.dump() << '\n';
            if (!stream_)
                throw std::runtime_error("native cohort output write");
        }

    private:
        static std::filesystem::path output_path() {
#ifdef _WIN32
            std::size_t length = 0;
            if (_wgetenv_s(&length, nullptr, 0, L"FJ_HOST_CASES") != 0)
                throw std::runtime_error("native case output environment read");
            if (length > 1) {
                std::wstring path(length, L'\0');
                if (_wgetenv_s(&length, path.data(), path.size(), L"FJ_HOST_CASES") != 0)
                    throw std::runtime_error("native case output environment read");
                path.resize(length - 1);
                return std::filesystem::path(path);
            }
#else
            if (const auto* path = std::getenv("FJ_HOST_CASES"); path && *path)
                return std::filesystem::path(path);
#endif
            return {};
        }

        CaseLog* previous_;
        std::filesystem::path path_;
        std::ofstream stream_;
    };

    class CaseEvent {
    public:
        // Caller and ID form the protocol key checked against intended rows.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        CaseEvent(std::string_view caller, std::string_view id) : caller_(caller), id_(id) {
            if (active_log)
                active_log->event(caller_, id_, "start");
        }
        CaseEvent(const CaseEvent&) = delete;
        CaseEvent& operator=(const CaseEvent&) = delete;
        ~CaseEvent() {
            if (active_log && !terminal_) {
                try {
                    active_log->event(caller_, id_, "terminal", "failed");
                } catch (...) {
                    // A missing terminal stays invalid; keep the original
                    // failure when even the best-effort failure write fails.
                    return;
                }
            }
        }
        void passed() {
            complete(true);
        }
        void complete(bool success) {
            if (active_log)
                active_log->event(caller_, id_, "terminal", success ? "passed" : "failed");
            terminal_ = true;
        }

    private:
        std::string caller_;
        std::string id_;
        bool terminal_ = false;
    };

    // The owning fixture/parameter table declares these IDs before actuation.
    // Runtime events remain separate, so a skipped loop row stays incomplete.
    class Cohort {
    public:
        Cohort(std::string caller, const std::vector<std::string>& ids) : caller_(std::move(caller)) {
            if (active_log)
                active_log->membership(caller_, ids);
        }
        Cohort(std::string caller, std::size_t count) : Cohort(std::move(caller), indices(count)) {}
        CaseEvent case_event(std::string_view id) const {
            return CaseEvent(caller_, id);
        }

    private:
        static std::vector<std::string> indices(std::size_t count) {
            std::vector<std::string> ids;
            ids.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
                ids.push_back(std::to_string(i));
            return ids;
        }
        std::string caller_;
    };

    // Listing preserves the same independently checked protocol key order.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    inline void list_case(std::string_view caller, std::string_view id) {
        const auto record = nlohmann::json{{"caller", std::string(caller)}, {"id", std::string(id)}};
        std::puts(record.dump().c_str());
    }

    inline bool list_requested(int argc, char** argv) {
        return argc > 1 && std::string_view(argv[argc - 1]) == "--list-cases";
    }

    template <typename Rows>
    void list_rows(std::string_view caller, const Rows& rows, std::string_view key = "id") {
        for (const auto& row : rows)
            list_case(caller, row.at(key).template get_ref<const std::string&>());
    }
} // namespace JuicerTests
