#pragma once

#include <memory>

#include <gtest/gtest.h>

#include "case_results.h"

namespace JuicerTests {
    inline int gtest_failures() {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        if (!info)
            throw std::runtime_error("row cohort requires an active GoogleTest case");
        int failures = 0;
        const auto* result = info->result();
        for (int i = 0; i < result->total_part_count(); ++i)
            failures += result->GetTestPartResult(i).failed() ? 1 : 0;
        return failures;
    }

    inline std::string gtest_context(std::string_view cohort) {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        if (!info)
            throw std::runtime_error("row cohort requires an active GoogleTest case");
        return std::string(info->test_suite_name()) + "." + info->name() + "/" + std::string(cohort);
    }

    class GtestCase {
    public:
        GtestCase(const Cohort& cohort, std::string_view id)
            : event_(cohort.case_event(id)), failures_(gtest_failures()) {}
        void passed() {
            event_.complete(gtest_failures() == failures_);
        }

    private:
        CaseEvent event_;
        int failures_;
    };

    class GtestCohort {
    public:
        GtestCohort(std::string_view name, const std::vector<std::string>& ids)
            : cohort_(gtest_context(name), ids) {}
        GtestCohort(std::string_view name, std::size_t count)
            : cohort_(gtest_context(name), count) {}
        GtestCase case_event(std::string_view id) const {
            return GtestCase(cohort_, id);
        }
        std::unique_ptr<GtestCase> deferred_case(std::string_view id) const {
            return std::make_unique<GtestCase>(cohort_, id);
        }

    private:
        Cohort cohort_;
    };
} // namespace JuicerTests
