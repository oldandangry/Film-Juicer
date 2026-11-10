#include <cstdio>
#include <exception>
#include <string>
#include <string_view>

#include "case_results.h"

#ifdef JUICER_PREPARED_VALUE_TEST

#ifdef _WIN32
extern "C" int fj_test_prepared_values_w(int, wchar_t**);
#else
extern "C" int fj_test_prepared_values(int, char**);
#endif

namespace {
    constexpr std::string_view kCaller = "prepared_values";
    constexpr std::string_view kCaseId = "complete";

#ifndef _WIN32
    int run_values(int argc, char** argv) {
        JuicerTests::CaseLog log;
        JuicerTests::Cohort cohort(std::string(kCaller), {std::string(kCaseId)});
        auto event = cohort.case_event(kCaseId);
        const int result = fj_test_prepared_values(argc, argv);
        event.complete(result == 0);
        log.finish();
        return result;
    }
#endif

#ifdef _WIN32
    int run_values(int argc, wchar_t** argv) {
        JuicerTests::CaseLog log;
        JuicerTests::Cohort cohort(std::string(kCaller), {std::string(kCaseId)});
        auto event = cohort.case_event(kCaseId);
        const int result = fj_test_prepared_values_w(argc, argv);
        event.complete(result == 0);
        log.finish();
        return result;
    }
#endif
} // namespace

#else

extern "C" int fj_test_output_color_values(void);

namespace {
    constexpr std::string_view kCaller = "output_color_values";
    constexpr std::string_view kCaseId = "complete";

    int run_values() {
        JuicerTests::CaseLog log;
        JuicerTests::Cohort cohort(std::string(kCaller), {std::string(kCaseId)});
        auto event = cohort.case_event(kCaseId);
        const int result = fj_test_output_color_values();
        event.complete(result == 0);
        log.finish();
        return result;
    }
} // namespace

#endif

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    try {
        if (argc > 1 && std::wstring_view(argv[argc - 1]) == L"--list-cases") {
#ifdef JUICER_PREPARED_VALUE_TEST
            JuicerTests::list_case("prepared_values", "complete");
#else
            JuicerTests::list_case("output_color_values", "complete");
#endif
            return std::fflush(stdout) == 0 ? 0 : 1;
        }
#ifdef JUICER_PREPARED_VALUE_TEST
        return run_values(argc, argv);
#else
        return run_values();
#endif
    } catch (const std::exception& error) {
        std::fprintf(stderr, "value probe failed: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("value probe failed with an unknown exception\n", stderr);
        return 1;
    }
}
#else
int main(int argc, char** argv) {
    try {
        if (JuicerTests::list_requested(argc, argv)) {
#ifdef JUICER_PREPARED_VALUE_TEST
            JuicerTests::list_case("prepared_values", "complete");
#else
            JuicerTests::list_case("output_color_values", "complete");
#endif
            return std::fflush(stdout) == 0 ? 0 : 1;
        }
#ifdef JUICER_PREPARED_VALUE_TEST
        return run_values(argc, argv);
#else
        return run_values();
#endif
    } catch (const std::exception& error) {
        std::fprintf(stderr, "value probe failed: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("value probe failed with an unknown exception\n", stderr);
        return 1;
    }
}
#endif
