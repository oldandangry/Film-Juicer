#include <cstdio>
#include <exception>

#include <gtest/gtest.h>

#include "case_results.h"

int main(int argc, char** argv) try {
    ::testing::InitGoogleTest(&argc, argv);
    if (::testing::GTEST_FLAG(list_tests))
        return RUN_ALL_TESTS();
    JuicerTests::CaseLog log;
    const int result = RUN_ALL_TESTS();
    log.finish();
    return result;
} catch (const std::exception& error) {
    std::fprintf(stderr, "Native test output: %s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("Native test output failed\n", stderr);
    return 1;
}
