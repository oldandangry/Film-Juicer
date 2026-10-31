#pragma once

#include <array>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "juicer_cuda_call_scope.h"
#include "juicer_test_api.h"

namespace JuicerTestSupport {

    namespace detail {

        template <class Function, class Result>
        struct HostPreparationInvocation {
            Function* function = nullptr;
            std::optional<Result> result;
            std::exception_ptr exception;
            bool invoked = false;
        };

        template <class Function, class Result>
        FjStatus invoke_host_preparation(
            void* user,
            const FjHostPreparation* preparation,
            FjErrorBuffer*) noexcept {
            auto& invocation = *static_cast<HostPreparationInvocation<Function, Result>*>(user);
            try {
                if (invocation.invoked || !preparation) {
                    throw std::runtime_error("invalid host-preparation fixture callback");
                }
                invocation.invoked = true;
                invocation.result.emplace(std::invoke(*invocation.function, preparation));
                return {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
            } catch (...) {
                invocation.exception = std::current_exception();
                return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
            }
        }

    } // namespace detail

    // The Rust test facade owns any callback backing until this synchronous
    // body returns. Preserve its result separately from the C status so a
    // legitimate false cache result remains false instead of becoming success.
    template <class Function>
    auto with_host_preparation(Function&& function) {
        using Callable = std::remove_reference_t<Function>;
        using Result = std::invoke_result_t<Callable&, const FjHostPreparation*>;
        static_assert(!std::is_void_v<Result>);

        detail::HostPreparationInvocation<Callable, Result> invocation{
            std::addressof(function), std::nullopt, nullptr, false};
        std::array<char, 512> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        const FjStatus status = fj_test_with_host_preparation(
            &invocation,
            &detail::invoke_host_preparation<Callable, Result>,
            &error);

        if (invocation.exception) {
            std::rethrow_exception(invocation.exception);
        }
        if (!invocation.invoked || !invocation.result) {
            throw std::runtime_error(
                error.length ? std::string(diagnostic.data(), error.length)
                             : "Rust host-preparation fixture did not invoke its body");
        }
        if (status.category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error(
                error.length ? std::string(diagnostic.data(), error.length)
                             : "Rust host-preparation fixture failed");
        }
        return std::move(*invocation.result);
    }

} // namespace JuicerTestSupport
