#include <cmath>
#include <cstdlib>
#include <iostream>

#include "portfolio_math.h"

namespace {

void expect_close(const char* name, float actual, float expected, float tolerance) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << name << " mismatch: actual=" << actual
                  << " expected=" << expected
                  << " tolerance=" << tolerance << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    {
        const float log_returns[] = {0.010f, -0.020f, 0.005f};
        const float positions[] = {0.0f, 0.0f, 0.0f};
        const float actions[] = {0.50f, 0.25f, -0.10f};
        const float long_margin_requirement[] = {1.0f / 30.0f, 1.0f / 20.0f, 1.0f / 10.0f};
        const float short_margin_requirement[] = {1.0f / 30.0f, 1.0f / 20.0f, 1.0f / 10.0f};
        constexpr float equity_before = 10'000.0f;
        constexpr float transaction_cost_bps = 1.0f;

        const forex_cuda::StepStats stats = forex_cuda::compute_portfolio_step(
            log_returns,
            1,
            positions,
            actions,
            long_margin_requirement,
            short_margin_requirement,
            equity_before,
            3,
            transaction_cost_bps,
            1.0f,
            1.0f
        );

        const float expected_weighted_log_return = -0.0005f;
        const float expected_turnover = 0.85f;
        const float expected_cost = equity_before * expected_turnover * 0.0001f;
        const float expected_equity =
            equity_before * std::exp(expected_weighted_log_return) - expected_cost;
        const float expected_reward = std::log(expected_equity / equity_before);
        const float expected_margin =
            equity_before * (0.50f / 30.0f + 0.25f / 20.0f + 0.10f / 10.0f);
        const float expected_margin_ratio = expected_margin / equity_before;
        const float expected_free_cash = expected_equity - expected_margin;

        expect_close("weighted_log_return", stats.weighted_log_return, expected_weighted_log_return, 1e-7f);
        expect_close("turnover", stats.turnover, expected_turnover, 1e-6f);
        expect_close("margin_used", stats.margin_used, expected_margin, 1e-4f);
        expect_close("margin_ratio", stats.margin_ratio, expected_margin_ratio, 1e-6f);
        expect_close("transaction_cost", stats.transaction_cost, expected_cost, 1e-4f);
        expect_close("equity_after", stats.equity_after, expected_equity, 1e-2f);
        expect_close("reward", stats.reward, expected_reward, 1e-6f);
        expect_close("free_cash", stats.free_cash, expected_free_cash, 1e-2f);
    }

    {
        const float log_returns[] = {0.01f, 0.01f, 0.01f};
        const float positions[] = {0.0f, 0.0f, 0.0f};
        const float actions[] = {1.0f, 1.0f, 1.0f};
        const float long_margin_requirement[] = {1.0f, 1.0f, 1.0f};
        const float short_margin_requirement[] = {1.0f, 1.0f, 1.0f};

        const forex_cuda::StepStats stats = forex_cuda::compute_portfolio_step(
            log_returns,
            1,
            positions,
            actions,
            long_margin_requirement,
            short_margin_requirement,
            100.0f,
            3,
            0.0f,
            1.0f,
            1.0f
        );

        expect_close("scaled_weighted_log_return", stats.weighted_log_return, 0.01f, 1e-6f);
        expect_close("scaled_turnover", stats.turnover, 1.0f, 1e-6f);
        expect_close("scaled_margin_used", stats.margin_used, 100.0f, 1e-5f);
        expect_close("scaled_free_cash", stats.free_cash, 100.0f * std::exp(0.01f) - 100.0f, 1e-4f);
    }

    std::cout << "portfolio_math tests passed\n";
    return EXIT_SUCCESS;
}
