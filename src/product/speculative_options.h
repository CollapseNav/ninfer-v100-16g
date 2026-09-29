#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        // The sm_70 width-6+ target-verify regression (draft window >= 5 drifting off the greedy
        // argmax) tracked back to the dedicated Volta small_t_i8/bf16 verify kernels dropped during
        // the DFlash2 merge, and both are restored (small_t_i8_volta.cuh, small_t_bf16_volta.cuh).
        //
        // MEASURED AND DISPROVEN: the follow-on claim that "every draft window through 7 is
        // bit-exact against --spec none" does not hold on this artifact. Two prompts x two KV
        // dtypes x three windows, every other setting identical within a row (prompt tokens, KV
        // capacity, token count, finish reason; --greedy confirmed in the log):
        //
        //   prompt P / int8   K=1 DIFF@11   K=3 DIFF@11   K=7 MATCH
        //   prompt P / bf16   K=1 DIFF@11   K=3 DIFF@11   K=7 DIFF@11
        //   prompt Q / int8   K=1 DIFF@25   K=3 DIFF@25   K=7 DIFF@25
        //   prompt Q / bf16   K=1 MATCH     K=3 MATCH     K=7 MATCH
        //
        // So the window is not the variable: on P/bf16 all three windows diverge and on Q/bf16 all
        // three match. Where arms diverge, all speculative windows agree with each other, and the
        // no-spec arm switches sides with the KV dtype, with the same two continuations appearing
        // in every configuration -- i.e. a near-tie in the top-2 logits decided by whichever kernel
        // computed them (the verify band runs the small_t tensor-core attention where a T=1 step
        // runs the fp32 flash kernel). Also ruled out as causes: the GEMV unroll depth (1/4/8 give
        // byte-identical ids), the wide-tile verify routing, --lm-head-draft, and run-to-run noise.
        //
        // The cap below is still the right one -- it bounds register pressure, not exactness -- but
        // the speculative path must not be described as token-identical to --spec none. The verify
        // pass is the authority for what it emits. See docs/ternary-port.md.
        if (options.draft_tokens == 0 || options.draft_tokens > 7) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,7]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
