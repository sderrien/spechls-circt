#ifndef SPECHLS_DIALECT_PASSES_H
#define SPECHLS_DIALECT_PASSES_H

namespace SpecHLS {

[[maybe_unused]] std::unique_ptr<mlir::Pass> createGecosSchedulePass();

#define GEN_PASS_DECL_GECOSSCHEDULEPASS
#define GEN_PASS_REGISTRATION
#include "Scheduling/Transforms/Passes.h.inc"

} // namespace SpecHLS

#endif // SPECHLS_DIALECT_PASSES_H