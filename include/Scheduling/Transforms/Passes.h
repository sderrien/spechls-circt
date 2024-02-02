#ifndef SPECHLS_DIALECT_PASSES_H
#define SPECHLS_DIALECT_PASSES_H

#include "Scheduling/Transforms/MobilityPass.h"
#include <circt/Dialect/SSP/SSPDialect.h>

namespace SpecHLS {

#define GEN_PASS_REGISTRATION
#include "Scheduling/Transforms/Passes.h.inc"

} // namespace SpecHLS

#endif // SPECHLS_DIALECT_PASSES_H