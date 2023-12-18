#ifndef SPECHLS_DIALECT_ALGORITHMS_H
#define SPECHLS_DIALECT_ALGORITHMS_H

#include "Scheduling/Problems.h"

mlir::LogicalResult scheduleASAP(GammaMobilityProblem &prob, float cycleTime);

#endif // SPECHLS_DIALECT_ALGORITHMS_H