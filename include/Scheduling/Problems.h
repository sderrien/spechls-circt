#ifndef SPECHLS_DIALECT_PROBLEMS_H
#define SPECHLS_DIALECT_PROBLEMS_H

#include "SpecHLS/SpecHLSDialect.h"
#include "circt/Dialect/SSP/SSPDialect.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"

#define DEFINE_COMMON_MEMBERS(ProblemClass)                                    \
protected:                                                                     \
  ProblemClass() {}                                                            \
                                                                               \
public:                                                                        \
  static constexpr auto PROBLEM_NAME = #ProblemClass;                          \
  static ProblemClass get(mlir::Operation *containingOp) {                     \
    ProblemClass prob;                                                         \
    prob.setContainingOp(containingOp);                                        \
    return prob;                                                               \
  }

class GammaMobilityProblem
    : public virtual circt::scheduling::ChainingCyclicProblem {
  DEFINE_COMMON_MEMBERS(GammaMobilityProblem)

public:
  mlir::LogicalResult verify() override;
};

#undef DEFINE_COMMON_MEMBERS
#endif // SPECHLS_DIALECT_PROBLEMS_H