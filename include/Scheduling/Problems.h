#ifndef SPECHLS_DIALECT_PROBLEMS_H
#define SPECHLS_DIALECT_PROBLEMS_H

#include "SpecHLS/SpecHLSDialect.h"
#include "circt/Dialect/SSP/SSPDialect.h"
#include "circt/Dialect/SSP/Utilities.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"

namespace SpecHLS {

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

  OperationProperty<unsigned> minPosition;

public:
  mlir::LogicalResult verify() override;

  void setMinPosition(mlir::Operation *op, unsigned value) {
    minPosition[op] = value;
  }

  std::optional<unsigned> getMinPosition(mlir::Operation *op) {
    return minPosition.lookup(op);
  }
};

}; // namespace SpecHLS

#undef DEFINE_COMMON_MEMBERS
#endif // SPECHLS_DIALECT_PROBLEMS_H