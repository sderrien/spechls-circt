#ifndef SPECHLS_DIALECT_PROBLEM_H
#define SPECHLS_DIALECT_PROBLEM_H

#include <circt/Dialect/SSP/SSPOps.h>
#include <llvm/ADT/SmallVector.h>

class Operation {};

class MobilityProblem {
private:
  llvm::SmallVector<Operation> ops;

public:
  MobilityProblem(circt::ssp::InstanceOp op);
};

#endif // SPECHLS_DIALECT_PROBLEM_H