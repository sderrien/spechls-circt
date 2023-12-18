#include "Scheduling/Problems.h"
#include "Scheduling/PassDetails.h"
#include "circt/Dialect/SSP/SSPAttributes.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Dialect/SSP/SSPPasses.h"
#include "circt/Dialect/SSP/Utilities.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Transforms/Passes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;

LogicalResult GammaMobilityProblem::verify() {
  if (failed(verifyInitiationInterval()))
    return failure();

  for (auto *op : getOperations())
    if (failed(verifyStartTime(op)) || failed(verifyStartTimeInCycle(op)))
      return failure();

  for (auto *op : getOperations()) {
    if (op->hasAttr("SpecHLS.gamma")) {
      bool hasSucceeded = false;
      for (auto &dep : getDependences(op))
        if (succeeded(verifyPrecedence(dep)) &&
            succeeded(verifyPrecedenceInCycle(dep))) {
          hasSucceeded = true;
          break;
        }
      if (!hasSucceeded)
        return failure();
    } else
      for (auto &dep : getDependences(op)) {
        if (failed(verifyPrecedence(dep)) ||
            failed(verifyPrecedenceInCycle(dep)))
          return failure();
      }
  }

  return success();
}