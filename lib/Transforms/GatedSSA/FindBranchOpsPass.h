#ifndef FIND_BRANCH_OPS_PASS_H
#define FIND_BRANCH_OPS_PASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include <map>
#include <vector>

namespace mlir {
std::unique_ptr<Pass> createFindBranchOpsPass();

namespace cf {
void registerFindBranchOpsPass();
} // end namespace cf
} // end namespace mlir

#endif // FIND_BRANCH_OPS_PASS_H
