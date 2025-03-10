#include "OperationRanker.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

// Recursive function to compute rank of an operation
int computeRank(
    mlir::Operation *op,
    llvm::DenseMap<mlir::Operation *, int> &opRanks,
    llvm::SmallPtrSet<mlir::Operation *, 16> &visited,
    llvm::SmallPtrSet<mlir::Operation *, 16> &recursionStack) {

  // If rank is already computed, return it
  auto it = opRanks.find(op);
  if (it != opRanks.end()) {
    return it->second;
  }

  // Detect cycles (back-edges)
  if (recursionStack.contains(op)) {
    // Back-edge detected, ignore this path
    return -1; // Indicates that this path should not contribute to rank
  }

  visited.insert(op);
  recursionStack.insert(op);

  int maxOperandRank = -1;
  for (mlir::Value operand : op->getOperands()) {
    if (auto definingOp = operand.getDefiningOp()) {
      int operandRank = computeRank(definingOp, opRanks, visited, recursionStack);
      if (operandRank >= 0) { // Ignore back-edges
        maxOperandRank = std::max(maxOperandRank, operandRank)+1;
      }
    } else {
x      // Operand is a block argument or undefined; ignore for ranking
    }
  }

  recursionStack.erase(op);

  int opRank = maxOperandRank + 1;
  opRanks[op] = opRank;
  return opRank;
}

llvm::DenseMap<mlir::Operation *, int> rankOperations(mlir::Block &block) {
  llvm::DenseMap<mlir::Operation *, int> opRanks;
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;
  llvm::SmallPtrSet<mlir::Operation *, 16> recursionStack;

  // Iterate over all operations in the block
  for (mlir::Operation &op : block) {
    if (!visited.contains(&op)) {
      computeRank(&op, opRanks, visited, recursionStack);
    }
  }

  return opRanks;
}
