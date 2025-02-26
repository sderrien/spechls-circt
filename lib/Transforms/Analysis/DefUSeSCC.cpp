//
// Created by Steven on 18/10/2024.
//


#include <Common/DefUseSCC.h>
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

// Recursive DFS helper function for Tarjan's algorithm
void SCCComputer::dfs(mlir::Operation *op) {
  opIndex[op] = index;
  opLowLink[op] = index;
  index++;

  stack.push_back(op);
  onStack.insert(op);

  // Iterate over all uses of the operation (def/use graph)
  for (mlir::Operation *userOp : op->getUsers()) {
    if (opIndex.find(userOp) == opIndex.end()) {
      // If userOp hasn't been visited, recurse
      dfs(userOp);
      opLowLink[op] = std::min(opLowLink[op], opLowLink[userOp]);
    } else if (onStack.count(userOp)) {
      // If userOp is on the stack, it's part of the current SCC
      opLowLink[op] = std::min(opLowLink[op], opIndex[userOp]);
    }
  }

  // If op is a root node, pop the stack to generate an SCC
  if (opLowLink[op] == opIndex[op]) {
    DefUseSCC scc;
    mlir::Operation *w;
    do {
      w = stack.pop_back_val();
      onStack.erase(w);
      scc.operations.push_back(w);
    } while (w != op);

    sccs.push_back(std::move(scc));
  }
}

// Main function to compute SCCs over a given operation range
SCCComputer::SCCCollection SCCComputer::computeSCCs(mlir::Block &block) {
  for (mlir::Operation &op : block.getOperations()) {

    if (opIndex.find(&op) == opIndex.end()) {
      dfs(&op);
    }
  }
  return sccs;
}

