#include "mlir/IR/Operation.h"
#include <unordered_map>

using namespace mlir;

enum class NodeState { NotVisited, Visiting, Visited };

bool dfsVisit(Operation *op, std::unordered_map<Operation*, NodeState> &stateMap) {
  if (stateMap[op] == NodeState::Visiting) {
    // A cycle is detected if we visit a node that is already being visited
    return true;
  }
  if (stateMap[op] == NodeState::Visited) {
    // If the node is already fully visited, no cycle can be formed from it
    return false;
  }

  // Mark the node as being visited
  stateMap[op] = NodeState::Visiting;

  // Visit all predecessors (operations that define the operands of this operation)
  for (auto operand : op->getOperands()) {
    if (Operation *defOp = operand.getDefiningOp()) {
      if (dfsVisit(defOp, stateMap)) {
        return true; // Cycle detected in the predecessor
      }
    }
  }

  // Mark the node as fully visited
  stateMap[op] = NodeState::Visited;
  return false;
}

bool detectCycles(SmallVector<OperationData, 16> &schedule) {
  std::unordered_map<Operation*, NodeState> stateMap;

  // Initialize all nodes as NotVisited
  for (auto &opData : schedule) {
    stateMap[opData.op] = NodeState::NotVisited;
  }

  // Perform DFS from each node that hasn't been fully visited yet
  for (auto &opData : schedule) {
    if (stateMap[opData.op] == NodeState::NotVisited) {
      if (dfsVisit(opData.op, stateMap)) {
        return true; // Cycle detected
      }
    }
  }

  return false; // No cycles found
}
