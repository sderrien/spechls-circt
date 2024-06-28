#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>
#include <unordered_map>

using namespace mlir;
using namespace llvm;

class DataFlowDominatorAnalysis {
  llvm:: DenseMap<Operation*, BitVector> dom;
  std::unordered_map<Operation*, int> operationIndices;

public:
  DataFlowDominatorAnalysis(Operation* operation) {
    // Iterate through all operations in the region of the provided operation
    Region& region = operation->getRegion(0); // Assumes there's only one region

    SmallVector<Operation*, 16> nodes;
    int index = 0;
    for (Block& block : region) {
      for (Operation& op : block) {
        nodes.push_back(&op);
        operationIndices[&op] = index++;
      }
    }

    // Initialize dominator sets for each operation using BitVector
    for (Operation* op : nodes) {
      BitVector bitVec(nodes.size(), true); // Initially, each operation dominates all others
      dom[op] = bitVec;
    }

    bool changed = true;

    // Iterate until no more changes occur
    while (changed) {
      changed = false;
      for (Operation* op : nodes) {
        BitVector tmp(nodes.size(), true);

        // Collect the defining operations of the operands
        for (auto value : op->getResults()) {
          for (auto user : value.getUsers()) {
            if (Operation* userOp = user->getParentOp()) {
              //        for (auto operand : op->getOperands()) {
              //          if (Operation* definingOp = operand.getDefiningOp()) {
              tmp &= dom[userOp]; // Intersect with the dominator set of the defining operation
            }

          }
        }
        int opIndex = operationIndices[op];
        tmp.set(opIndex); // An operation always dominates itself

        if (tmp != dom[op]) {
          dom[op] = tmp;
          changed = true;
        }
      }
    }
  }

  bool dominates(Operation* a, Operation* b) const {
    auto it = dom.find(b);
    if (it != dom.end()) {
      int aIndex = operationIndices.at(a);
      return it->second[aIndex];
    }
    return false;
  }

  // Print dominator sets for debugging
  void printDominatorSets() const {
    for (const auto& pair : dom) {
      std::cout << "Operation at " << pair.first << " is dominated by: ";
      const BitVector& bitVec = pair.second;
      for (auto& opIndexPair : operationIndices) {
        if (bitVec[opIndexPair.second]) {
          std::cout << opIndexPair.first << " ";
        }
      }
      std::cout << std::endl;
    }
  }
};
/*
int main() {
  // Initialize an MLIR context
  MLIRContext context;

  // For demonstration, create an example module with operations
  OpBuilder builder(&context);
  auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
  auto funcType = builder.getFunctionType({}, {});
  auto funcOp = builder.create<FuncOp>(builder.getUnknownLoc(), "example_func", funcType);
  Region &region = funcOp.getBody();

  Block *entryBlock = builder.createBlock(&region);
  auto type = builder.getIntegerType(32);
  Operation *op1 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), type, builder.getI32IntegerAttr(1));
  Operation *op2 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), type, builder.getI32IntegerAttr(2));
  Operation *op3 = builder.create<mlir::AddIOp>(builder.getUnknownLoc(), op1->getResult(0), op2->getResult(0));
  builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), op3->getResult(0));

  // Construct data-flow based dominator analysis
  DataFlowDominatorAnalysis analysis(funcOp);

  // Print dominator sets
  analysis.printDominatorSets();

  return 0;
}
*/