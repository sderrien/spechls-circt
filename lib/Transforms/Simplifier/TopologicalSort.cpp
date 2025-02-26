//===- MergeGammas.cpp - Arith-to-comb mapping pass ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the MergeGammas pass.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

namespace SpecHLS {

bool isSynchronousBarrier(Operation *innerOp) {
  return TypeSwitch<Operation *, bool>(innerOp)
      .Case<SpecHLS::MuOp>([&](auto op) { return true; })
      .Case<SpecHLS::DelayOp>([&](auto op) { return true; })
      .Default([&](auto op) { return false; });
}



void TopoSort(hw::HWModuleOp hwModule) {

  std::vector<mlir::Operation *> sortedOps;
  // Map from operation to in-degree (number of dependencies)
  llvm::DenseMap<mlir::Operation *, unsigned> inDegree;

  // Map from operation to list of operations that depend on it
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 4>>
      adjacencyList;

  auto block = hwModule.getBodyBlock();
  if (!block) {
    mlir::emitError(hwModule->getLoc(),
                    "No block in HwModule " + hwModule.getName());
    return;
  }

  Operation* lastOp ;

  for (auto &op : *block) {
    mlir::Operation *opPtr = &op;
    // we skip outputops
    if (!dyn_cast<hw::OutputOp>(op)) {
      // Initialize in-degree to zero if not already present
      inDegree[opPtr] = inDegree.lookup(opPtr);
      if (!isSynchronousBarrier(&op)) {
        // For each operand of the operation, get the defining operation
        for (mlir::Value operand : op.getOperands()) {
          if (mlir::Operation *defOp = operand.getDefiningOp()) {
            // Record that op depends on defOp
            if (defOp)
              adjacencyList[defOp].push_back(opPtr);

            // Increment in-degree of op
            inDegree[opPtr]++;
          }
        }
      }
    }

  }

  // Initialize WorkList with operations that have in-degree zero
  llvm::SmallVector<mlir::Operation *, 16> workList;

  for (auto &pair : inDegree) {
    if (pair.second == 0) {
      workList.push_back(pair.first);
    }
  }

  // Perform Topo sort
  while (!workList.empty()) {
    mlir::Operation *op = workList.pop_back_val();

    sortedOps.push_back(op);

    // For each operation that depends on op
    for (mlir::Operation *dependentOp : adjacencyList[op]) {
      // Decrement in-degree of dependent operation
      inDegree[dependentOp]--;

      // If in-degree becomes zero, add to WorkList
      if (inDegree[dependentOp] == 0) {
        //llvm::errs() << "Op " << dependentOp << "is ready\n";
        workList.push_back(dependentOp);
      }
    }
  }

  // Check if there was a cycle
  if (sortedOps.size() != inDegree.size()) {
    // There is a cycle in the dependency graph
    //llvm::errs() << "Cycle detected in region operations\n";
    sortedOps.clear();
    mlir::emitError(hwModule->getLoc(),
                    "Cycle detected in region\n");
  }

  auto headOp = sortedOps[0];
  // we stop at sortedOps.size()-1 to make sure hw.output is still there

  for (size_t k = 1; k < sortedOps.size(); k++) {
      //llvm::errs() << "Moving  " << *sortedOps[k] << " after "<< *sortedOps[k - 1] <<"\n";
      sortedOps[k]->moveAfter(sortedOps[k - 1]);
  }

  // fuse init-Mu nodes and const
  for (size_t k = 1; k < sortedOps.size(); k++) {
    auto op = sortedOps[k];

    TypeSwitch<Operation *, void >(sortedOps[k])
      .Case<SpecHLS::MuOp>([&](auto op) {
        auto defOp0= op->getOperand(0).getDefiningOp();
        if (auto initOp = dyn_cast<SpecHLS::InitOp>(defOp0)) {
          initOp->moveBefore(op);
        };
      });

  }

  //llvm::errs() << hwModule;
  mlir::verify(hwModule);
}


struct TopoSortPass  : public impl::TopoSortPassBase<TopoSortPass> {
public:
  void runOnOperation() override {
    auto top = getOperation();
    auto *block = top.getBody();

    for (auto &op : block->getOperations()) {
      if (auto topModule = dyn_cast<hw::HWModuleOp>(op)) {
        TopoSort(topModule);
      }
    }
    mlir::verify(top, true);
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createTopoSortPass(){
  return std::make_unique<TopoSortPass>();
}
} // namespace SpecHLS
