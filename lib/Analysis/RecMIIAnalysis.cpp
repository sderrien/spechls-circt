//===- SchedulingAnalysis.cpp - scheduling analyses -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements methods that perform analysis involving scheduling.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Utilities.h"
#include "circt/Scheduling/Algorithms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include <limits>
#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

/// RecMIIAnalysis constructs a CyclicProblem for each AffineForOp by
/// performing a memory dependence analysis and inserting dependences into the
/// problem. The client should retrieve the partially complete problem to add
/// and associate operator types.
namespace SpecHLS {

void recminAnalysis(Operation *op, AnalysisManager &am) {

  //  auto hwmodule = cast<circt::hw::HWModuleOp>(op);
  //
  //  auto delays = hwmodule.getOps<SpecHLS::DelayOp>();
  //  auto mus = hwmodule.getOps<SpecHLS::MuOp>();
  //  // auto all = std::merge(mus.begin(), mus.end(), );
  //  SmallVector<Operation, 128> content;
  //  auto operations = hwmodule->getBlock()->getOperations();
  //  // Only consider innermost loops of perfectly nested AffineForOps.
  //  for (int i=0;i<operations.size();i++) {
  //    llvm::errs() << operations[i];
  //  }
  //}
}
//
//    return llvm::TypeSwitch<Operation *, LogicalResult>(term)
//        .Case<cf::ClockedBranchOp>([&](auto branchOp) {
//          rewriter.replaceOpWithNewOp<cf::BranchOp>(branchOp, newDest,
//                                                    branchOp->getOperands());
//          return success();
//        })
//        .Case<cf::CondBranchOp>([&](auto condBr) {
//          auto cond = condBr.getCondition();
//
//          Block *trueDest = condBr.getTrueDest();
//          Block *falseDest = condBr.getFalseDest();
//
//          // Change to the correct destination.
//          if (trueDest == oldDest)
//            trueDest = newDest;
//
//          if (falseDest == oldDest)
//            falseDest = newDest;
//
//          rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
//              condBr, cond, trueDest, condBr.getTrueOperands(), falseDest,
//              condBr.getFalseOperands());
//          return success();
//        })
//        .Default([&](Operation *op) {
//          return op->emitError("Unexpected terminator that cannot be handled.");
//        });
//
//    SmallVector<AffineForOp> nestedLoops;
//    getPerfectlyNestedLoops(nestedLoops, root);
//    analyzeForOp(nestedLoops.back(), memoryAnalysis);
//  }
//}
//
// void circt::analysis::RecMIIAnalysis::analyzeForOp(
//    HWModuleOp forOp, MemoryDependenceAnalysis memoryAnalysis) {
//  // Create a cyclic scheduling problem.
//  CyclicProblem problem = CyclicProblem::get(forOp);
//
//  // Insert memory dependences into the problem.
//  forOp.getBody()->walk([&](Operation *op) {
//    // Insert every operation into the problem.
//    problem.insertOperation(op);
//
//    ArrayRef<MemoryDependence> dependences = memoryAnalysis.getDependences(op); if (dependences.empty())
//      return;
//
//    for (MemoryDependence memoryDep : dependences) {
//      // Don't insert a dependence into the problem if there is no dependence.
//      if (!hasDependence(memoryDep.dependenceType))
//        continue;
//
//      // Insert a dependence into the problem.
//      Problem::Dependence dep(memoryDep.source, op);
//      auto depInserted = problem.insertDependence(dep);
//      assert(succeeded(depInserted));
//      (void)depInserted;
//
//      // Use the lower bound of the innermost loop for this dependence. This
//      // assumes outer loops execute sequentially, i.e. one iteration of the
//      // inner loop completes before the next iteration is initiated. With
//      // proper analysis and lowerings, this can be relaxed.
//      unsigned distance = *memoryDep.dependenceComponents.back().lb;
//      if (distance > 0)
//        problem.setDistance(dep, distance);
//    }
//  });
//
//  // Insert conditional dependences into the problem.
//  forOp.getBody()->walk([&](Operation *op) {
//    Block *thenBlock = nullptr;
//    Block *elseBlock = nullptr;
//    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
//      thenBlock = ifOp.thenBlock();
//      elseBlock = ifOp.elseBlock();
//    } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
//      thenBlock = ifOp.getThenBlock();
//      if (ifOp.hasElse())
//        elseBlock = ifOp.getElseBlock();
//    } else {
//      return WalkResult::advance();
//    }
//
//    // No special handling required for control-only `if`s.
//    if (op->getNumResults() == 0)
//      return WalkResult::skip();
//
//    // Model the implicit value flow from the `yield` to the `if`'s result(s).
//    Problem::Dependence depThen(thenBlock->getTerminator(), op);
//    auto depInserted = problem.insertDependence(depThen);
//    assert(succeeded(depInserted));
//    (void)depInserted;
//
//    if (elseBlock) {
//      Problem::Dependence depElse(elseBlock->getTerminator(), op);
//      depInserted = problem.insertDependence(depElse);
//      assert(succeeded(depInserted));
//      (void)depInserted;
//    }
//
//    return WalkResult::advance();
//  });
//
//  // Set the anchor for scheduling. Insert dependences from all stores to the
//  // terminator to ensure the problem schedules them before the terminator.
//  auto *anchor = forOp.getBody()->getTerminator();
//  forOp.getBody()->walk([&](Operation *op) {
//    if (!isa<AffineStoreOp, memref::StoreOp>(op))
//      return;
//    Problem::Dependence dep(op, anchor);
//    auto depInserted = problem.insertDependence(dep);
//    assert(succeeded(depInserted));
//    (void)depInserted;
//  });
//
//  // Handle explicitly computed loop-carried values, i.e. excluding the
//  // induction variable. Insert inter-iteration dependences from the definers of
//  // "iter_args" to their users.
//  if (unsigned nIterArgs = anchor->getNumOperands(); nIterArgs > 0) {
//    auto iterArgs = forOp.getRegionIterArgs();
//    for (unsigned i = 0; i < nIterArgs; ++i) {
//      Operation *iterArgDefiner = anchor->getOperand(i).getDefiningOp();
//      // If it's not an operation, we don't need to model the dependence.
//      if (!iterArgDefiner)
//        continue;
//
//      for (Operation *iterArgUser : iterArgs[i].getUsers()) {
//        Problem::Dependence dep(iterArgDefiner, iterArgUser);
//        auto depInserted = problem.insertDependence(dep);
//        assert(succeeded(depInserted));
//        (void)depInserted;
//
//        // Values always flow between subsequent iterations.
//        problem.setDistance(dep, 1);
//      }
//    }
//  }
//
//  // Store the partially complete problem.
//  problems.insert(std::pair<Operation *, CyclicProblem>(forOp, problem));
//}
//
// CyclicProblem &
// circt::analysis::RecMIIAnalysis::getProblem(AffineForOp forOp) {
//  auto problem = problems.find(forOp);
//  assert(problem != problems.end() && "expected problem to exist");
//  return problem->second;
//}
}