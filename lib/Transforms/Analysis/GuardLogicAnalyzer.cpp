//===- GroupControlNode.cpp - SV Simulation Extraction Pass --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass extracts simulation constructs to sunewModuleules.
// It will take simulation operations, write, finish, assert, assume, and cover
// and extract them and the dataflow into them into a separate module.  This
// module is then instantiated in the original module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"

#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/LogicalEquivalence/LogicExporter.h"
#include "circt/LogicalEquivalence/Utility.h"
#include "circt/Support/Namespace.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/GenericDomTree.h"
#include <Common/OutliningUtils.h>
#include <Common/TransitiveClosure.h>

/*
 * Checks that target Op has only one boolean output and checks for its
 * satisfiability using the circt lec framework
 */
mlir::LogicalResult analyseForSatifability(mlir::ModuleOp &topModule,
                                           hw::HWModuleOp &hwModule,
                                           mlir::Operation &targetOp) {

  if (targetOp.getNumResults() != 1 ||
      !targetOp.getResult(0).getType().isInteger(1)) {
    return mlir::LogicalResult::failure();
  }

  auto opfilter = [&](Operation *op) {
    bool res =
        TypeSwitch<Operation *, bool>(op)
            .Case<circt::comb::AddOp>([&](auto op) { return true; })
            .Case<circt::comb::SubOp>([&](auto op) { return true; })
            .Case<circt::comb::ICmpOp>([&](auto op) { return true; })
            .Case<circt::comb::AndOp>([&](auto op) { return true; })
            .Case<circt::comb::OrOp>([&](auto op) { return true; })
            .Case<circt::comb::XorOp>([&](auto op) { return true; })
            .Case<circt::comb::ExtractOp>([&](auto op) { return true; })
            .Case<circt::comb::ConcatOp>([&](auto op) { return true; })
            .Case<circt::hw::ConstantOp>([&](auto op) { return true; })
            .Case<circt::comb::MuxOp>([&](auto op) { return true; })
            .Case<circt::comb::TruthTableOp>([&](auto op) { return true; })
            .Case<SpecHLS::LookUpTableOp>([&](auto op) { return true; })
            .Default([&](auto op) { return false; });
    return res;
  };

  if (!opfilter(&targetOp)) {
    return mlir::LogicalResult::failure();
  }

  SetVector<Operation *> slice = {};
  SetVector<Value> inputs;
  SetVector<Value> outputs;

  getBackwardSlice(targetOp, slice, opfilter);
  getSliceInputs(slice, inputs);

  for (auto res : targetOp.getResults()) {
    outputs.insert(res);
  }

  auto newName = hwModule.getName() + "_sat";
  auto newModule = outlineSliceAsHwModule(hwModule, targetOp, slice, inputs,
                                          outputs, newName);

  auto builder = OpBuilder(newModule.getContext());

  /*
   This sequence of instructions builds an empty clone of the
   topModule and sets its output to the constant '1'. This module will
   be used as a comparison element to determine if the output of
   topModule is always 'false' during the LEC (Logic Equivalence
   Checking) step */

  auto cloneOp = builder.cloneWithoutRegions(newModule);
  builder.setInsertionPointToStart(cloneOp.getBodyBlock());
  auto newRet = builder.create<hw::OutputOp>(cloneOp.getLoc());
  auto newConst =
      builder.create<hw::ConstantOp>(cloneOp.getLoc(), builder.getI1Type(), 0);
  newRet.setOperand(0, newConst->getResult(0));

  /* This following instanciates a LEC solver and determine its results */

  Solver s(topModule.getContext(), true);
  auto moduleName = topModule->getName().getStringRef();
  Solver::Circuit *c1 = s.addCircuit(moduleName);
  auto cloneName = cloneOp->getName().getStringRef();
  Solver::Circuit *c2 = s.addCircuit(moduleName);

  // Initialize a logic exporter for the first circuit then run it on
  // the top-level module of the first input file.
  llvm::outs() << "Analyzing the first circuit\n";
  auto exporter = std::make_unique<LogicExporter>(moduleName, c1);
  if (failed(exporter->run(topModule)))
    return mlir::LogicalResult::failure();

  // Initialize a logic exporter for the first circuit then run it on
  // the top-level module of the first input file.
  llvm::outs() << "Analyzing the second circuit\n";
  auto exporter2 = std::make_unique<LogicExporter>(moduleName, c2);
  if (failed(exporter->run(topModule)))
    return mlir::LogicalResult::failure();

  // The logical constraints have been exported to their respective
  // circuit representations and can now be solved for equivalence.
  auto result = s.solve();
  return result;
}

SmallVector<std::tuple<mlir::Operation, mlir::Operation>>
extractMutuallyExclusiveCheckCandidates(mlir::ModuleOp &topModule) {}

mlir::LogicalResult checkIfMutuallyExclusive(mlir::Operation &a,
                                             mlir::Operation &b) {
  if (a.getParentOp() != b.getParentOp())
    return failure();
  SetVector<Operation *> forwardSlice;
  SetVector<Operation *> backwardSlice;
  BackwardSliceOptions options;
  options.filter = [&](Operation *op) {
    bool res = TypeSwitch<Operation *, bool>(op)
                   .Case<SpecHLS::MuOp>([&](auto op) { return false; })
                   .Case<SpecHLS::DelayOp>([&](auto op) { return false; })
                   .Case<hw::HWModuleOp>([&](auto op) { return false; })
                   .Default([&](auto op) { return false; });
    return res;
  };
  getBackwardSlice(&a, &forwardSlice, options);

  // DominanceInfo domInfo;
  // auto domTree = domInfo.getDomTree(region);
  // domTree
  //  Traverse each block in the region to print its dominator
  //  for (Block &block : region) {
  //    if (Block *idom = domInfo.getNode(&block)->getIDom()) {
  //      llvm::outs() << "Block " << block.getName() << " is dominated by block
  //      " << idom->getName() << "\n";
  //    } else {
  //      llvm::outs() << "Block " << block.getName() << " has no dominator
  //      (entry block).\n";
  //    }
  //  }
  //  auto  succa = tc.getTransitiveSuccessors(&a);
  //  SmallVector<mlir::Value> guards;
  //  for (auto succ : succa) {
  //    if (auto gamma = dyn_cast<SpecHLS::GammaOp>(succ)) {
  //      guards.push_back(gamma.getOperand(0));
  //    }
  //  }
  //
  //  auto  succb = tc.getTransitiveSuccessors(&b);
}

struct GuardLogicAnalyzerPass
    : public SpecHLS::impl::GuardLogicAnalyzerPassBase<GuardLogicAnalyzerPass> {
  GuardLogicAnalyzerPass() {}

public:
  void runOnOperation() override;
};
//
void GuardLogicAnalyzerPass::runOnOperation() {
  auto top = getOperation();

  llvm::outs() << "GroupControlNodeImplPass on design " << top << "\n";

  auto *topLevelModule = top.getBody();

  for (auto &op : llvm::make_early_inc_range(topLevelModule->getOperations())) {
    if (auto topModule = dyn_cast<hw::HWModuleOp>(op)) {
      if (!topModule.getBody().empty()) {

        llvm::SmallVector<SpecHLS::GammaOp *> gammas;
        llvm::SmallVector<SpecHLS::ArrayReadOp *> reads;

        topModule.getBodyBlock()->walk([&gammas, &reads](mlir::Operation *op) {
          if (auto gamma = dyn_cast<SpecHLS::GammaOp>(op)) {
            gammas.push_back(&gamma);
          }
          if (auto read = dyn_cast<SpecHLS::ArrayReadOp>(op)) {
            reads.push_back(&read);
          }
        });

        OpBuilder builder(topModule);

        PathAnalyzer analyzer(topModule.getRegion());

        for (auto gamma : gammas) {
          if (!(gamma->getNumOperands() > 0)) {
            continue;
          }


          llvm::outs() << "Successors of " << gamma << "\n";
          for (auto succ : analyzer.getTransitiveSuccessors(gamma->getOperation())) {
            llvm::outs() << " - " << *succ << "\n";
          }

          for (size_t pos = 0; pos < gamma->getNumOperands(); pos++) {
            auto value = gamma->getOperand(pos);
            llvm::outs() << "Predecessors of gamma input " << value << "\n";
            if (auto definingOp = value.getDefiningOp()) {
              for (auto pred :
                   analyzer.getTransitivePredecessors(definingOp)) {
                llvm::outs() << " - " << *pred << "\n";
              }
            }
          }
        }

        for (auto read : reads) {
          llvm::SmallVector<mlir::Operation*> lookupTables;

          for (auto gamma : gammas) {
            llvm::SmallVector<int> offsets;
            for (size_t pos = 1; pos < gamma->getNumOperands(); pos++) {
              if (analyzer.isTransitivePredecessor(gamma->getOperation(),read->getOperation())) {
                offsets.push_back(1);
              } else {
                offsets.push_back(1);
              }
            }
            auto content = builder.getI32ArrayAttr(offsets);
            auto lut = builder.create<SpecHLS::LookUpTableOp>(gamma->getLoc(),builder.getI1Type(),gamma->getOperand(0),content);
            lookupTables.push_back(lut);
          }

          mlir::Operation* current = builder.create<hw::ConstantOp>(read->getLoc(),builder.getI1Type(),1);
          for (auto lut : lookupTables) {
            current = builder.create<comb::AndOp>(read->getLoc(),builder.getI1Type(),current->getResult(0));
          }

          SetVector<mlir::Operation*> slice;
          SetVector<mlir::Value> inputs;
          SetVector<mlir::Value> outputs;
          auto predModule = outlineSliceAsHwModule(topModule,*current, slice, inputs,outputs, "predicate");

          llvm::errs() << predModule;




        }
      }
    }
  }
  mlir::verify(top, true);
  return;
}


namespace SpecHLS {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createGuardLogicAnalyzerPass() {
  // llvm::outs() << "GroupControlNodeImplPass created " << "\n";
  return std::make_unique<GuardLogicAnalyzerPass>();
}
} // namespace SpecHLS