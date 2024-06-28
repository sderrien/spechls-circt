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
#include "Common/OutliningUtils.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SetVector.h"

#include <set>

using namespace mlir;
using namespace circt;



//===----------------------------------------------------------------------===//
// StubExternalModules Pass
//===----------------------------------------------------------------------===//

struct GroupControlNodePass
    : public SpecHLS::impl::GroupControlNodePassBase<GroupControlNodePass> {
  int max_bitwidth = 4;
  GroupControlNodePass() {}

public:
  void runOnOperation() override;
};
//
void GroupControlNodePass::runOnOperation() {
  auto top = getOperation();

  llvm::outs() << "GroupControlNodeImplPass on design " << top << "\n";

  auto *topLevelModule = top.getBody();
  int gammaId = 0;

  for (auto &op : llvm::make_early_inc_range(topLevelModule->getOperations())) {
    if (auto topModule = dyn_cast<hw::HWModuleOp>(op)) {
      if (!topModule.getBody().empty()) {

        for (auto &innerOp : llvm::make_early_inc_range(
                 topModule.getBodyBlock()->getOperations())) {
          if (auto gamma = dyn_cast<SpecHLS::GammaOp>(innerOp)) {
            gammaId++;
            if (!(gamma->getNumOperands() > 0)) {
              continue;
            }
            auto controlValue = gamma->getOperand(0);
            auto controlOp = controlValue.getDefiningOp();

            /*
             * Slices control logic of gamma node
             */
            SetVector<Operation *> slice = {};
            auto opfilter =
             [&](Operation *op) {
              // llvm::outs() << " default filter  " << *op << "\n";
              bool res =
                  TypeSwitch<Operation *, bool>(op)
                      .Case<circt::comb::AddOp>([&](auto op) {
                        // llvm::outs() << " found and " << *op << "\n";
                        circt::comb::AddOp _op = op;
                        return (_op.getResult()
                                    .getType()
                                    .getIntOrFloatBitWidth()) < max_bitwidth;
                      })
                      .Case<circt::comb::SubOp>([&](auto op) {
                        // llvm::outs() << " found and " << *op << "\n";
                        circt::comb::SubOp _op = op;
                        return (_op.getResult()
                                    .getType()
                                    .getIntOrFloatBitWidth()) < max_bitwidth;
                      })
                      .Case<circt::comb::ICmpOp>([&](auto op) {
                        // llvm::outs() << " found and " << *op << "\n";
                        circt::comb::ICmpOp _op = op;
                        return (_op.getResult()
                                    .getType()
                                    .getIntOrFloatBitWidth()) < max_bitwidth;
                      })
                      .Case<circt::comb::AndOp>([&](auto op) {
                        // llvm::outs() << " found and " << *op << "\n";
                        return true;
                      })
                      .Case<circt::comb::OrOp>([&](auto op) { return true; })
                      .Case<circt::comb::XorOp>([&](auto op) { return true; })
                      .Case<circt::comb::ExtractOp>(
                          [&](auto op) { return true; })
                      .Case<circt::comb::ConcatOp>(
                          [&](auto op) { return true; })
                      .Case<circt::hw::ConstantOp>(
                          [&](auto op) { return true; })
                      .Case<circt::comb::MuxOp>([&](auto op) { return true; })
                      .Case<circt::comb::TruthTableOp>(
                          [&](auto op) { return true; })
                      .Case<SpecHLS::LookUpTableOp>(
                          [&](auto op) { return true; })
                      .Default([&](auto op) {
                        // llvm::outs() << " default filter  " << *op << "\n";
                        return false;
                      });
              // llvm::outs() << " res " << res << "\n";
              return res;
            };

           if (!opfilter(controlOp)) {
              continue;
           }


            getBackwardSlice(*controlOp, slice, opfilter);
            slice.insert(controlOp);
            // Find the dataflow into the clone set
            SetVector<Value> inputs;
            getSliceInputs(slice,inputs);

            SetVector<Value> outputs;
            for (auto res : controlOp->getResults()) {
              outputs.insert(res);
            }
            auto newName = topModule.getName() + "_ctrl_" + std::to_string(gammaId);
            auto newModule = outlineSliceAsHwModule(topModule,*controlOp,slice,inputs,outputs,newName);
            if (newModule) {
              auto builder = OpBuilder(topModule.getContext());

              SmallVector<Value, 8> operands;
              for (auto i : inputs) {
                operands.push_back(i);
              }

              builder.setInsertionPoint(gamma);

              auto inst = builder.create<hw::InstanceOp>(
                  controlOp->getLoc(), newModule,
                  builder.getStringAttr(newModule.getName()), operands, ArrayAttr());

              gamma.setOperand(0, inst.getResult(0));

              newModule->setAttr(builder.getStringAttr("#pragma"),
                                 builder.getStringAttr("CONTROL_NODE"));

            }
          }
        }
      }
    }
  }
  mlir::verify(top, true);
}

namespace SpecHLS {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createGroupControlNodePass() {
  // llvm::outs() << "GroupControlNodeImplPass created " << "\n";
  return std::make_unique<GroupControlNodePass>();
}
} // namespace SpecHLS