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

struct GroupGammaNodes
    : public SpecHLS::impl::GroupGammaNodesPassBase<GroupGammaNodes> {
  GroupGammaNodes() {}

public:
  void runOnOperation() override;
};
//
void GroupGammaNodes::runOnOperation() {
  auto top = getOperation();


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
            bool leaf = true;
            for (auto user : gamma.getResult().getUsers()) {
              if (dyn_cast<SpecHLS::GammaOp>(*user)) {
                leaf= false;
              }
            }
            if (!leaf)
              continue;
            /*
             * Slices control logic of gamma node
             */
            SetVector<Operation *> slice = {};
            SetVector<Value*> inputs = {};
            auto opfilter = [&](Operation *op) {
              return TypeSwitch<Operation *, bool>(op)
                .Case<SpecHLS::GammaOp>([&](auto op) { return true;})
                .Default([&](auto op) { return false; });
            };

            getBackwardSlice(*gamma.getOperation(), slice, inputs, opfilter);
            if (slice.size()==0)
              continue;

            auto builder = OpBuilder(topModule.getContext());
            //builder.create<comb::ConcatOp>(gamma->getLoc(),)

            SetVector<Value*> outputs;
            for (auto res : gamma->getResults())
              outputs.insert(&res);

            auto newName = topModule.getName() + "_ctrl_" + std::to_string(gammaId);
            auto newModule = outlineSliceAsHwModule(topModule,slice,inputs,outputs,newName);
            if (newModule) {

              SmallVector<Value, 8> operands;
              for (auto i : inputs) {
                operands.push_back(*i);
              }

              builder.setInsertionPoint(gamma);

              auto inst = builder.create<hw::InstanceOp>(
                  gamma->getLoc(), newModule,
                  builder.getStringAttr(newModule.getName()), operands, ArrayAttr());

              gamma.getResult().replaceAllUsesWith(inst.getResult(0));

              newModule->setAttr(builder.getStringAttr("#pragma"),
                                 builder.getStringAttr("INLINE"));

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
createGroupGammaNodesPass() {
  // llvm::outs() << "GroupControlNodeImplPass created " << "\n";
  return std::make_unique<GroupGammaNodes>();
}
} // namespace SpecHLS