//
// Created by Steven on 02/01/2024.
//

#include "InlineOptimizedModules.h"
//===- InlineModules.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombDialect.h"

#define DEBUG_TYPE "arc-inline-modules"

namespace circt {
} // namespace circt

using namespace circt;
using namespace hw;
using namespace igraph;
using mlir::InlinerInterface;

namespace {
struct InlineOptimizedModulesPass
    : public SpecHLS::impl::InlineModulesBase<InlineOptimizedModulesPass> {
  void runOnOperation() override;
};

/// A simple implementation of the `InlinerInterface` that marks all inlining as
/// legal since we know that we only ever attempt to inline `HWModuleOp` bodies
/// at `InstanceOp` sites.
struct PrefixingInliner : public InlinerInterface {
  StringRef prefix;
  PrefixingInliner(MLIRContext *context, StringRef prefix)
      : InlinerInterface(context), prefix(prefix) {}

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const override {
    assert(isa<hw::OutputOp>(op));
    for (auto [from, to] : llvm::zip(valuesToRepl, op->getOperands()))
      from.replaceAllUsesWith(to);
  }

  void processInlinedBlocks(
      iterator_range<Region::iterator> inlinedBlocks) override {
    for (Block &block : inlinedBlocks)
      block.walk([&](Operation *op) { updateNames(op); });
  }

  StringAttr updateName(StringAttr attr) const {
    if (attr.getValue().empty())
      return attr;
    return StringAttr::get(attr.getContext(), prefix + "/" + attr.getValue());
  }




  void updateNames(Operation *op) const {
    if (auto name = op->getAttrOfType<StringAttr>("name"))
      op->setAttr("name", updateName(name));
    if (auto name = op->getAttrOfType<StringAttr>("instanceName"))
      op->setAttr("instanceName", updateName(name));
    if (auto namesAttr = op->getAttrOfType<ArrayAttr>("names")) {
      SmallVector<Attribute> names(namesAttr.getValue().begin(),
                                   namesAttr.getValue().end());
      for (auto &name : names)
        if (auto nameStr = name.dyn_cast<StringAttr>())
          name = updateName(nameStr);
      op->setAttr("names", ArrayAttr::get(namesAttr.getContext(), names));
    }
  }
};
} // namespace

bool hasConstantOutputs(HWModuleOp op) {
  for (auto &_innerop : op.getBodyBlock()->getOperations()) {
    bool ok = TypeSwitch<Operation *, bool>(&_innerop)
                  .Case<circt::hw::ConstantOp>([&](auto op) { return true; })
                  .Case<circt::hw::OutputOp>([&](auto op) { return true; })
                  .Default([&](auto op) {
                    llvm::outs() << "Operation " << _innerop  << "is not constant\n";
                    return false;
                  });

    if (!ok)
      return false;
  }
  return true;
}
void InlineOptimizedModulesPass::runOnOperation() {


  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
  DenseSet<Operation *> handled;


  // Iterate over all instances in the instance graph. This ensures we visit
  // every module, even private top modules (private and never instantiated).
  for (auto *startNode : instanceGraph) {
    llvm::outs() << "startnode  " << startNode << "\n";

    if (handled.count(startNode->getModule().getOperation()))
      continue;

    // Visit the instance subhierarchy starting at the current module, in a
    // depth-first manner. This allows us to inline child modules into parents
    // before we attempt to inline parents into their parents.
    for (InstanceGraphNode *node : llvm::post_order(startNode)) {
      if (!handled.insert(node->getModule().getOperation()).second)
        continue;

      unsigned numUsesLeft = node->getNumUses();
      if (numUsesLeft == 0)
        continue;

      for (auto *instRecord : node->uses()) {
        // Only inline private `HWModuleOp`s (no extern or generated modules).
        auto hwmodule =
            dyn_cast_or_null<HWModuleOp>(node->getModule().getOperation());
        if (!hwmodule || !hwmodule.isPrivate())
          continue;

        if (!hasConstantOutputs(hwmodule)) {
          continue;

        }


        // Only inline at plain old HW `InstanceOp`s.
        auto inst = dyn_cast_or_null<InstanceOp>(
            instRecord->getInstance().getOperation());
        if (!inst)
          continue;

        llvm::outs() << "analyzing " << inst << "\n";
        bool isLastModuleUse = --numUsesLeft == 0;

          // Retrieve the symbolic name associated with the InstanceOp operand

        auto symbolicNameAttr = inst.getInnerSymAttr();
        llvm::outs() << "inlining " << inst << "\n";
        PrefixingInliner inliner(&getContext(), inst.getInstanceName());

        if (failed(mlir::inlineRegion(inliner, &hwmodule.getBody(), inst,
                                      inst.getOperands(), inst.getResults(),
                                      std::nullopt, !isLastModuleUse))) {
          inst.emitError("failed to inline '")
              << hwmodule.getModuleName() << "' into instance '"
              << inst.getInstanceName() << "'";
          return signalPassFailure();
        }

        inst.erase();
        if (isLastModuleUse)
          hwmodule->erase();
      }
    }
  }
}

std::unique_ptr<mlir::OperationPass<circt::hw::HWModuleOp>> SpecHLS::createInlineModulesPass() {
  return std::make_unique<InlineOptimizedModulesPass>();
}
