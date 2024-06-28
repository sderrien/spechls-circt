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
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// StubExternalModules Helpers
//===----------------------------------------------------------------------===//

// Reimplemented from SliceAnalysis to use a worklist rather than recursion and
// non-insert ordered set.  Implement this as a DFS and not a BFS so that the
// order is stable across changes to intermediary operations.  (It is then
// necessary to use the _operands_ as a worklist and not the _operations_.)
void getBackwardSlice(Operation &rootOp, SetVector<Operation *> &backwardSlice,
                      function_ref<bool(Operation *)> filter) {
  SmallVector<Value> worklist(rootOp.getOperands());

  while (!worklist.empty()) {
    Value operand = worklist.pop_back_val();
    Operation *definingOp = operand.getDefiningOp();

    if (!definingOp ||
        definingOp->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
      continue;

    // outs() << "defining op " << *definingOp << "\n";
    //  definingOp->t

    // Evaluate whether we should keep this def.
    // This is useful in particular to implement scoping; i.e. return the
    // transitive backwardSlice in the current scope.
    if (filter) {
      auto res = filter(definingOp);
      // outs() << "filter ( " << *definingOp << ")- > " << res << "\n";
      if (!res) {
        // outs() << "we dont keep def " << *definingOp << "\n";
        continue;
      } else {
        // outs() << "we continue with def " << *definingOp << "\n";
      }
    }

    if (definingOp) {
      if (!backwardSlice.contains(definingOp))
        for (auto newOperand : llvm::reverse(definingOp->getOperands())) {
          // outs() << "searching for operand  " << newOperand << "\n";
          worklist.push_back(newOperand);
        }
    } else if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
      Block *block = blockArg.getOwner();
      Operation *parentOp = block->getParentOp();
      // TODO: determine whether we want to recurse backward into the other
      // blocks of parentOp, which are not technically backward unless they
      // flow into us. For now, just bail.
      assert(parentOp->getNumRegions() == 1 &&
             parentOp->getRegion(0).getBlocks().size() == 1);
      if (!backwardSlice.contains(parentOp))
        for (auto newOperand : llvm::reverse(parentOp->getOperands()))
          worklist.push_back(newOperand);
    } else {
      llvm_unreachable("No definingOp and not a block argument.");
    }

    backwardSlice.insert(definingOp);
  }
}

// Some blocks have terminators, some don't
void setInsertPointToEndOrTerminator(OpBuilder &builder, Block *block) {
  if (!block->empty() && isa<hw::HWModuleOp>(block->getParentOp()))
    builder.setInsertionPoint(&block->back());
  else
    builder.setInsertionPointToEnd(block);
}

// Shallow clone, which we use to not clone the content of blocks, doesn't
// clone the regions, so create all the blocks we need and update the mapping.
void addBlockMapping(IRMapping &cutMap, Operation *oldOp, Operation *newOp) {
  assert(oldOp->getNumRegions() == newOp->getNumRegions());
  for (size_t i = 0, e = oldOp->getNumRegions(); i != e; ++i) {
    auto &oldRegion = oldOp->getRegion(i);
    auto &newRegion = newOp->getRegion(i);
    for (auto oi = oldRegion.begin(), oe = oldRegion.end(); oi != oe; ++oi) {
      cutMap.map(&*oi, &newRegion.emplaceBlock());
    }
  }
}

void getSliceInputs(mlir::SetVector<Operation *> &slice,
                    SetVector<Value> &inputs) {
  // Find the dataflow into the clone set
  for (auto *op : slice) {
    for (auto arg : op->getOperands()) {
      auto argOp = arg.getDefiningOp(); // may be null
      if (argOp == NULL) {
        inputs.insert(arg);
      } else {
        // If a value is not used by any op in the slice, it should be
        // considered as an input
        if (!slice.count(argOp))
          inputs.insert(arg);
      }
    }
  }
};

/*
 * void getBackwardSlice(Operation &rootOp, SetVector<Operation *> &slice,
                      llvm::function_ref<bool(Operation *)> filter);
*/

// Given a set of values, construct a module and bind instance of that module
// that passes those values through.  Returns the new module and the instance
// pointing to it.

hw::HWModuleOp outlineSliceAsHwModule(hw::HWModuleOp op, Operation &root,
                                      SetVector<Operation *> &slice,
                                      SetVector<Value> &inputs,
                                      SetVector<Value> &outputs,
                                      Twine newName) {

  bool verbose = false;

  auto builder = OpBuilder(op.getContext());
  auto moduleName = builder.getStringAttr(newName);

  if (verbose) {
  }

  for (auto v : outputs) {
    if (!slice.contains(v.getDefiningOp())) {
      llvm::errs() << "inconsistent output " << v << " : " << *v.getDefiningOp()
                   << " is not in the slice"
                   << "\n";
      return NULL;
    }
  }

  // Create the extracted module right next to the original one.
  OpBuilder b(op);

  // Construct the ports, this is just the input Values
  SmallVector<hw::PortInfo> ports;
  {
    Namespace portNames;
    for (auto port : enumerate(inputs)) {
      auto name = portNames.newName("in_" + Twine(port.index()));

      ports.push_back({{b.getStringAttr(name), port.value().getType(),
                        hw::ModulePort::Direction::Input},
                       port.index()});
    }
    for (auto port : enumerate(outputs)) {
      auto name = portNames.newName("out_" + Twine(port.index()));
      ports.push_back({{b.getStringAttr(name), port.value().getType(),
                        hw::ModulePort::Direction::Output},
                       port.index()});
    }
  }

  // Create the module, setting the output path if indicated.
  auto newModule = b.create<hw::HWModuleOp>(op->getLoc(), moduleName, ports);
  llvm::errs() << newModule;
  b.setInsertionPointToStart(newModule.getBodyBlock());

  IRMapping cutMap;
  auto outputOp = newModule.getBodyBlock()->getTerminator();

  // Update the mapping from old values to cloned values
  for (auto port : enumerate(inputs)) {
    cutMap.map(port.value(), newModule.getBody().getArgument(port.index()));
  }

  op.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (slice.count(op)) {
      auto newOp = b.cloneWithoutRegions(*op, cutMap);
    }
  });

  for (auto port : enumerate(outputs)) {
    auto newVal = cutMap.getValueMap().at(port.value());
    outputOp->insertOperands(port.index(),newVal);
  }

  mlir::verify(newModule, true);
  return newModule;
}
