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

#include "SpecHLS/SpecHLSOps.h"
#include "SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SetVector.h"

#include <set>

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// StubExternalModules Helpers
//===----------------------------------------------------------------------===//

// Reimplemented from SliceAnalysis to use a worklist rather than recursion and
// non-insert ordered set.  Implement this as a DFS and not a BFS so that the
// order is stable across changes to intermediary operations.  (It is then
// necessary to use the _operands_ as a worklist and not the _operations_.)
static void
getBackwardSliceSimple(Operation *rootOp, SetVector<Operation *> &backwardSlice,
                       llvm::function_ref<bool(Operation *)> filter) {
  SmallVector<Value> worklist(rootOp->getOperands());

  while (!worklist.empty()) {
    Value operand = worklist.pop_back_val();
    Operation *definingOp = operand.getDefiningOp();


    if (!definingOp ||
        definingOp->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
      continue;

    llvm::outs() << "defining op " << *definingOp << "\n";
    // definingOp->t

    // Evaluate whether we should keep this def.
    // This is useful in particular to implement scoping; i.e. return the
    // transitive backwardSlice in the current scope.
    if (filter) {
      auto res = filter(definingOp);
      llvm::outs() << "filter ( " << *definingOp << ")- > " << res << "\n";
      if (!res) {
        llvm::outs() << "we dont keep def " << *definingOp << "\n";
        continue;
      } else {
        llvm::outs() << "we continue with def " << *definingOp << "\n";
      }
    }

    if (definingOp) {
      if (!backwardSlice.contains(definingOp))
        for (auto newOperand : llvm::reverse(definingOp->getOperands())) {
          llvm::outs() << "searching for operand  " << newOperand << "\n";
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

// Given a set of values, construct a module and bind instance of that module
// that passes those values through.  Returns the new module and the instance
// pointing to it.
static hw::HWModuleOp createModuleForCut(hw::HWModuleOp op,
                                         StringAttr moduleName,
                                         SetVector<Value> &inputs,
                                         SetVector<Value> &outputs,
                                         IRMapping &cutMap) {
  // Filter duplicates and track duplicate reads of elements so we don't
  // make ports for them
  SmallVector<Value> realInputs;
  SmallVector<Value> realOutputs;
  DenseMap<Value, Value> dups; // wire,reg,lhs -> read
  DenseMap<Value, SmallVector<Value>>
      realReads; // port mapped read -> dup reads
  for (auto v : inputs) {
    realInputs.push_back(v);
  }
  for (auto v : outputs) {
    realOutputs.push_back(v);
  }

  // Create the extracted module right next to the original one.
  OpBuilder b(op);

  // Construct the ports, this is just the input Values
  SmallVector<hw::PortInfo> ports;
  {
    Namespace portNames;
    for (auto port : llvm::enumerate(realInputs)) {
      auto name = portNames.newName("in_" + Twine(port.index()));
      ports.push_back({{b.getStringAttr(name), port.value().getType(),
                        hw::ModulePort::Direction::Input},
                       port.index()});
    }
    for (auto port : llvm::enumerate(realOutputs)) {
      auto name = portNames.newName("out_" + Twine(port.index()));
      ports.push_back({{b.getStringAttr(name), port.value().getType(),
                        hw::ModulePort::Direction::Output},
                       port.index()});
    }
  }
  for (auto p : ports) {
    llvm::outs() << "port  " << p << "\n";
  }

  // Create the module, setting the output path if indicated.
  auto newMod = b.create<hw::HWModuleOp>(op->getLoc(), moduleName, ports);

  // Update the mapping from old values to cloned values
  for (auto port : llvm::enumerate(realInputs)) {
    cutMap.map(port.value(), newMod.getBody().getArgument(port.index()));
    for (auto extra : realReads[port.value()])
      cutMap.map(extra, newMod.getBody().getArgument(port.index()));
  }
  cutMap.map(op.getBodyBlock(), newMod.getBodyBlock());

  return newMod;
}

// Some blocks have terminators, some don't
static void setInsertPointToEndOrTerminator(OpBuilder &builder, Block *block) {
  if (!block->empty() && isa<hw::HWModuleOp>(block->getParentOp()))
    builder.setInsertionPoint(&block->back());
  else
    builder.setInsertionPointToEnd(block);
}

// Shallow clone, which we use to not clone the content of blocks, doesn't
// clone the regions, so create all the blocks we need and update the mapping.
static void addBlockMapping(IRMapping &cutMap, Operation *oldOp,
                            Operation *newOp) {
  assert(oldOp->getNumRegions() == newOp->getNumRegions());
  for (size_t i = 0, e = oldOp->getNumRegions(); i != e; ++i) {
    auto &oldRegion = oldOp->getRegion(i);
    auto &newRegion = newOp->getRegion(i);
    for (auto oi = oldRegion.begin(), oe = oldRegion.end(); oi != oe; ++oi) {
      cutMap.map(&*oi, &newRegion.emplaceBlock());
    }
  }
}

//===----------------------------------------------------------------------===//
// StubExternalModules Pass
//===----------------------------------------------------------------------===//

struct GroupControlNodePass
    : public SpecHLS::impl::GroupControlNodePassBase<GroupControlNodePass> {
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
            llvm::outs() << "   Control input value " << controlValue << "\n";
            llvm::outs() << "   Control input defining op " << *controlOp
                         << "\n";

            /*
             * Slices control logic of gamma node
             */
            SetVector<Operation *> slice = {};
            getBackwardSliceSimple(controlOp, slice, [&](Operation *op) {
              llvm::outs() << " default filter  " << *op << "\n";
              bool res =
                  TypeSwitch<Operation *, bool>(op)
                      .Case<circt::comb::AndOp>([&](auto op) {
                        llvm::outs() << " found and " << *op << "\n";
                        return true;
                      })
                      .Case<circt::comb::OrOp>([&](auto op) { return true; })
                      .Case<circt::comb::XorOp>([&](auto op) { return true; })
                      .Case<circt::comb::ExtractOp>(
                          [&](auto op) { return true; })
                      .Case<circt::comb::ConcatOp>(
                          [&](auto op) { return true; })
                      .Case<circt::comb::MuxOp>([&](auto op) { return true; })
                      .Case<circt::comb::TruthTableOp>(
                          [&](auto op) { return true; })
                      .Case<SpecHLS::LookUpTableOp>(
                          [&](auto op) { return true; })
                      .Default([&](auto op) {
                        llvm::outs() << " default filter  " << *op << "\n";
                        return false;
                      });
              llvm::outs() << " res " << res << "\n";
              return res;
            });
            slice.insert(controlOp);
            llvm::outs() << "Slice for " << gamma << " contains "
                         << slice.size() << " ops\n";
            for (auto s : slice) {
              llvm::outs() << " - " << *s << "\n";
            }

            // Find the dataflow into the clone set
            SetVector<Value> inputs;
            SetVector<Value> outputs;
            for (auto *op : slice) {
              llvm ::outs() << " op in slide   " << *op << " to slice \n";
              for (auto arg : op->getResults()) {
                for (auto user : arg.getUsers()) {
                  llvm::outs() << " sink   " << *user << " to slice \n";
                  if (!slice.count(user))
                    // outputs.insert(arg);
                    llvm::outs()
                        << "register output  " << arg << " to slice \n";
                }
              }

              for (auto arg : op->getOperands()) {
                auto argOp = arg.getDefiningOp(); // may be null

                if (argOp==NULL) {
                  llvm::outs() << "  no source     to slice \n";
                  inputs.insert(arg);
                } else {
                  llvm::outs() << " source   " << *argOp << " to slice \n";
                  if (!slice.count(argOp))
                    inputs.insert(arg);
                  llvm::outs() << "register input  " << arg << " to slice \n";

                }
              }
            }

            //
            //
            // Make a module to contain the clone set, with arguments being the
            // cut

            auto builder = OpBuilder(op.getContext());

            IRMapping cutMap;
            auto moduleName =
                builder.getStringAttr(topModule.getName()+ "ctrl_" + std::to_string(gammaId));

            auto newModule = createModuleForCut(topModule, moduleName, inputs,
                                                outputs, cutMap);

            builder.setInsertionPointToStart(newModule.getBodyBlock());

            topModule.walk<WalkOrder::PreOrder>([&](Operation *op) {
              if (slice.count(op)) {
                auto newOp = builder.cloneWithoutRegions(*op, cutMap);
                addBlockMapping(cutMap, op, newOp);
              }
            });

            /*
             *
             */
            SetVector<Operation *> oplist = {};
            for (auto &s : newModule.getBodyBlock()->getOperations()) {
              oplist.insert(&s);
              llvm::outs() << "adding " << s << "\n";
            }

            int outputId = 0;
            for (auto s : oplist) {
              bool isOutputOp = true;
              llvm::outs() << "op " << *s << "\n";
              for (auto user : s->getUsers()) {
                llvm::outs() << *s << "-> " << *user << "\n";
                if (oplist.count(user)) {
                  isOutputOp = false;
                }
              }
              if (isOutputOp) {
                int numres = s->getNumResults();
                switch (numres) {
                case 0: {
                  llvm::outs() << " op with no output  " << *s << "\n";
                  break;
                }
                case 1: {
                  llvm::outs()
                      << "appending output for " << s->getResult(0) << "\n";
                  newModule.appendOutput(
                      builder.getStringAttr("out_" + std::to_string(outputId)),
                      s->getResult(0));
                  outputId++;
                  break;
                }
                default: {
                  llvm::errs() << "error op with non unit output  " << *s
                               << " size = " << s->getNumResults() << "\n";
                }
                }
              }
            }

            llvm::outs() << "module " << newModule << "\n";
            // Add an instance in the old module for the extracted module
            OpBuilder b =
                OpBuilder::atBlockTerminator(topModule.getBodyBlock());
            auto innerSym = hw::InnerSymAttr::get(b.getStringAttr("tot"));

            SmallVector<Value, 8> operands;
            for (auto i : inputs) {
              operands.push_back(i);
            }
            b.setInsertionPoint(gamma);
            auto inst = b.create<hw::InstanceOp>(
                controlOp->getLoc(), newModule,
                b.getStringAttr(newModule.getName()), operands, ArrayAttr());

            b = OpBuilder::atBlockEnd(
                &op.getParentOfType<mlir::ModuleOp>()->getRegion(0).front());

            gamma.setOperand(0, inst.getResult(0));

            newModule->setAttr(b.getStringAttr("#pragma"), b.getStringAttr("CONTROL_NODE"));
            //
            //  Register the newly created module in the instance graph.
            // instanceGraph->addHWModule(newModule);
            //
          }
        }
      }
    }
  }
}

namespace SpecHLS {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createGroupControlNodePass() {
  llvm::outs() << "GroupControlNodeImplPass created "
               << "\n";
  return std::make_unique<GroupControlNodePass>();
}
} // namespace SpecHLS
