#include "mlir/Pass/Pass.h"

#include "Common/OutliningUtils.h"
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
#include "llvm/ADT/SetVector.h"

#include <set>

using namespace mlir;
using namespace circt;



struct ExtractSpeculativeThreadsPass : public SpecHLS::impl::ExtractSpeculativeThreadsPassBase<ExtractSpeculativeThreadsPass> {
  int max_bitwidth = 4;
  bool verbose= true;
  ExtractSpeculativeThreadsPass() {}

public:
  void runOnOperation() override;
};
//
void ExtractSpeculativeThreadsPass::runOnOperation() {
  auto top = getOperation();


  auto *topLevelModule = top.getBody();
  int gammaId = 0;

  for (auto &op : topLevelModule->getOperations()) {
    if (auto topModule = dyn_cast<hw::HWModuleOp>(op)) {
      if (!topModule.getBody().empty()) {

        for (auto &innerOp : topModule.getBodyBlock()->getOperations()) {
          if (auto gamma = dyn_cast<SpecHLS::GammaOp>(innerOp)) {
            if (!(gamma->getNumOperands() > 0)) {
              continue;
            }
            auto controlValue = gamma->getOperand(0);
            auto controlOp = controlValue.getDefiningOp();

            /*
             * Slices control logic of gamma node
             */
            SetVector<Operation *> slice = {};
            auto opfilter = [&](Operation *op) {
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
              return res;
            };

            if (!opfilter(controlOp)) {
              continue;
            }

            SetVector<Value*> inputs;
            getBackwardSlice(*controlOp, slice, inputs, opfilter);


            SetVector<Value*> outputs;
            for (auto res : controlOp->getResults()) {
              outputs.insert(&res);
            }
            auto newName =
                topModule.getName() + "_ctrl_" + std::to_string(gammaId++);
            auto newModule = outlineSliceAsHwModule(
                topModule,  slice, inputs, outputs, newName);
            if (newModule) {
              auto builder = OpBuilder(topModule.getContext());

              SmallVector<Value, 8> operands;
              for (auto i : inputs) {
                operands.push_back(*i);
              }

              builder.setInsertionPoint(gamma);

              auto inst = builder.create<hw::InstanceOp>(
                  controlOp->getLoc(), newModule,
                  builder.getStringAttr(newModule.getName()), operands,
                  ArrayAttr());

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
createExtractSpeculativeThreadsPass() {
  // llvm::outs() << "ExtractSpeculativeThreadsImplPass created " << "\n";
  return std::make_unique<ExtractSpeculativeThreadsPass>();
}
} // namespace SpecHLS
