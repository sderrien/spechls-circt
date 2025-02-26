#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Transforms/Passes.h"
#include "Transforms/SpecHLSConversion.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"

#include <Common/DefUSeSCC.h>
#include <Common/OutliningUtils.h>

using namespace mlir;
using namespace circt;

namespace SpecHLS {

struct OutlineSCCPass : public impl::OutlineSCCPassBase<OutlineSCCPass> {
  void runOnOperation() override;
};

void OutlineSCCPass::runOnOperation() {

  auto top = getOperation();

  auto *topLevelModule = top.getBody();
  int gammaId = 0;

  for (auto &op : topLevelModule->getOperations()) {
    if (auto HKernelOp = dyn_cast<SpecHLS::HKernelOp>(op)) {

      auto &block = HKernelOp.getRegion().front();

      SCCComputer comp;
      auto sccs = comp.computeSCCs(block);
      int scc_id = 0;
      for (auto scc : sccs) {
        auto candidateOps = scc.operations;
        auto size = scc.operations.size();

        /* add operations that have no predecessors to cndidate lits (typpically
         * constant and init operations) */
        for (Operation *op : candidateOps) {
          for (auto operand : op->getOperands()) {
            if (auto defOp = operand.getDefiningOp()) {
              if (defOp->getNumOperands() == 0) {
                scc.operations.push_back(defOp);
              }
            }
          }
        }

        // llvm::errs() << "SCC " << scc << "\n";
        if (candidateOps.size() <= 1) {
          for (auto op : candidateOps) {
            llvm::errs() << "op " << *op << "\n";
          }

        } else {
          llvm::errs() << "SCC with " << size << " nodes\n";
          // Collect inputs and outputs for the SCC
          SetVector<Value> inputs, outputs;
          SetVector<Operation *> sccOps;

          for (Operation *op : candidateOps) {
            llvm::errs() << " - " << *op << "\n";
            sccOps.insert(op);
          }
          for (Operation *op : candidateOps) {
            for (Value operand : op->getOperands()) {
              if (std::find(candidateOps.begin(), candidateOps.end(),
                            operand.getDefiningOp()) == candidateOps.end())
                inputs.insert(operand);
            }
            for (Value result : op->getResults()) {
              for (Operation *user : result.getUsers()) {
                if (std::find(candidateOps.begin(), candidateOps.end(), user) ==
                    candidateOps.end())
                  outputs.insert(result);
              }
            }
          }

          // Create the new HWModuleOp for the SCC

          auto region = &HKernelOp.getBodyRegion();
          OpBuilder builder(region);
          //
          SmallVector<Type, 8> inputTypes, outputTypes;

          for (Value input : inputs)
            inputTypes.push_back(input.getType());

          for (Value output : outputs)
            outputTypes.push_back(output.getType());

          Twine name =
              HKernelOp.getNameAttr().getValue().str() + "_SCC_" + std::to_string(scc_id);

          auto hthread =
              outlineSliceAsHwThread(HKernelOp, sccOps, inputs, outputs, name);

          SmallVector<Value, 8> callOperands(inputs.begin(), inputs.end());

          // Replace the outputs in the original module with the results of the
          // call
          for (auto result : llvm::enumerate(outputs)) {
            auto res = result.value();
            res.replaceAllUsesWith(hthread.getResult(result.index()));
          }
        }

        llvm::errs() << HKernelOp;
        mlir::verify(HKernelOp);
      }
    }
  }
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOutlineSCCPass() {
  return std::make_unique<OutlineSCCPass>();
}

} // namespace SpecHLS
