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

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

namespace SpecHLS {

struct GammaMergingPattern : OpRewritePattern<GammaOp> {

  using OpRewritePattern<GammaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GammaOp op, PatternRewriter &rewriter) const override {

    auto nbOuterInputs = op.getInputs().size();
    for (int i = 0; i < op.getInputs().size(); i++) {
      auto input = op.getInputs()[i].getDefiningOp();
      if (input != NULL) {
    if (llvm::isa<SpecHLS::GammaOp>(input)) {

          auto innerGamma = cast<SpecHLS::GammaOp>(input);

          auto users = innerGamma->getUsers();
          auto nbUsers = std::distance(users.begin(), users.end());

          if (nbUsers == 1) {

            auto nbInnerInputs = innerGamma.getInputs().size();
            int newDepth = int(ceil(log(nbOuterInputs + nbInnerInputs) / log(2)));
            //  Value* muxOperands = new Value[cwidth * (cwidth+1)];
            Operation::operand_range in = innerGamma.getInputs();
            SmallVector<Value, 8> muxOperands;
            // muxOperands.append(innerGamma.getInputs())

            for (int pos = 0; pos < nbInnerInputs; pos++) {
              muxOperands.push_back(innerGamma.getInputs()[pos]);
            }
            for (int pos = 0; pos < nbOuterInputs; pos++) {
              if (pos != i) {
                muxOperands.push_back(op.getInputs()[pos]);
              }
            }
            auto controlType = rewriter.getIntegerType(
                op.getSelect().getType().getIntOrFloatBitWidth() +
                innerGamma.getSelect().getType().getIntOrFloatBitWidth());

            auto concatOp = rewriter.create<comb::ConcatOp>(
                op.getLoc(), controlType,
                ValueRange({op.getSelect(), innerGamma.getSelect()}));

            ArrayAttr tab;
            SmallVector<int, 1024> content;
            int offset = 0;
            int index = 0;
            int innerPow2Inputs =
                1 << innerGamma.getSelect().getType().getIntOrFloatBitWidth();
            int outerPow2Inputs =
                1 << op.getSelect().getType().getIntOrFloatBitWidth();
            for (int o = 0; o < outerPow2Inputs; o++) {

              for (int inner = 0; inner < innerPow2Inputs; inner++) {
                // llvm::errs() << "\t\t- LUT["<< inner << "," << o << "->" <<
                // index << "] =" << offset << "\n";
                content.push_back(offset);
                index++;
                if (o == i)
                  offset++;
              }
              if (o != i)
                offset++;
            }

            int depth =
                int(ceil(log(nbInnerInputs + nbOuterInputs - 1) / log(2)));
            mlir::Type addrType = rewriter.getIntegerType(depth);

            auto lutSelect = rewriter.create<LookUpTableOp>(
                op.getLoc(), addrType, concatOp.getResult(),
                rewriter.getI32ArrayAttr(content));

            rewriter.replaceOpWithNewOp<GammaOp>(op, op->getResultTypes(),op.getName(),
                                                 lutSelect.getResult(),
                                                 ValueRange(muxOperands));

            lutSelect->getParentOfType<mlir::ModuleOp>().dump();

            return success();
          }
        }
      }
    }

    return failure();
  }
};

struct MergeGammasPass : public impl::MergeGammasPassBase<MergeGammasPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    patterns.insert<GammaMergingPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      llvm::errs() << "partial conversion failed pattern  \n";
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<>> createMergeGammasPass() {
  return std::make_unique<MergeGammasPass>();
}
} // namespace SpecHLS
