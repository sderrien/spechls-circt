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
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

namespace SpecHLS {

struct GammaMergingPattern : OpRewritePattern<GammaOp> {

  using OpRewritePattern<GammaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GammaOp op,
                                PatternRewriter &rewriter) const override {

    auto nbOuterInputs = op.getInputs().size();
    for (int i = 0; i < op.getInputs().size(); i++) {
      auto input = op.getInputs()[i].getDefiningOp();
      if (input != NULL) {
        auto innerGamma = dyn_cast<SpecHLS::GammaOp>(input);
        if (innerGamma) {

          auto users = innerGamma->getUsers();
          auto nbUsers = std::distance(users.begin(), users.end());


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

          int innerSelectBW = innerGamma.getSelect().getType().getIntOrFloatBitWidth();
          int outerSelectBW = op.getSelect().getType().getIntOrFloatBitWidth();
          int offset = 0;

          int innerPow2Inputs = 1 << innerSelectBW;
          int outerPow2Inputs = 1 << outerSelectBW;


          // FIXME :
          auto outerUB = op.getInputs().size()-1;
          auto innerUB = innerGamma.getInputs().size()-1;
          for (int o = 0; o < i; o++) {
            for (int inner = 0; inner < innerPow2Inputs; inner++) {
              content.push_back(offset);
            }
            if (o<outerUB) offset++;
          }

          for (int inner = 0; inner < innerPow2Inputs; inner++) {
            content.push_back(offset);
            if (inner<=innerUB) offset++;
          }

          for (int o = i+1; o < outerPow2Inputs; o++) {
            for (int inner = 0; inner < innerPow2Inputs; inner++) {
              content.push_back(offset);
            }
            if (o<outerUB) offset++;
          }


          int lutWidth = APInt(32, offset).getActiveBits();
          mlir::Type lutType = rewriter.getIntegerType(lutWidth);

          auto lutSelect = rewriter.create<LookUpTableOp>(
              op.getLoc(), lutType, concatOp.getResult(),
              rewriter.getI32ArrayAttr(content));

          auto newGammaOp = rewriter.create<GammaOp>(
              op.getLoc(), op.getResult().getType(), op.getNameAttr(),
              lutSelect.getResult(), ValueRange(muxOperands));

          llvm::errs() << "Merging " << innerGamma << " with " << op
                       << " into " << newGammaOp << "\n";

          auto parent = op->getParentOp();
          rewriter.replaceOp(op, newGammaOp);
          if (nbUsers == 1)
            rewriter.eraseOp(innerGamma);
          mlir::verify(newGammaOp,true);
          parent->dump();
          return success();
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

    mlir::verify(getOperation(),true);

  }
};

std::unique_ptr<OperationPass<>> createMergeGammasPass() {
  return std::make_unique<MergeGammasPass>();
}
} // namespace SpecHLS
