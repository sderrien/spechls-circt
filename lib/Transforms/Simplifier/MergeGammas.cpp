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

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

namespace SpecHLS {

struct GammaMergingPattern : OpRewritePattern<GammaOp> {
  bool verbose = true;
  using OpRewritePattern<GammaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GammaOp outerGamma,
                                PatternRewriter &rewriter) const override {

    auto nbOuterInputs = outerGamma.getInputs().size();
    for (int i = 0; i < outerGamma.getInputs().size(); i++) {
      auto input = outerGamma.getInputs()[i].getDefiningOp();
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

          /* The pass first collects both inner and outer gamma operands and store
           * them into the muxOperands vector*/

          for (int pos = 0; pos < nbInnerInputs; pos++) {
            muxOperands.push_back(innerGamma.getInputs()[pos]);
          }
          for (int pos = 0; pos < nbOuterInputs; pos++) {
            if (pos != i) {
              muxOperands.push_back(outerGamma.getInputs()[pos]);
            }
          }

          auto outerGammaSelType = outerGamma.getSelect().getType();
          auto innerGammaSelType = innerGamma.getSelect().getType();

          auto controlType = rewriter.getIntegerType(
              outerGammaSelType.getIntOrFloatBitWidth() +
              innerGammaSelType.getIntOrFloatBitWidth());

          /* Cast select inputs into standard integers (in case they are unisgned) */
          auto castLeft = rewriter.create<SpecHLS::CastOp>(outerGamma.getLoc(),rewriter.getIntegerType(outerGammaSelType.getIntOrFloatBitWidth()), outerGamma.getSelect());
          auto castRight = rewriter.create<SpecHLS::CastOp>(innerGamma.getLoc(),rewriter.getIntegerType(innerGammaSelType.getIntOrFloatBitWidth()),innerGamma.getSelect());


          /* Create merged select command from inner and  outer gamma select inputs */
          auto concatOp = rewriter.create<comb::ConcatOp>(
              outerGamma.getLoc(), controlType,
              ValueRange({castLeft, castRight}));


          ArrayAttr tab;
          SmallVector<int, 1024> content;

          int innerSelectBW = innerGammaSelType.getIntOrFloatBitWidth();
          int outerSelectBW = outerGammaSelType.getIntOrFloatBitWidth();
          int offset = 0;

          int innerPow2Inputs = 1 << innerSelectBW;
          int outerPow2Inputs = 1 << outerSelectBW;

          // FIXME :
          auto outerUB = outerGamma.getInputs().size() - 1;
          auto innerUB = innerGamma.getInputs().size() - 1;

          /* Fills the LookupTable with the reindexing information */

          for (int o = 0; o < i; o++) {
            for (int inner = 0; inner < innerPow2Inputs; inner++) {
              if (verbose) llvm::outs() << "rewiring outer " << o << " to " << offset << " at " << content.size() << "\n";
              content.push_back(offset);
            }
            if (o < outerUB)
              offset++;
          }

          for (int inner = 0; inner < innerPow2Inputs; inner++) {
            content.push_back(offset);
            if (verbose) llvm::outs() << "rewiring inner " << inner << " to " << offset << " at " << content.size() << "\n";
            if (inner < innerUB)
              offset++;
          }

          for (int o = i + 1; o < outerPow2Inputs; o++) {
            for (int inner = 0; inner < innerPow2Inputs; inner++) {
              content.push_back(offset);
              if (verbose) llvm::outs() << "rewiring outer " << o << " to " << offset << " at " << content.size() << "\n";
            }
            if (o < outerUB)
              offset++;
          }


          int lutWidth = APInt(32, muxOperands.size()-1).getActiveBits();
          if (verbose) llvm::outs() << "LUT content size " << content.size() << " -> address width " << lutWidth << "\n";

          for (int o = content.size(); o < (1<<lutWidth); o++) {
            if (verbose)  llvm::outs() << "padding lut content at " << content.size() << "with"  << offset << "\n";
            content.push_back(offset);
          }


          mlir::Type lutType = rewriter.getIntegerType(lutWidth);


          auto lutSelect = rewriter.create<LookUpTableOp>(
              outerGamma.getLoc(), lutType, concatOp.getResult(),
              rewriter.getI32ArrayAttr(content));

          auto newGammaOp = rewriter.create<GammaOp>(
              outerGamma.getLoc(), outerGamma.getResult().getType(), outerGamma.getNameAttr(),
              lutSelect.getResult(), ValueRange(muxOperands));

          if (verbose) llvm::errs() << "Merging " << innerGamma << " with " << outerGamma << " into "
                       << newGammaOp << "\n";

          auto parent = outerGamma->getParentOp();
          rewriter.replaceOp(outerGamma, newGammaOp);
          if (nbUsers == 1)
            rewriter.eraseOp(innerGamma);
          mlir::verify(newGammaOp, true);
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

    mlir::verify(getOperation(), true);
  }
};

std::unique_ptr<OperationPass<>> createMergeGammasPass() {
  return std::make_unique<MergeGammasPass>();
}
} // namespace SpecHLS