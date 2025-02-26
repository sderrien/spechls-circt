//===- EliminateRedundantGammaInputs.cpp - Arith-to-comb mapping pass -*- C++ -*-===//
//  
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//  
//===----------------------------------------------------------------------------===//
//  
// Contains the definitions of the EliminateRedundantGammaInputs pass.
//  
//===----------------------------------------------------------------------------===//

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

namespace SpecHLS 
{

struct EliminateRedundantGammaInputs : OpRewritePattern<GammaOp> {
  using OpRewritePattern<GammaOp>::OpRewritePattern;
  bool verbose = false;

private:
  //
  //
  //  Cette fonction construit une liste avec la position des arguments produit
  //  par des op d'une même classe d'equivalence (definie par checkmatch)
  //
  LogicalResult extractMatches(GammaOp op,
                               SmallVector<int32_t> &matches) const {
    // llvm::errs() << "analyzing  " << op << " \n";
    auto nbInputs = op.getInputs().size();
    for (uint32_t i = 0; i < nbInputs; i++) {
      auto rootValue = op.getInputs()[i];
      if (rootValue != NULL) {
        matches.clear();
        matches.push_back(i);
        for (uint32_t k = i + 1; k < nbInputs; k++) {
          if (op.getInputs()[k] == rootValue) {
            matches.push_back(k);
          }
        }
        if (matches.size() > 1) {
          return success();
        }
      }
    }

    return failure();
  }

  SpecHLS::LookUpTableOp
  createOuterReindexingLUT(GammaOp op, SmallVector<int32_t> &matches,
                           PatternRewriter &rewriter) const {
    /*
     * creates a LUT for reindexing outer Gamma inputs, by skipping
     * inputs that have hoisted out to the inner Gamma
     */
    auto nbInputs = op.getInputs().size();
    size_t outerLutSize = 1<<(op.getSelect().getType().getWidth());
    if (verbose) {
      llvm::errs() << "## createOuterReindexingLUT for " << op << "\n";
      if (verbose) {
        llvm::errs() << "Reindexing Outer gamma " << op << "  \n";
        llvm::errs() << "LUT size  " << outerLutSize << "  \n";
        llvm::errs() << "outerLutSize  " << outerLutSize << "  \n";
      }
    }
    auto firstMatchIndex = matches[0];
    SmallVector<int> outerLutContent;
    for (int k = 0; k <= firstMatchIndex; k++) {
      outerLutContent.push_back(k);
      if (verbose) llvm::errs() << " - input " << op.getInputs()[k] << " reindexed to " << k << " \n";
    }
    u_int32_t pos = firstMatchIndex;
    for (int k = firstMatchIndex + 1; k < nbInputs; k++) {
      if (std::count_if(matches.begin(), matches.end(),
                        [&](const auto &item) { return (k == item); })) {
        if (verbose) llvm::errs() << " - input " << op.getInputs()[k] << " reindexed to " << firstMatchIndex << " \n";
        outerLutContent.push_back(firstMatchIndex);
      } else {
        if (pos == firstMatchIndex)
        {
          pos++;
        }
        outerLutContent.push_back(pos++);
        if (verbose) {
          if (k<op.getInputs().size())
            llvm::errs() << " - input " << op.getInputs()[k] << " reindexed to " << outerLutContent[k] << " \n";
        }
      }
    }
    for (size_t k = nbInputs; k < outerLutSize; k++)
    {
      outerLutContent.push_back(pos);
      if (verbose)
          llvm::errs() << " - don't care input " << k << " reindexed to " << outerLutContent[k] << " \n";
    }

    size_t maxIndex = 0;
    for (size_t k = 0; k < outerLutContent.size(); k++)  {
      if (outerLutContent[k]>maxIndex)
        maxIndex = outerLutContent[k];
    }

    auto selWidth = op.getSelect().getType().getWidth();
    int lutOutputWidth = APInt(32, maxIndex).getActiveBits();

    if (lutOutputWidth<selWidth) {
      if (verbose) {
        llvm::errs() << "Resizing lut address witdh from "<< selWidth <<" to "<< lutOutputWidth <<" (maxIndex = "<< maxIndex<<"\n";
      }

      selWidth=lutOutputWidth;
    }

    auto lutAddressType = rewriter.getIntegerType(selWidth);

    return rewriter.create<SpecHLS::LookUpTableOp>(
        op->getLoc(), lutAddressType, op.getSelect(),
        rewriter.getI32ArrayAttr(outerLutContent));
  }

public:
  LogicalResult matchAndRewrite(GammaOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<int32_t> matches;
    u_int32_t nbInputs = op.getInputs().size();
    if (extractMatches(op, matches).succeeded()) {
      if (matches.size() == nbInputs) {
        llvm::errs() << "Eliminating " << op
                     << " because it has all the same inputs :\n";
        op.getResult().replaceAllUsesWith(op.getInputs()[0]);
        rewriter.eraseOp(op);
        return success();
      } else {
        if (verbose) {
          llvm::errs() << "## eliminate redundant gamma inputs " << op << "\n";
        }

        auto lut = createOuterReindexingLUT(op, matches, rewriter);

        // filter out redundant input values
        SmallVector<Value> args;
        for (int32_t k = 0; k < nbInputs; k++) {
          bool found = false;
          for (u_int32_t j = 1; j < matches.size(); j++) {
            if (k == matches[j]) {
              found = true;
              break;
            }
          }
          if (!found) {
            args.push_back(op.getInputs()[k]);
          }
        }
        auto gamma = rewriter.create<SpecHLS::GammaOp>(op->getLoc(), op->getResultTypes(), op.getName(), lut->getResult(0), args);
        if (verbose) {
          llvm::errs() << "Simplifying  " << op << " into  " << gamma << "\n";
        }

        verify(gamma);

        rewriter.replaceOp(op, gamma);
        return success();
      }

    } else {
      return failure();
    }
  }
};


struct EliminateRedundantGammaInputsPass
    : public impl::EliminateRedundantGammaInputsPassBase<
          EliminateRedundantGammaInputsPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);

//    patterns.insert<FactorGammaInputsPattern>(ctx);
    patterns.insert<EliminateRedundantGammaInputs>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      llvm::errs() << "partial conversion failed pattern  \n";
      signalPassFailure();
    }
    mlir::verify(getOperation(), true);
  }
};
std::unique_ptr<OperationPass<>> createEliminateRedundantGammaInputsPass() {
  return std::make_unique<EliminateRedundantGammaInputsPass>();
}

} // namespace SpecHLS
