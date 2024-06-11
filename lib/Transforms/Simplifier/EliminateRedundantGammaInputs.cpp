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
    size_t outerLutSize = 1 << (size_t)(std::ceil(log(nbInputs) / log(2)));
    auto firstMatchIndex = matches[0];
    SmallVector<int> outerLutContent;
    u_int32_t pos = 0;
    for (int k = 0; k <= firstMatchIndex; k++) {
      outerLutContent.push_back(k);
    }
    for (int k = firstMatchIndex + 1; k < nbInputs; k++) {
      if (std::count_if(matches.begin(), matches.end(),
                        [&](const auto &item) { return (k == item); })) {
        outerLutContent.push_back(firstMatchIndex);
      } else {
        if (pos == firstMatchIndex)
        {
          pos++;
        }
        outerLutContent.push_back(pos++);
      }
    }
    for (size_t k = nbInputs; k < outerLutSize; k++)
    {
      outerLutContent.push_back(pos);
    }

    if (verbose)
    {
      llvm::outs() << "Outer gamma " << op << " reindexing  \n";
      for (int k = 0; k < nbInputs; k++) {
        llvm::outs() << " - input " << k << " reindexed to "
          << outerLutContent[k] << " \n";
      }
    }

    return rewriter.create<SpecHLS::LookUpTableOp>(
        op->getLoc(), op.getSelect().getType(), op.getSelect(),
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
        auto gamma = rewriter.create<SpecHLS::GammaOp>(
            op->getLoc(), op->getResultTypes(), op.getName(), lut->getResult(0),
            args);

        llvm::errs() << "Simplifying  " << op << " into  " << gamma << "\n";
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
