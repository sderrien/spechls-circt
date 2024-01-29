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

struct FactorGammaInputsPattern : OpRewritePattern<GammaOp> {

  using OpRewritePattern<GammaOp>::OpRewritePattern;
private:
  bool checkmatch(GammaOp op, int i, int k, SmallVector<int32_t> matches) const {
    Value va = op.getOperand(i);
    Value vb = op.getOperand(k);

    if (va.getType() == vb.getType()) {
      auto a  = va.getDefiningOp();
      auto b  = vb.getDefiningOp();
      if (b != NULL) {
        if (a->getName() == b->getName()) {
          if (a->getDialect() == b->getDialect()) {
            if (a->getNumOperands() == b->getNumOperands()) {
              if (b->getNumResults()==1 && b->getResult(0).hasOneUse()) {
                llvm::outs() << "Found match " << *a  << "and "<< *b << "\n";
                matches.push_back(k);
                return true;
              }
            }
          }
        }
      }
    }
    return false;
  }

public:
  LogicalResult matchAndRewrite(GammaOp op,
                                PatternRewriter &rewriter) const override {

    // llvm::errs() << "analyzing  " << op << " \n";
    for (int i = 1; i < op.getInputs().size(); i++) {
      auto rootValue = op.getInputs()[i];
      auto root = rootValue.getDefiningOp();
      if (root!=NULL) {
        if (root->getNumResults()==1) {
          if (root->getResult(0).hasOneUse()) {
            SmallVector<int32_t> matches;
            matches.push_back(i);
            for (int k = i + 1; i < op->getNumOperands(); i++) {
              checkmatch(op, i, k, matches);
            }
            if (matches.size()>1) {
              llvm::outs() << "match set :\n";

              for (auto m : matches) {
                llvm::outs() << "   - "<< m << "\n";
              }

              /*
               * creates a LUT for reindexing innergamme inputs
               */
              SmallVector<int> lutcontent;
              int pos = 0;
              for (int k= 0; k<root->getNumOperands(); k++) {
                if (std::count_if(matches.begin(), matches.end(), [&](const auto &item) {
                      return (k==item);
                    })) {
                  lutcontent.push_back(pos++);
                } else {
                  lutcontent.push_back(0);
                }
              }
              auto innerLut = rewriter.create<SpecHLS::LookUpTableOp>(op->getLoc(),op->getResultTypes(),op.getOperand(0), rewriter.getI32ArrayAttr(lutcontent));


              /*
               * creates a LUT for reindexing outergamme inputs
               */
              SmallVector<int> outerLutContent;
              pos = 0;
              for (int k= 0; k<=i; k++) {
                outerLutContent.push_back(k);
              }
              for (int k= i+1; k<op->getNumOperands(); k++) {
                if (std::count_if(matches.begin(), matches.end(), [&](const auto &item) {
                      return (k==item);
                    })) {
                  outerLutContent.push_back(i);
                } else {
                  outerLutContent.push_back(pos++);
                }
              }

              llvm::outs() << "Outer gamma "<< op<< "reindexing  \n";
              for (int k= 0; k<op->getNumOperands(); k++) {
                llvm::outs() << " - input " << k <<" reindexed to "<< outerLutContent[k] <<" \n";
              }

              auto outerLut = rewriter.create<SpecHLS::LookUpTableOp>(op->getLoc(),op->getResultTypes(),op.getOperand(0), rewriter.getI32ArrayAttr(outerLutContent));

              /*
               * Create a LUT for reindexing inner gamma inputs.
               *
               */

              SmallVector<Value> newGammas;
              for (int j=0;j<root->getNumOperands();j++) {
                SmallVector<Value> args;
                llvm::outs() << "extracting all " << j <<"th args in matched ops \n";

                for (auto mid : matches) {
                  auto matchedValue = op.getOperand(mid);
                  if (!matchedValue) {
                    auto matchedOp = matchedValue.getDefiningOp();
                    if (!matchedOp) {
                      llvm::outs() << " analyzing match "<< matchedOp <<" at offset  "<< j << "\n";
                      auto matchedArgValue = matchedOp->getOperand(j);
                      if (!matchedArgValue) {
                        llvm::outs() << "     extracting value "<< matchedArgValue << " \n";
                        args.push_back(matchedArgValue);
                      } else {
                        llvm::errs() << "No valid arrgValue at offset " << j << "\n";
                        return failure();
                      }
                    } else {
                      llvm::errs() << "No defining op for value "<< matchedValue << " at offset " << mid << "\n";
                      return failure();
                    }
                  } else {
                    llvm::errs() << "Invalid value at " << mid << "\n";
                    return failure();
                  }
                }
                auto gamma = rewriter.create<SpecHLS::GammaOp>(op->getLoc(),op->getResultTypes(), op.getName(),innerLut->getResult(0),args);
                llvm::outs() << "creating inner gamma "<< gamma<<" at offset  "<< j << "\n";
                newGammas.push_back(gamma);
              }

              assert(newGammas.size()==root->getNumOperands());
              for (int j=0;j<root->getNumOperands();j++) {
                llvm::outs() << "replace arg["<<j <<"]="<< root->getOperand(j) <<" by  "<< newGammas[j] << "\n";
                root->setOperand(j,newGammas[j]);
              }
              llvm::outs() << "Root op is now "<< root << "\n";

              for (auto mid : matches) {
                llvm::outs() << "change value  "<< op->getOperand(mid) << " by "<< rootValue << "\n";
                op->setOperand(mid,rootValue);
              }
              llvm::outs() << "After rewiring outer gamma "<< op << "\n";
              return success();
            }
          }
        }
      }
    }
    return failure();
  }
};

struct FactorGammaInputsPass
    : public impl::FactorGammaInputsPassBase<FactorGammaInputsPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    patterns.insert<FactorGammaInputsPattern>(ctx);
    // llvm::errs() << "inserted pattern  \n";

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      llvm::errs() << "partial conversion failed pattern  \n";
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<>> createFactorGammaInputsPass() {
  return std::make_unique<FactorGammaInputsPass>();
}
} // namespace SpecHLS
