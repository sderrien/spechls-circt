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
using namespace comb;

namespace SpecHLS {

struct SimplifyCastToExtract : OpRewritePattern<CastOp> {

  using OpRewritePattern<CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp op,  PatternRewriter &rewriter) const override {

        auto outType = dyn_cast<IntegerType>(op.getResult().getType());
        auto inType = dyn_cast<IntegerType>(op.getOperand().getType());

        if (inType && outType) {
          uint32_t inWL  = inType.getWidth();
          uint32_t outWL  = outType.getWidth();
          // WL reduction cast : we replace it by an extract node
          if (inWL > outWL && inType.isSignless())  {
            llvm::errs() << "Simplify  " << op << "\n";
            auto newOutType = rewriter.getIntegerType(outWL);
            auto extract = rewriter.create<ExtractOp>(op.getLoc(), newOutType,op.getOperand(),0);
            if (outType.isSignless()) {
              llvm::errs() << " : replace by " << extract << "\n";
              rewriter.replaceOp(op,extract);
            } else {
              llvm::errs() << " : insert " << extract << " and update "<<op << "\n";
              op.setOperand(extract.getResult());
            }

            return success();
          }

          if (outWL > inWL) {
            // WL expansion cast : we replace it by ConcatOp node and perform zero/sign extension
            auto narrowCast = rewriter.create<CastOp>(op.getLoc(),rewriter.getIntegerType(inWL),op.getOperand());

            if (outType.isUnsigned()) {
              auto zero = rewriter.create<hw::ConstantOp>(
                  op.getLoc(), rewriter.getIntegerType(outWL - inWL), 0);
              auto concat = rewriter.create<ConcatOp>(
                  op.getLoc(), narrowCast, zero.getResult());
              llvm::errs() << " replace " << op << " by "  << zero << " and  "<<concat << "\n";
              rewriter.replaceOp(op, concat);
            } else  {
              // sign extension
              auto msb = rewriter.create<ExtractOp>(op.getLoc(),rewriter.getIntegerType(1),narrowCast,inWL-1);
              auto replicate = rewriter.create<ReplicateOp>(op.getLoc(),msb,(outWL-inWL));
              auto concat = rewriter.create<ConcatOp>(op.getLoc(),narrowCast,replicate);
              llvm::errs() << " replace " << op << " by "  << msb << " and  "<< replicate << " and  "<< concat << "\n";
              rewriter.replaceOp(op,concat);
            }
            return success();
          }
        }
    return failure();
  }

};

struct SimplifyCastsPass: public impl::SimplifyCastsPassBase<SimplifyCastsPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    patterns.insert<SimplifyCastToExtract>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      llvm::errs() << "partial conversion failed pattern  \n";
      signalPassFailure();
    }
    mlir::verify(getOperation(), true);
  }
};
std::unique_ptr<OperationPass<>> createSimplifyCastsPass() {
  return std::make_unique<SimplifyCastsPass>();
}

} // namespace SpecHLS
