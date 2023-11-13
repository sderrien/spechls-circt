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
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

namespace SpecHLS {

struct GammaMergingPattern : OpConversionPattern<GammaOp> {

  using OpConversionPattern<GammaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GammaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::errs() << "analyzing  " << op << " \n";
    auto nbOuterInputs = op.getInputs().size();
    for (int i = 0; i < op.getInputs().size(); i++) {
      llvm::errs() << "input  value " << op.getInputs()[i] << " \n";
      auto input = op.getInputs()[i].getDefiningOp();
      if (input!=NULL) {

        llvm::errs() << "input op " << input << " \n";

        if (llvm::isa<SpecHLS::GammaOp>(input)) {
          llvm::errs() << "Found nested gamma \n";
          auto innerGamma = cast<SpecHLS::GammaOp>(input);

          llvm::errs() << " inner " << innerGamma << "\n";
          llvm::errs() << " outer " << op << "\n";

          auto nbInnerInputs = innerGamma.getInputs().size();
          int newDepth = int(ceil(log(nbOuterInputs + nbInnerInputs) / log(2)));
          //  Value* muxOperands = new Value[cwidth * (cwidth+1)];
          Operation::operand_range in = innerGamma.getInputs();
          SmallVector<Value, 8> muxOperands;
          // muxOperands.append(innerGamma.getInputs())

          for (int pos = 0; pos < nbInnerInputs; pos++) {
            muxOperands.push_back(op.getInputs()[pos]);
          }
          for (int pos = 0; pos < nbOuterInputs; pos++) {
            if (pos != i) {
              muxOperands.push_back(op.getInputs()[pos]);
            }
          }
          auto newSelect = rewriter.create<comb::ConcatOp>(
              op.getLoc(), op->getResultTypes(),
              ValueRange({op.getSelect(), innerGamma.getSelect()}));

          llvm::errs() << "created concat " << newSelect << "\n";
          ArrayAttr tab;

          auto lutSelect =
              rewriter.create<LookUpTableOp>(op.getLoc(), op->getResultTypes(),
                                             ValueRange(newSelect.getResult()));

          llvm::errs() << "created Lut " << lutSelect << "\n";

          // newSelect.setContentAttr(tab);
          int depth = int(ceil(log(nbInnerInputs+nbOuterInputs-1)/log(2)));
          mlir::Type addrType = rewriter.getIntegerType(depth);
          auto newGamma =
              rewriter.create<GammaOp>(op.getLoc(), addrType,
                                       op.getSelect(), ValueRange(muxOperands));

          llvm::errs() << "created newGamma  " << newGamma << "\n";

          rewriter.replaceOp(op, newGamma);

          llvm::errs() << " op preds " << op->getPrevNode() << "\n";
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
    //
    ConversionTarget target(*ctx);
    target.addLegalDialect<comb::CombDialect, hw::HWDialect>();
    //    target.addIllegalDialect<arith::ArithDialect>();
    //    MapArithTypeConverter typeConverter;
    RewritePatternSet patterns(ctx);
    //
    patterns.insert<GammaMergingPattern>(ctx);
    llvm::errs() << "inserted pattern  \n";

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      llvm::errs() << "partial conversion faile pattern  \n";
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<>> createMergeGammasPass() {
  return std::make_unique<MergeGammasPass>();
}
} // namespace SpecHLS
