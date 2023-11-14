//===- MergeLookUpTables.cpp - Arith-to-comb mapping pass ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the MergeLookUpTables pass.
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

struct LookUpMergingPattern : OpConversionPattern<LookUpTableOp> {

  using OpConversionPattern<LookUpTableOp>::OpConversionPattern;

  ArrayAttr updateLUTContent(ArrayAttr inner, ArrayAttr outer, ConversionPatternRewriter &rewriter) const {
    SmallVector<int, 1024> newcontent;
    int innerSize = inner.size();
    int outerSize = outer.size();
    for (int o = 0; o < innerSize; o++) {

      if (o > inner.size()) {
        llvm::errs() << "out of bound access at " << o << " for " << inner << "  \n";
        return NULL;
      }
      auto innerValue =
          cast<IntegerAttr>(inner.getValue()[o]).getInt();
      if (innerValue > outer.size()) {
        llvm::errs() << "out of bound access at " << innerValue << " for " << outer << "  \n";
        return NULL;
      }

      auto outerValue =
          cast<IntegerAttr>(outer.getValue()[innerValue]).getInt();
      newcontent.push_back(outerValue);
    }
    return rewriter.getI32ArrayAttr(newcontent);
  }

  LogicalResult matchAndRewrite(LookUpTableOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

    llvm::errs() << "Analyzing  " << op << " \n";
    auto input = op.getInput().getDefiningOp();
    if (input!=NULL && llvm::isa<SpecHLS::LookUpTableOp>(input)) {
      auto inputLUT = cast<SpecHLS::LookUpTableOp>(input);
      llvm::errs() << "Found nested LUTs \n";
      llvm::errs() << "\t " << op << "  \n";
      llvm::errs() << "\t " << input << "  \n";
      int innerSize = inputLUT.getContent().size();
      int outerSize = op.getContent().size();
      int innerWL = inputLUT.getType().getWidth();
      int outerWL = op.getType().getWidth();
      auto lutSelect = rewriter.create<LookUpTableOp>(op.getLoc(), op->getResultTypes(), input->getOperands());
      // FIXME : Why can't I directly create A LUT with Attrbute using the builder ?
      ArrayAttr newAttr = updateLUTContent(inputLUT.getContent(),op.getContent(),rewriter);
      lutSelect.setContentAttr(newAttr);

      llvm::errs() << "\t- created LUT  " << lutSelect << "\n";

      rewriter.replaceOp(op, lutSelect);
      rewriter.eraseOp(inputLUT);

      llvm::errs() << "\t-sucess ?  " << lutSelect << "\n";
      return success();
    }

    return failure();
  }
};

struct MergeLookUpTablesPass : public impl::MergeLookUpTablesPassBase<MergeLookUpTablesPass> {
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
    patterns.insert<LookUpMergingPattern>(ctx);
    llvm::errs() << "inserted pattern  \n";

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      llvm::errs() << "partial conversion faile pattern  \n";
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<>> createMergeLookUpTablesPass() {
  return std::make_unique<MergeLookUpTablesPass>();
}
} // namespace SpecHLS
