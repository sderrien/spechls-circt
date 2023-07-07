//===- SpecHLSToComb.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include "SpecHLS/SpecHLSOps.h"
#include "SpecHLS/SpecHLSDialect.h"
#include "mlir/Pass/Pass.h"

namespace SpecHLS {
#define GEN_PASS_DEF_SPECHLSTOCOMB
#include "Conversion/Passes.h.inc"
}
using namespace circt;
using namespace hw;
using namespace comb;
using namespace mlir;
using namespace SpecHLS;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a comb::GammaOp operation to a comb::GammaOp
struct GammaToMuxOpConversion : OpConversionPattern<GammaOp> {
  using OpConversionPattern<GammaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GammaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type inputType = op.getSelect().getType();
    if (op.getInputs().size()==2) {
        if (inputType.isa<IntegerType>() &&
          inputType.getIntOrFloatBitWidth() == 1) {
          rewriter.replaceOpWithNewOp<MuxOp>(op,  op.getSelect(),  *op.getInputs().begin(),*op.getInputs().end());
          return success();
        }
    }

//
//    if (inputType.isa<IntegerType>() &&
//        inputType.getIntOrFloatBitWidth() == 1) {
//      Type outType = rewriter.getIntegerType(op.getMultiple());
//      rewriter.replaceOpWithNewOp<ExtSIOp>(op, outType, adaptor.getInput());
//      return success();
//    }
//
//    SmallVector<Value> inputs(op.getMultiple(), adaptor.getInput());
//    rewriter.replaceOpWithNewOp<GammaOp>(op, inputs);
//    return success();
  }
};




/// Lower a comb::GammaOp operation to the arith dialect
struct GammaOpConversion : OpConversionPattern<GammaOp> {
  using OpConversionPattern<GammaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GammaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
//    Type type = op.getResult().getType();
//    Location loc = op.getLoc();
//    unsigned nextInsertion = type.getIntOrFloatBitWidth();
//
//    Value aggregate =
//        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(type, 0));
//
//    for (unsigned i = 0, e = op.getNumOperands(); i < e; i++) {
//      nextInsertion -=
//          adaptor.getOperands()[i].getType().getIntOrFloatBitWidth();
//
//      Value nextInsValue = rewriter.create<arith::ConstantOp>(
//          loc, IntegerAttr::get(type, nextInsertion));
//      Value extended =
//          rewriter.create<ExtUIOp>(loc, type, adaptor.getOperands()[i]);
//      Value shifted = rewriter.create<ShLIOp>(loc, extended, nextInsValue);
//      aggregate = rewriter.create<OrIOp>(loc, aggregate, shifted);
//    }
//
//    rewriter.replaceOp(op, aggregate);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to Arith pass
//===----------------------------------------------------------------------===//

namespace {
// CRTP pattern
struct ConvertSpecHLSToCombPass : public ConvertSpecHLSToCombPassBase<OperationPass<ModuleOp>> {

  void runOnOperation() override;
  StringRef getName() const override {
    return "ConvertSpecHLSToCombPass";
  }
};
} // namespace

void populateSpecHLSToCombConversionPatterns(
    TypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<GammaToMuxOpConversion,GammaOpConversion>(
      converter, patterns.getContext());
}

void ConvertSpecHLSToCombPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalOp<SpecHLS::GammaOp>();
  target.addLegalDialect<SpecHLSDialect>();
  //target.addLegalDialect<ArithDialect>();
  // Arith does not have an operation equivalent to comb.parity. A lowering
  // would result in undesirably complex logic, therefore, we mark it legal
  // here.
  //target.addLegalOp<comb::ParityOp>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });
  // TODO: a pattern for comb.parity
  populateSpecHLSToCombConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertSpecHLSToCombPass() {
  return std::make_unique<ConvertSpecHLSToCombPass>();
}

namespace SpecHLS {
void registerConvertSpecHLSToCombPass() {
  PassRegistration<ConvertSpecHLSToCombPass>();
}
} // namespace mlir
