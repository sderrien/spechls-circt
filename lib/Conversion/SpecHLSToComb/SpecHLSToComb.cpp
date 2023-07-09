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

    llvm::outs() << "hello world";
    Type inputType = op.getSelect().getType();
    if (op.getInputs().size()==2) {
      llvm::outs() << "two data inputs";
        if (inputType.isa<IntegerType>()) {
          llvm::outs() << "select is integer";
          if (inputType.getIntOrFloatBitWidth() == 1) {
            llvm::outs() << "select is bool";
            rewriter.replaceOpWithNewOp<MuxOp>(op, op.getSelect(),
                                               *op.getInputs().begin(),
                                               *op.getInputs().end());
            return success();
          }
        }
    }
    return success();

  }
};




///// Lower a comb::GammaOp operation to the arith dialect
//struct GammaOpConversion : OpConversionPattern<GammaOp> {
//  using OpConversionPattern<GammaOp>::OpConversionPattern;

//  LogicalResult
//  matchAndRewrite(GammaOp op, OpAdaptor adaptor,
//                  ConversionPatternRewriter &rewriter) const override {
////    Type type = op.getResult().getType();
////    Location loc = op.getLoc();
////    unsigned nextInsertion = type.getIntOrFloatBitWidth();
////
////    Value aggregate =
////        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(type, 0));
////
////    for (unsigned i = 0, e = op.getNumOperands(); i < e; i++) {
////      nextInsertion -=
////          adaptor.getOperands()[i].getType().getIntOrFloatBitWidth();
////
////      Value nextInsValue = rewriter.create<arith::ConstantOp>(
////          loc, IntegerAttr::get(type, nextInsertion));
////      Value extended =
////          rewriter.create<ExtUIOp>(loc, type, adaptor.getOperands()[i]);
////      Value shifted = rewriter.create<ShLIOp>(loc, extended, nextInsValue);
////      aggregate = rewriter.create<OrIOp>(loc, aggregate, shifted);
////    }
////
////    rewriter.replaceOp(op, aggregate);
//    return success();
//  }
//};
} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to Arith pass
//===----------------------------------------------------------------------===//


namespace {


// CRTP pattern
struct ConvertSpecHLSToCombPass : public SpecHLS::impl::SpecHLSToCombBase<ConvertSpecHLSToCombPass> {
  void runOnOperation() override;
//  virtual StringRef getName() ;
//  virtual std::unique_ptr<Pass> clonePass() ;

};
} // namespace

void populateSpecHLSToCombConversionPatterns(
    TypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<GammaToMuxOpConversion/*,GammaOpConversion*/>(
      converter, patterns.getContext());
}

void ConvertSpecHLSToCombPass::runOnOperation() {
  ConversionTarget target(getContext());


  //target.addIllegalOp<SpecHLS::GammaOp>();
  target.addLegalDialect<SpecHLSDialect>();
  target.addLegalDialect<CombDialect>();

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
