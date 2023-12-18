//===- SpecHLSToComb.cpp
//----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace SpecHLS {
#define GEN_PASS_DEF_SPECHLSTOARC
#include "Conversion/Passes.h.inc"
} // namespace SpecHLS
using namespace circt;
using namespace hw;
using namespace comb;
using namespace mlir;
using namespace arc;
using namespace SpecHLS;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a comb::GammaOp operation to a comb::GammaOp
struct GammaToMuxOpConversion : OpConversionPattern<GammaOp> {
  using OpConversionPattern<GammaOp>::OpConversionPattern;

  bool match(const Value seed, const Value candidate) const {
    if (candidate.getDefiningOp() == seed.getDefiningOp()) {
      if (candidate.getType() == candidate.getType()) {
        return true;
      }
    }
    return false;
  }

  unsigned long upper_power_of_two(const unsigned long _v) const {
    unsigned long v = _v;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
  }

  LogicalResult
  matchAndRewrite(GammaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::outs() << "==== Lowering " << op << "\n";

    int width = op.getInputs().size();
    int depth = 0;
    int cwidth = upper_power_of_two(width);
    Value *muxOperands = new Value[cwidth * (cwidth + 1)];
    // S//mallVector<Value,SPEC_GAMMAOP_MAXOPERANDS> muxOperands;

    for (int k = 0; k < cwidth; k += 1) {
      if (k < width) {
        muxOperands[k] = op.getInputs()[k];
      } else {
        muxOperands[k] = op.getInputs()[width - 1];
      }
      llvm::outs() << "\t-mux[" << k << "," << 0 << "] = " << muxOperands[k]
                   << "\n";
    }
    while (cwidth > 1) {
      auto selBit =
          rewriter.create<ExtractOp>(op.getLoc(), op.getSelect(), depth, 1);
      llvm::outs() << "depth = " << depth << " -> extract " << selBit << "\n";
      for (int k = 0; k < cwidth; k += 2) {
        llvm::outs() << "  k=  " << k << "\n";
        int offset = (width * depth);
        Value opa = muxOperands[k + offset];
        Value opb = NULL;
        llvm::outs() << "opa = mux[" << k << "," << depth << "] = " << opa
                     << "\n";

        if (k + 1 < cwidth) {
          opb = muxOperands[k + 1 + offset];
          llvm::outs() << "opb = mux[" << k + 1 << "," << depth << "] = " << opb
                       << "\n";
        } else {
          opb = muxOperands[k + offset];
          llvm::outs() << "opb = mux[" << k << "," << depth << "]= " << opb
                       << "\n";
          ;
        }
        if (opa || opb) {
          llvm::outs() << " null";
        }
        auto mux = rewriter.create<MuxOp>(op.getLoc(), selBit, opa, opb);
        llvm::outs() << "mux[" << k / 2 << "," << depth + 1 << "] = ";
        llvm::outs() << mux;
        muxOperands[k / 2 + offset + width] = mux;
      }
      cwidth = cwidth / 2;
      depth++;
    }
    llvm::outs() << "result is " << muxOperands[width * depth] << "\n";
    rewriter.replaceOp(op, muxOperands[width * depth]);
    delete muxOperands;
    return success();

    return failure();
  }
};

/// Lower a comb::GammaOp operation to the arith dialect
struct MuToRegOpConversion : OpConversionPattern<MuOp> {
  using OpConversionPattern<MuOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MuOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    //  auto selBit = rewriter.create<RegOp>(op.getLoc(), op.getSelect(),depth,
    //  1);

    //    Type type = op.getResult().getType();
    //    Location loc = op.getLoc();
    //    unsigned nextInsertion = type.getIntOrFloatBitWidth();
    //
    //    Value aggregate =
    //        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(type,
    //        0));
    //
    //    for (unsigned i = 0, e = op.getNumOperands(); i < e; i++) {
    //      nextInsertion -=
    //          adaptor.getOperands()[i].getType().getIntOrFloatBitWidth();
    //
    //      Value nextInsValue = rewriter.create<arith::ConstantOp>(
    //          loc, IntegerAttr::get(type, nextInsertion));
    //      Value extended =
    //          rewriter.create<ExtUIOp>(loc, type, adaptor.getOperands()[i]);
    //      Value shifted = rewriter.create<ShLIOp>(loc, extended,
    //      nextInsValue); aggregate = rewriter.create<OrIOp>(loc, aggregate,
    //      shifted);
    //    }
    //
    //    rewriter.replaceOp(op, aggregate);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert Arc to Arith pass
//===----------------------------------------------------------------------===//

namespace {

// CRTP pattern
struct ConvertSpecHLSToArcPass
    : public SpecHLS::impl::SpecHLSToArcBase<ConvertSpecHLSToArcPass> {
  void runOnOperation() override;
  //  virtual StringRef getName() ;
  //  virtual std::unique_ptr<Pass> clonePass() ;
};
} // namespace

void populateSpecHLSToArcConversionPatterns(TypeConverter &converter,
                                            mlir::RewritePatternSet &patterns) {
  patterns.add<GammaToMuxOpConversion /*,GammaOpConversion*/>(
      converter, patterns.getContext());
  patterns.add<MuToRegOpConversion /*,GammaOpConversion*/>(
      converter, patterns.getContext());
}

void ConvertSpecHLSToArcPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<SpecHLSDialect>();
  target.addIllegalOp<SpecHLS::GammaOp>();
  target.addLegalDialect<ArcDialect>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });
  // TODO: a pattern for comb.parity
  populateSpecHLSToArcConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertSpecHLSToArcPass() {
  return std::make_unique<ConvertSpecHLSToArcPass>();
}

namespace SpecHLS {
void registerConvertSpecHLSToArcPass() {
  PassRegistration<ConvertSpecHLSToArcPass>();
}
} // namespace SpecHLS
