//===- SpecHLSToComb.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
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

namespace SpecHLS {
  unsigned long upper_power_of_two(const unsigned long _v) {
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
}

namespace {
/// Lower a comb::GammaOp operation to a comb::GammaOp
struct GammaToMuxOpConversion : OpConversionPattern<GammaOp> {
  using OpConversionPattern<GammaOp>::OpConversionPattern;

  bool match(const Value seed,const  Value candidate) const {
    if (candidate.getDefiningOp()==seed.getDefiningOp()) {
      if (candidate.getType()==candidate.getType()) {
        return true;
      }
    }
    return false;
  }


  LogicalResult
  matchAndRewrite(GammaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {


    //llvm::errs() << "==== Lowering "<< op << "\n";

    int width = op.getInputs().size();
    int depth = 0;
    int cwidth = upper_power_of_two(width);
    Value* muxOperands = new Value[cwidth * (cwidth+1)];
    //S//mallVector<Value,SPEC_GAMMAOP_MAXOPERANDS> muxOperands;

    for (int k=0;k<cwidth;k+=1) {
      if (k<width) {
        muxOperands[k] = op.getInputs()[k];
      } else {
        muxOperands[k] = op.getInputs()[width-1];
      }
      //llvm::errs() << "\t-mux[" << k << "," << 0 << "] = "<< muxOperands[k] <<"\n" ;
    }
    while (cwidth>1) {
      auto selBit = rewriter.create<ExtractOp>(op.getLoc(), op.getSelect(),depth, 1);
      //llvm::errs()  << "depth = " << depth << " -> extract "<< selBit <<"\n";
      for (int k=0;k<cwidth;k+=2) {
        //llvm::errs()  << "  k=  " << k <<"\n";
        int offset = (width*depth);
        Value opa = muxOperands[k+offset];
        Value opb = NULL;
        //llvm::errs() << "opa = mux[" << k << "," << depth << "] = "<< opa <<"\n" ;

        if (k+1<cwidth) {
          opb =muxOperands[k+1+offset];
          //llvm::errs() << "opb = mux[" << k+1 << "," << depth << "] = "<< opb <<"\n" ;
        } else {
          opb =muxOperands[k+offset];
          //llvm::errs() << "opb = mux[" << k << "," << depth << "]= "<< opb <<"\n" ; ;
        }
        if (opa || opb) {
          //llvm::errs() << " null";
        }
        auto mux = rewriter.create<MuxOp>(op.getLoc(), selBit, opa,opb);
        //llvm::errs() << "mux[" << k/2 << "," << depth+1 << "] = " ;
        //llvm::errs() << mux;
        muxOperands[k/2+offset+width] =mux;
      }
      cwidth = cwidth/2;
      depth++;
    }
    //llvm::errs() << "result is " <<  muxOperands[width*depth] << "\n" ;
      rewriter.replaceOp(op, muxOperands[width*depth]);
      delete muxOperands;
      return success();

    return failure();

  }
};


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


  target.addLegalDialect<SpecHLSDialect>();
  target.addLegalDialect<CombDialect>();
  target.addIllegalOp<SpecHLS::GammaOp>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;

  converter.addConversion([](Type type) { return type; });

  populateSpecHLSToCombConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}


namespace SpecHLS {

std::unique_ptr<OperationPass<ModuleOp>> createConvertSpecHLSToCombPass() {
  return std::make_unique<ConvertSpecHLSToCombPass>();
}

void registerConvertSpecHLSToCombPass() {
  PassRegistration<ConvertSpecHLSToCombPass>();

}
} // namespace mlir
