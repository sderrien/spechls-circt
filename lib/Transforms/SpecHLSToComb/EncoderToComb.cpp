//
// Created by Steven on 30/12/2023.
//

#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace SpecHLS {
#define GEN_PASS_DEF_SPECHLSTOCOMB
#include "Conversion/Passes.h.inc"

} // namespace SpecHLS
using namespace circt;
using namespace hw;
using namespace comb;
using namespace mlir;
using namespace SpecHLS;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace SpecHLS {}
namespace {
/// Lower a comb::EncoderOp operation to a comb::EncoderOp
struct EncoderToCombOpConversion : OpConversionPattern<EncoderOp> {
  using OpConversionPattern<EncoderOp>::OpConversionPattern;

  bool match(const Value seed, const Value candidate) const {
    if (candidate.getDefiningOp() == seed.getDefiningOp()) {
      if (candidate.getType() == candidate.getType()) {
        return true;
      }
    }
    return false;
  }

  LogicalResult
  matchAndRewrite(EncoderOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    uint32_t uiwidth = op.getData().getType().getWidth();

    APInt width = APInt(32,uiwidth);
    int bw = width.exactLogBase2();

    auto concatInput = rewriter.create<ConcatOp>(op.getLoc(), op->getOperands());

    llvm::outs() << "width " << bw << "\n";
    if (uiwidth<=8) {
      SmallVector<Attribute> table;
      for (auto k= 0;k<(1<<uiwidth);k++) {
        auto val = APInt(bw,k, false);
        auto ldz = val.countLeadingZeros();
        table.push_back(rewriter.getI32IntegerAttr(ldz));
      }
      auto content = rewriter.getArrayAttr(ArrayRef(table));
      auto selBit = rewriter.create<LookUpTableOp>(op.getLoc(), op.getResult().getType(),concatInput->getResults()[0], content);
    }

    auto extractBit = rewriter.create<ExtractOp>(op.getLoc(), op.getData(), 0,width.getZExtValue()/2);


    //    Value *muxOperands = new Value[cwidth * (cwidth + 1)];
//    // S//mallVector<Value,SPEC_EncoderOp_MAXOPERANDS> muxOperands;
//
//    for (int k = 0; k < cwidth; k += 1) {
//      if (k < width) {
//        muxOperands[k] = op.getInputs()[k];
//      } else {
//        muxOperands[k] = op.getInputs()[width - 1];
//      }
//      // llvm::errs() << "\t-mux[" << k << "," << 0 << "] = "<< muxOperands[k]
//      // <<"\n" ;
//    }
//    while (cwidth > 1) {
//      // llvm::errs()  << "depth = " << depth << " -> extract "<< selBit <<"\n";
//      for (int k = 0; k < cwidth; k += 2) {
//        // llvm::errs()  << "  k=  " << k <<"\n";
//        int offset = (width * depth);
//        Value opa = muxOperands[k + offset];
//        Value opb = NULL;
//        // llvm::errs() << "opa = mux[" << k << "," << depth << "] = "<< opa
//        // <<"\n" ;
//
//        if (k + 1 < cwidth) {
//          opb = muxOperands[k + 1 + offset];
//          // llvm::errs() << "opb = mux[" << k+1 << "," << depth << "] = "<< opb
//          // <<"\n" ;
//        } else {
//          opb = muxOperands[k + offset];
//          // llvm::errs() << "opb = mux[" << k << "," << depth << "]= "<< opb
//          // <<"\n" ; ;
//        }
//        if (opa || opb) {
//          // llvm::errs() << " null";
//        }
//        auto mux = rewriter.create<MuxOp>(op.getLoc(), selBit, opa, opb);
//        // llvm::errs() << "mux[" << k/2 << "," << depth+1 << "] = " ;
//        // llvm::errs() << mux;
//        muxOperands[k / 2 + offset + width] = mux;
//      }
//      cwidth = cwidth / 2;
//      depth++;
//    }
//    // llvm::errs() << "result is " <<  muxOperands[width*depth] << "\n" ;
//    rewriter.replaceOp(op, muxOperands[width * depth]);
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
struct ConvertSpecHLSToCombPass
    : public SpecHLS::impl::SpecHLSToCombBase<ConvertSpecHLSToCombPass> {
  void runOnOperation() override;
  //  virtual StringRef getName() ;
  //  virtual std::unique_ptr<Pass> clonePass() ;
};
} // namespace

void populateSpecHLSToCombConversionPatterns(
    TypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<EncoderToCombOpConversion /*,EncoderOpConversion*/>(
      converter, patterns.getContext());
}

void ConvertSpecHLSToCombPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<SpecHLSDialect>();
  target.addLegalDialect<CombDialect>();
  target.addIllegalOp<SpecHLS::EncoderOp>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;

  converter.addConversion([](Type type) { return type; });

  populateSpecHLSToCombConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    signalPassFailure();
}

namespace SpecHLS {

std::unique_ptr<OperationPass<ModuleOp>> createConvertSpecHLSToCombPass() {
  return std::make_unique<ConvertSpecHLSToCombPass>();
}

void registerConvertSpecHLSToCombPass() {
  PassRegistration<ConvertSpecHLSToCombPass>();
}
} // namespace SpecHLS
