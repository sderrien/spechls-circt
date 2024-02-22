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

  LogicalResult matchAndRewrite(EncoderOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

    uint32_t uiwidth = op.getData().getType().getWidth();
    auto loc = op.getLoc();
    APInt width = APInt(32, uiwidth);
    int bw = width.exactLogBase2();
    auto resType= op.getResult().getType();

    auto concatInput = rewriter.create<ConcatOp>(loc, op->getOperands());

    if (uiwidth <= 8) {
      SmallVector<Attribute> table;
      for (auto k = 0; k < (1 << uiwidth); k++) {
        auto val = APInt(bw, k, false);
        auto ldz = val.countLeadingZeros();
        table.push_back(rewriter.getI32IntegerAttr(ldz));
      }
      auto content = rewriter.getArrayAttr(ArrayRef(table));
      auto value = concatInput->getResults()[0];
      auto selBit = rewriter.create<LookUpTableOp>(loc, resType, value, content);
    }

    auto extractBit = rewriter.create<ExtractOp>(loc, op.getData(), 0, width.getZExtValue() / 2);

    return success();
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
