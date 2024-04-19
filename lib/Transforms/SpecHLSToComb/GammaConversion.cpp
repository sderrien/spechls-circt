//
// Created by Steven on 19/01/2024.
//
#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Transforms/SpecHLSConversion.h"

LogicalResult
GammaToMuxOpConversion::matchAndRewrite(GammaOp op,
                                        PatternRewriter &rewriter) const {

  auto addrBW = op.getSelect().getType().getIntOrFloatBitWidth();
  auto nbInputs = op.getInputs().size();
  int nbCase = 1 << addrBW;
  auto loc = op.getLoc();
  auto inputs = op.getInputs();
  auto type = op->getResult(0).getType();
  auto flatSym = op.getNameAttr();

  if (nbInputs > 2) {

    auto selMSBBit =
        rewriter.create<ExtractOp>(loc, op.getSelect(), addrBW - 1, 1);
    auto selLSBBits =
        rewriter.create<ExtractOp>(loc, op.getSelect(), 0, addrBW - 1);
    auto gammaLeft = rewriter.create<GammaOp>(loc, type, flatSym, selLSBBits,
                                              inputs.slice(0, (nbCase / 2)));
    auto gammaRight = rewriter.create<GammaOp>(
        loc, type, flatSym, selLSBBits,
        inputs.slice(nbCase / 2, (nbInputs - nbCase / 2)));

    auto mux =
        rewriter.create<MuxOp>(loc, type, selMSBBit, gammaLeft, gammaRight);
    rewriter.replaceOp(op, mux);

    return success();
  } else if (nbInputs == 2) {
    auto mux = rewriter.create<MuxOp>(op.getLoc(), op.getType(), op.getSelect(),
                                      op.getInputs()[0], op.getInputs()[1]);
    rewriter.replaceOp(op, mux);
    return success();
  } else {
    return failure();
  }
}