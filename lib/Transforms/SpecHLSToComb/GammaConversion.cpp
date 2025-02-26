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

#define VERBOSE false

unsigned bitwidth(unsigned  int x) {
  unsigned  int power = 1;
  unsigned  int offset = 1;
  while(power < x) {
    power*=2;
    offset++;
  }
  return offset;
}
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


    auto rightNbElts = (nbInputs - nbCase / 2);
    auto bw= bitwidth(rightNbElts);
    if (VERBOSE) llvm::errs() <<  op << "[" << nbInputs << ";"<< nbCase <<"] ";
    if (VERBOSE) llvm::errs() <<  "[" << rightNbElts << ";"<< addrBW <<"] - >{\n";

    auto castSelect = rewriter.create<CastOp>(loc, rewriter.getIntegerType(addrBW), op.getSelect());
    auto selMSBBit = rewriter.create<ExtractOp>(loc, castSelect, addrBW - 1, 1);
    auto selLeftLSBBits = rewriter.create<ExtractOp>(loc, castSelect, 0, addrBW - 1);

    if (VERBOSE) llvm::errs() << "\t-left sel[0:"<<(addrBW-1)<<"] -> {";
    for (size_t k=0;k<nbCase/2;k++) {
      if (VERBOSE) {
        if (k>0) llvm::errs() << ",";
        llvm::errs() << "in["<<k<<"]";
      }

    }
    if (VERBOSE) llvm::errs() << "}\n\t-right sel[0:"<<bw<<") -> {";
    for (size_t k=(nbCase/2);k<nbInputs;k++) {
      if (VERBOSE) {
        if (k > 0)
          llvm::errs() << ",";

        llvm::errs() << "in[" << k << "]";
      }
    }
    if (VERBOSE) llvm::errs() << "}\n";

    auto selRightLSBBits = rewriter.create<ExtractOp>(loc, castSelect, 0, bw-1);

    auto gammaLeft = rewriter.create<GammaOp>(loc, type, flatSym, selLeftLSBBits, inputs.slice(0, (nbCase / 2)));
    if (VERBOSE) llvm::errs() << "\t-created "<<gammaLeft<<"\n";

    auto gammaRight = rewriter.create<GammaOp>(
        loc, type, flatSym, selRightLSBBits,
        inputs.slice(nbCase / 2, rightNbElts));
    if (VERBOSE) llvm::errs() << "\t-created "<<gammaRight<<"\n";

    auto mux= rewriter.create<MuxOp>(loc, type, selMSBBit, gammaRight, gammaLeft);
    rewriter.replaceOp(op, mux);

    if (VERBOSE) llvm::errs() << "} \n";
    return success();
  } else if (nbInputs == 2) {
    if (VERBOSE) llvm::errs() << " Gamma to mux "<< op << "\n";
    // slice bit 0 in case addrBW>1
    auto muxSel = rewriter.create<ExtractOp>(loc, op.getSelect(), 0, 1);
    auto mux = rewriter.create<MuxOp>(op.getLoc(), op.getType(), muxSel,
                                      op.getInputs()[1], op.getInputs()[0]);
    rewriter.replaceOp(op, mux);
    return success();
  } else if (nbInputs == 1){
    op.getResult().replaceAllUsesWith(op.getInputs()[0]);
    rewriter.eraseOp(op);
    return success();
  }
}