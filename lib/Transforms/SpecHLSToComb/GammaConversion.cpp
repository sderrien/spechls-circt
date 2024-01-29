//
// Created by Steven on 19/01/2024.
//
#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Transforms/SpecHLSConversion.h"

LogicalResult
GammaToMuxOpConversion::matchAndRewrite(GammaOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) {

  // llvm::errs() << "==== Lowering "<< op << "\n";

  auto width = APInt(op.getInputs().size(), 32);
  int depth = 0;
  int cwidth = 1 << width.ceilLogBase2();
  Value *muxOperands = new Value[cwidth * (cwidth + 1)];
  // S//mallVector<Value,SPEC_GAMMAOP_MAXOPERANDS> muxOperands;

  for (int k = 0; k < cwidth; k += 1) {
    if (k < width.getZExtValue()) {
      muxOperands[k] = op.getInputs()[k];
    } else {
      muxOperands[k] = op.getInputs()[width.getZExtValue() - 1];
    }
    // llvm::errs() << "\t-mux[" << k << "," << 0 << "] = "<< muxOperands[k]
    // <<"\n" ;
  }
  while (cwidth > 1) {
    auto selBit =
        rewriter.create<ExtractOp>(op.getLoc(), op.getSelect(), depth, 1);
    // llvm::errs()  << "depth = " << depth << " -> extract "<< selBit <<"\n";
    for (int k = 0; k < cwidth; k += 2) {
      // llvm::errs()  << "  k=  " << k <<"\n";
      auto offset = (width * depth).getZExtValue();
      Value opa = muxOperands[k + offset];
      Value opb = NULL;
      // llvm::errs() << "opa = mux[" << k << "," << depth << "] = "<< opa
      // <<"\n" ;

      if (k + 1 < cwidth) {
        opb = muxOperands[k + 1 + offset];
        // llvm::errs() << "opb = mux[" << k+1 << "," << depth << "] = "<< opb
        // <<"\n" ;
      } else {
        opb = muxOperands[k + offset];
        // llvm::errs() << "opb = mux[" << k << "," << depth << "]= "<< opb
        // <<"\n" ; ;
      }
      if (opa || opb) {
        // llvm::errs() << " null";
      }
      auto mux = rewriter.create<MuxOp>(op.getLoc(), selBit, opa, opb);
      // llvm::errs() << "mux[" << k/2 << "," << depth+1 << "] = " ;
      // llvm::errs() << mux;
      muxOperands[k / 2 + offset + width.getZExtValue()] = mux;
    }
    cwidth = cwidth / 2;
    depth++;
  }
  // llvm::errs() << "result is " <<  muxOperands[width*depth] << "\n" ;
  rewriter.replaceOp(op, muxOperands[width.getZExtValue() * depth]);
  delete muxOperands;
  return success();
}
