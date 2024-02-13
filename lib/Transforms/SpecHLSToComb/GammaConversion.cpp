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
GammaToMuxOpConversion::matchAndRewrite(GammaOp op, PatternRewriter &rewriter) const  {

  // llvm::errs() << "==== Lowering "<< op << "\n";

  llvm::outs() << op <<"\n" ;
  auto addrBW = op.getSelect().getType().getIntOrFloatBitWidth();
  auto nbInputs = op.getInputs().size();
  int depth = 0;
  int nbCase = 1 << addrBW;
  auto loc = op.getLoc();
  auto inputs = op.getInputs();
  auto type = op->getResult(0).getType();
  auto flatSym = op.getNameAttr();
  if (nbInputs>2) {
    auto selMSBBit = rewriter.create<ExtractOp>(loc, op.getSelect(), addrBW-1, 1);
    auto selLSBBits = rewriter.create<ExtractOp>(loc, op.getSelect(), 0 , addrBW-1);
    auto gammaLeft = rewriter.create<GammaOp>(loc, type, flatSym, selLSBBits, inputs.slice(0,(nbCase/2)));
    auto gammaRight = rewriter.create<GammaOp>(loc, type, flatSym, selLSBBits, inputs.slice(nbCase/2,(nbInputs-nbCase/2)));
    auto mux = rewriter.create<MuxOp>(loc, type, selMSBBit, gammaLeft, gammaRight);
    llvm::outs() << mux << "\n";
    rewriter.replaceOp(op, mux);
    return success();
   // auto gammaRight = rewriter.create<GammaOp>(op.getLoc(), selBit, opa, opb);
  } else if (nbInputs==2) {
    auto mux = rewriter.create<MuxOp>(op.getLoc(), op.getType(), op.getSelect(), op.getInputs()[0], op.getInputs()[1]);
    llvm::outs() << mux << "\n";
    rewriter.replaceOp(op, mux);
    return success();
  } else {
    return failure();
  }
//  auto mux = rewriter.create<MuxOp>(op.getLoc(), selBit, opa, opb);
//  Value *muxOperands = new Value[100+nbCase * (nbCase + 1)];
//  // S//mallVector<Value,SPEC_GAMMAOP_MAXOPERANDS> muxOperands;
//
//
//  llvm::outs() << "width=" << addrBW << ", nbCase=" << nbCase  <<"\n" ;
//
//  for (int k = 0; k < nbCase; k += 1) {
//    llvm::outs() << "k=" << k << "\n" ;
//    if (k < nbInputs) {
//      muxOperands[k] = op.getInputs()[k];
//    } else {
//      muxOperands[k] = op.getInputs()[nbInputs-1];
//    }
//    llvm::outs() << "\t-mux[" << k << "," << 0 << "] = "<< muxOperands[k] <<"\n" ;
//  }
//  llvm::outs() << "next"<< "\n" ;
//  Operation *lastMux = NULL;
//  while (nbCase > 1) {
//     llvm::outs()  << "depth = " << depth  << ", nbcase = " << nbCase << " -> extract "<< selBit <<"\n";
//
//    for (int k = 0; k < nbCase; k += 2) {
//      llvm::outs()  << " k=" << k <<", nbCase="<< nbCase << "\n";
//      auto offset = nbCase * depth;
//      Value opa = muxOperands[k + offset];
//      Value opb = NULL;
//       llvm::outs() << "  - opa = mux[" << k << "," << depth << "] = "<< opa <<"\n" ;
//
//      if (k + 1 < nbCase) {
//        opb = muxOperands[k + 1 + offset];
//        llvm::outs() << " - opb = mux[" << k+1 << "," << depth << "] = "<< opb  <<"\n" ;
//      } else {
//        opb = muxOperands[k + offset];
//        llvm::outs() << " - opb = mux[" << k << "," << depth << "]= "<< opb <<"\n" ;
//      }
//      if (opa==NULL || opb==NULL) {
//        llvm::outs() << " error opb==null for " <<k <<"\n";
//        return failure();
//      }
//      auto mux = rewriter.create<MuxOp>(op.getLoc(), selBit, opa, opb);
//       llvm::outs() << "mux[" << k/2 << "," << depth+1 << "] = " ;
//       llvm::outs() << mux;
//      muxOperands[k / 2 + offset + nbCase] = mux;
//      lastMux = mux;
//    }
//    nbCase = nbCase / 2;
//    depth++;
//  }
//   llvm::outs() << "result is " <<  *lastMux << "\n" ;

}
