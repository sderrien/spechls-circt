//
// Created by Steven on 19/01/2024.
//

#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "Transforms/SpecHLSConversion.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


LogicalResult LookUpTableToTruthTableOpConversion::matchAndRewrite(LookUpTableOp op, PatternRewriter &rewriter) const {

  llvm :: outs() << "LookUpTableToTruthTableOpConversion for " << op <<"\n";

  auto content = op.getContent();
#ifdef USE_TRUTH_TABLE
  SmallVector<Attribute> newContent[op.getType().getWidth()];
  for (unsigned int i = 0; i < content.size(); i++) {
    auto innerValue = cast<IntegerAttr>(content[i]).getInt();
    rewriter.create<circt::hw::ConstantOp>(op.getLoc(),op.getType(),innerValue);
    for (unsigned int k = 0; k < op.getType().getWidth(); k++) {
      newContent[k].push_back(rewriter.getBoolAttr((innerValue >> k) & 0x1));
    }
  }
  Operation *res = NULL;

  SmallVector<Value> bits;
  for (uint32_t  i=0;i<op->getOperand(0).getType().getIntOrFloatBitWidth();i++) {
    auto extract = rewriter.create<circt::comb::ExtractOp>(op.getLoc(), op->getOperand(0),i,1);
    bits.push_back(extract.getResult());
  }
  for (unsigned int k = 0; k < op.getType().getWidth(); k++) {
      auto ttable = rewriter.create<circt::comb::TruthTableOp>(op.getLoc(), bits,rewriter.getArrayAttr(newContent[k]));
      if (res==NULL) {
        res = ttable;
      } else {
        res = rewriter.create<ConcatOp>(op.getLoc(), res->getResult(0),
                                        ttable.getResult());
      }
      llvm :: outs() << "TT["<<k <<"]  " << res <<"\n";

  }
  auto parentOp = op->getParentOp();
  auto value = op->getResult(0);
  //llvm :: outs() << *parentOp << "\n";
  // rewriter.replaceOp(op, res);
  llvm :: outs() << *parentOp << "\n";
  value.replaceAllUsesWith(res->getResult(0));
  rewriter.eraseOp(op);
  llvm :: outs() << *parentOp << "\n";
#else
  SmallVector<Value> constants;
  for (unsigned int i = 0; i < content.size(); i++) {
    auto innerValue = cast<IntegerAttr>(content[i]).getInt();
    auto constOp = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),op.getType(),innerValue);
    constants.push_back(constOp->getResult(0));
  }

  GammaOp res = rewriter.create<SpecHLS::GammaOp>(op.getLoc(), op->getResult(0).getType(),"LUT", op->getOperand(0),constants);

  auto value = op->getResult(0);
  value.replaceAllUsesWith(res->getResult(0));
  rewriter.replaceOp(op, res);

#endif

  return success();
}
