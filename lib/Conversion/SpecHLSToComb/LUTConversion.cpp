//
// Created by Steven on 19/01/2024.
//

#include "Conversion/SpecHLSConversion.h"
#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

LogicalResult LookUpTableToTruthTableOpConversion::matchAndRewrite(LookUpTableOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) {

  auto content = op.getContent();

  SmallVector<Attribute> newContent[op.getType().getWidth()];
  for (int i = 0; i < content.size(); i++) {
    auto innerValue = cast<IntegerAttr>(content[i]).getInt();

    for (int k = 0; k < op.getType().getWidth(); k++) {
      newContent[k].push_back(rewriter.getBoolAttr((innerValue >> k) & 0x1));
    }
  }
  Operation *res = NULL;
  for (int k = 0; k < op.getType().getWidth(); k++) {
    for (int i = 0; i < content.size(); i++) {
      auto concat = rewriter.create<circt::comb::ExtractOp>(
          op.getLoc(), op->getOperand(0), k, 1);
      auto ttable = rewriter.create<circt::comb::TruthTableOp>(
          op.getLoc(), concat->getOperand(0));
      ttable.setLookupTableAttr(rewriter.getArrayAttr(newContent[k]));
      if (res) {
        res = ttable;
      } else {
        res = rewriter.create<ConcatOp>(op.getLoc(), res->getResult(0),
                                        ttable.getResult());
      }
    }
  }

  rewriter.replaceOp(op, res);
  return failure();
}
