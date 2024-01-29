//
// Created by Steven on 19/01/2024.
//

#ifndef SPECHLS_DIALECT_SPECHLSCONVERSION_H
#define SPECHLS_DIALECT_SPECHLSCONVERSION_H

#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

using namespace circt;
using namespace hw;
using namespace comb;
using namespace mlir;
using namespace SpecHLS;


struct LookUpTableToTruthTableOpConversion : OpConversionPattern<LookUpTableOp> {
  using OpConversionPattern<LookUpTableOp>::OpConversionPattern;
  bool match(const Value seed, const Value candidate) ;
  LogicalResult matchAndRewrite(LookUpTableOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter);
};

struct GammaToMuxOpConversion : OpConversionPattern<GammaOp> {
  using OpConversionPattern<GammaOp>::OpConversionPattern;
  bool match(const Value seed, const Value candidate) ;
   LogicalResult matchAndRewrite(GammaOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter);
};

#endif // SPECHLS_DIALECT_SPECHLSCONVERSION_H
