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


struct LookUpTableToTruthTableOpConversion : OpRewritePattern<LookUpTableOp> {
  using OpRewritePattern<LookUpTableOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LookUpTableOp op,  PatternRewriter &rewriter) const override;
};

struct GammaToMuxOpConversion : OpRewritePattern<GammaOp> {
  using OpRewritePattern<GammaOp>::OpRewritePattern;
   LogicalResult matchAndRewrite(GammaOp op, PatternRewriter &rewriter) const override;
};

#endif // SPECHLS_DIALECT_SPECHLSCONVERSION_H
