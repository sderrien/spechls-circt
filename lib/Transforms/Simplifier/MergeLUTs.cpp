//===- MergeLookUpTables.cpp - Arith-to-comb mapping pass ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the MergeLookUpTables pass.
//
//===----------------------------------------------------------------------===//

#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

namespace SpecHLS {

struct LookUpMergingPattern : OpRewritePattern<LookUpTableOp> {

  using OpRewritePattern<LookUpTableOp>::OpRewritePattern;

  ArrayAttr updateLUTContent(ArrayAttr inner, ArrayAttr outer,
                             PatternRewriter &rewriter) const {
    SmallVector<int, 1024> newcontent;
    int innerSize = inner.size();
    int outerSize = outer.size();
    for (int o = 0; o < innerSize; o++) {

      if (o > inner.size()) {
        llvm::errs() << "out of bound access at " << o << " for " << inner
                     << "  \n";
        return NULL;
      }
      auto innerValue = cast<IntegerAttr>(inner.getValue()[o]).getInt();
      if (innerValue > outer.size()) {
        llvm::errs() << "out of bound access at " << innerValue << " for "
                     << outer << "  \n";
        return NULL;
      }

      auto outerValue =
          cast<IntegerAttr>(outer.getValue()[innerValue]).getInt();
      newcontent.push_back(outerValue);
    }
    return rewriter.getI32ArrayAttr(newcontent);
  }

  LogicalResult matchAndRewrite(LookUpTableOp op,
                                PatternRewriter &rewriter) const override {

    //    llvm::errs() << "Analyzing  " << op << " \n";
    auto input = op.getInput().getDefiningOp();

    if (input != NULL && llvm::isa<SpecHLS::LookUpTableOp>(input)) {
      auto inputLUT = cast<SpecHLS::LookUpTableOp>(input);

      ArrayAttr newAttr =
          updateLUTContent(inputLUT.getContent(), op.getContent(), rewriter);
      auto lutSelect = rewriter.replaceOpWithNewOp<LookUpTableOp>(
          op, op->getResultTypes(), inputLUT.getInput(), newAttr);

      rewriter.eraseOp(inputLUT);

      return success();
    }

    return failure();
  }
};

struct MergeLookUpTablesPass
    : public impl::MergeLookUpTablesPassBase<MergeLookUpTablesPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<LookUpMergingPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      llvm::errs() << "partial conversion failed pattern  \n";
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<>> createMergeLookUpTablesPass() {
  return std::make_unique<MergeLookUpTablesPass>();
}
} // namespace SpecHLS
