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

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

namespace SpecHLS {

struct LookUpMergingPattern : OpRewritePattern<LookUpTableOp> {

  using OpRewritePattern<LookUpTableOp>::OpRewritePattern;

  ArrayAttr updateLUTContent(LookUpTableOp op, ArrayAttr inner, ArrayAttr outer,
                             PatternRewriter &rewriter) const {
    SmallVector<int, 1024> newcontent;
    int innerSize = inner.size();
    int outerSize = outer.size();
    for (int o = 0; o < innerSize; o++) {

      auto innerValue = cast<IntegerAttr>(inner.getValue()[o]).getInt();

      if (innerValue >= outer.size()) {
        emitError(
            op->getLoc(),
            "Inconsistent indexing in nested LookUpTables (forcing to zero)");
        newcontent.push_back(0);
      } else {
        auto outerValue =
            cast<IntegerAttr>(outer.getValue()[innerValue]).getInt();
        newcontent.push_back(outerValue);
      }
    }
    return rewriter.getI32ArrayAttr(newcontent);
  }

  LogicalResult matchAndRewrite(LookUpTableOp op,
                                PatternRewriter &rewriter) const override {

    auto input = op.getInput().getDefiningOp();
    if (input) {
      auto inputLUT = dyn_cast<SpecHLS::LookUpTableOp>(input);

      if (inputLUT) {
        llvm::errs() << "Merging " << op << " and " << inputLUT << "\n";

        ArrayAttr newAttr = updateLUTContent(op, inputLUT.getContent(),
                                             op.getContent(), rewriter);
        auto lutSelect = rewriter.replaceOpWithNewOp<LookUpTableOp>(
            op, op->getResult(0).getType(), inputLUT.getInput(), newAttr);

        rewriter.eraseOp(inputLUT);

        return success();
      }
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
    mlir::verify(getOperation(), true);
  }
};

std::unique_ptr<OperationPass<>> createMergeLookUpTablesPass() {
  return std::make_unique<MergeLookUpTablesPass>();
}
} // namespace SpecHLS