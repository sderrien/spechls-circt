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
#include "llvm/Support/FormatVariadic.h" // For llvm::formatv
#include <iostream>                      // For std::cout
#include <numeric>

using namespace mlir;
using namespace circt;
using namespace SpecHLS;
using namespace comb;

namespace SpecHLS {

template <typename To>

[[nodiscard]] inline decltype(auto) dyn_cast_operand(int k, mlir::Operation *op) {
  assert(llvm::detail::isPresent(op) && "dyn_cast on a non-existent value");
  To castedOp =NULL;
  if (k>=0 && k<op->getNumOperands()) {
    if (auto defOp = op->getOperand(k).getDefiningOp()) {
      if (defOp) {
        return dyn_cast<To>(defOp);
      }
    }
  }
  return castedOp;
}


//struct LookUpCanonicalizePattern : OpRewritePattern<LookUpTableOp> {
//  using OpRewritePattern<LookUpTableOp>::OpRewritePattern;
//  LogicalResult matchAndRewrite(LookUpTableOp lut, PatternRewriter &rewriter) const override {
//    auto input = lut.getInput().getDefiningOp();
//
//    if (input) {
//      auto inputConcat = dyn_cast<comb::ConcatOp>(input);
//      if (inputConcat) {
//        auto numOperands = inputConcat.getNumOperands();
//        if (numOperands=2) {
//
//          if(dyn_cast_operand<SpecHLS::LookUpTableOp>(1,inputConcat)) {
//             llvm::errs() << "Should canonicalize \n"<< input << "\n and \n" << inputConcat << "\n";
//          }
//        } else if (numOperands>=2) {
//          llvm::errs() << "Should canonicalize \n"<< input << "\n and \n" << inputConcat << "\n";
//        }
//      }
//    }
//    return failure();
//  }
//};

struct LookUpMergingPattern : OpRewritePattern<LookUpTableOp> {

  bool verbose = false;

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

  LogicalResult matchAndRewrite(LookUpTableOp outerLUT,
                                PatternRewriter &rewriter) const override {

    auto input = outerLUT.getInput().getDefiningOp();
    auto outerLUTContent = outerLUT.getContent().getValue();

    if (input) {
      auto inputLUT = dyn_cast<SpecHLS::LookUpTableOp>(input);
      auto inputConcat = dyn_cast<comb::ConcatOp>(input);
      if (inputLUT) {
        if (verbose)
          llvm::errs() << "Merging " << outerLUT << " and " << inputLUT << "\n";

        ArrayAttr newAttr = updateLUTContent(outerLUT, inputLUT.getContent(),
                                             outerLUT.getContent(), rewriter);
        auto lutSelect = rewriter.replaceOpWithNewOp<LookUpTableOp>(
            outerLUT, outerLUT->getResult(0).getType(), inputLUT.getInput(),
            newAttr);
        return mlir::verify(lutSelect);
      } else if (inputConcat && inputConcat.getNumOperands()==2) {

        if (auto defOp = inputConcat.getOperand(0).getDefiningOp()) {
          if (defOp) {
            if (auto innerLUT = dyn_cast<SpecHLS::LookUpTableOp>(defOp)) {

              if (innerLUT) {

                auto innerLUTContent = innerLUT.getContent().getValue();
                auto innerLUTSize = innerLUTContent.size();
                auto innerLUTAddrWL = innerLUT.getInput().getType().getWidth();
                auto innerLUTOutWL = innerLUT.getResult().getType().getWidth();

                auto extendedOps = inputConcat.getOperands().slice(1, (inputConcat.getNumOperands() - 1));

                auto extOpsWL = 0;
                for (auto v : extendedOps) {
                  if (v.getType().isa<IntegerType>()) {
                    extOpsWL += v.getType().getIntOrFloatBitWidth();
                  } else {
                    llvm::errs() << " invalid type for " << v << " in "
                                 << outerLUT << "\n";
                    emitError(v.getLoc(),
                              llvm::formatv("Invalid type for {0} in {1}\n", v,
                                            outerLUT));
                    return failure();
                  }
                }

                if (verbose)llvm::errs() << "Merging \n\t- " << outerLUT << " with \n\t- "
                             << innerLUT << "\n\t- " << inputConcat << "\n ;";

                SmallVector<int, 1024> newcontent;
                if (verbose) llvm::errs() << "New LUT address width is " << innerLUTAddrWL
                             << ", " << extOpsWL << "\n";
                auto newOuterLUTWL = innerLUTAddrWL + extOpsWL;
                auto newOuterLUTSize = 1 << newOuterLUTWL;
                for (unsigned int k = 0; k < newOuterLUTSize; k++) {
                  // | <--- innerWidth --> | <-extWidth-> |

                  // offset in innerLut
                  auto innerLUTIndex = k % (1 << innerLUTAddrWL);
                  // offset in extOp
                  auto extIndex = (k >> innerLUTAddrWL);

                  if (innerLUTIndex < 0 || innerLUTIndex > innerLUTSize) {
                    llvm::errs()
                        << ": Out of bound innerLUT access (array has  "
                        << innerLUTContent.size() << " elements)\n";
                    return failure();
                  }

                  auto innerAttr = innerLUTContent[innerLUTIndex];
                  if (!innerAttr)
                    return failure();
                  auto innerIntAttr = innerAttr.dyn_cast<IntegerAttr>();
                  if (!innerIntAttr)
                    return failure();
                  auto innerIntValue = innerIntAttr.getInt();

                  auto newIndex = (extIndex << innerLUTOutWL) | innerIntValue;

                  if (verbose) {
                    llvm::errs() << "[" << k << "] ->  innerLut[" << innerLUTIndex << "]=" << innerIntValue << "\n";
                    llvm::errs() << "\t- newIndex = " << "{" << extIndex << ","
                                 << innerIntValue << "} = " << newIndex << "\n";

                  }

                  if (newIndex < 0 || newIndex >= outerLUTContent.size()) {
                    llvm::errs()
                        << ": Out of bound outerLut access (array has  "
                        << outerLUTContent.size() << " elements)\n";
                    return failure();
                  }

                  auto cellValueAttr =
                      outerLUTContent[newIndex].dyn_cast<IntegerAttr>();
                  if (!cellValueAttr)
                    return failure();

                  auto cellValue = cellValueAttr.getInt();

                  if (verbose) {
                    llvm::errs() << "\t- outerLUT [" << newIndex << "] = " << cellValue << "\n";
                    llvm::errs() << "\t- mergedLUT [" << k << "] = " << cellValue << "\n";
                  }

                  newcontent.push_back(cellValue);
                }

                auto arrayAttr = rewriter.getI32ArrayAttr(newcontent);

                SmallVector<mlir::Value> newOperands;
                newOperands.push_back(innerLUT.getOperand());
                for (auto value : extendedOps) {
                  newOperands.push_back(value);
                  // eliminate inner LUT
                }
                auto newConcat = rewriter.create<comb::ConcatOp>(
                    outerLUT.getLoc(), newOperands);

                if (verbose)llvm::errs() << " new concat " << newConcat << " \n";

                auto newLUT = rewriter.create<SpecHLS::LookUpTableOp>(
                    outerLUT.getLoc(), outerLUT.getResult().getType(),
                    newConcat, arrayAttr);

                if (verbose) llvm::errs() << " replace \n\t" << outerLUT << "\n by \n\t"
                             << newLUT << " \n";
                rewriter.replaceOp(outerLUT, newLUT);

                mlir::verify(newConcat);
                mlir::verify(newLUT);
                mlir::verify(newLUT->getParentOp());
                return success();
              }
            }
          }
        }
      }
    }

    return failure();
  }
};

struct ConcatMergingPattern : OpRewritePattern<comb::ConcatOp> {

  bool verbose = false;

  using OpRewritePattern<ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp op,
                                PatternRewriter &rewriter) const override {

    for (size_t i = 0; i < op.getOperands().size(); i++) {
      auto input = op.getOperand(i).getDefiningOp();
      if (input) {
        auto inputConcat = dyn_cast<ConcatOp>(*input);
        if (inputConcat) {

          if (verbose) llvm::errs() << "Merging " << op << " and " << inputConcat << "\n";

          auto lowerSlice = op->getOperands().drop_back(op.getNumOperands() - i);
          auto upperSlice = op->getOperands().drop_front(i + 1);

          llvm::SmallVector<Value, 8> operands;

          for (auto v : upperSlice)
            operands.push_back(v);
          for (auto v : inputConcat->getOperands())
            operands.push_back(v);
          for (auto v : lowerSlice)
            operands.push_back(v);

          int bw = 0;
          for (auto v : operands)
            bw += v.getType().getIntOrFloatBitWidth();

          auto newConcat = rewriter.create<ConcatOp>(op.getLoc(), rewriter.getIntegerType(bw), operands);

          mlir::verify(newConcat);
          if (verbose)llvm::errs() << "Merge concat result is = " << newConcat << "\n";

          rewriter.replaceOp(op, newConcat);

          if (verbose) llvm::errs() << "Parent Operation is \n " << *newConcat->getParentOp() << "\n";
          return success();
        }
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
    patterns.insert<ConcatMergingPattern>(ctx);
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