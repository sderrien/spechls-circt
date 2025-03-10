//===- MergeGammas.cpp - Arith-to-comb mapping pass ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the MergeGammas pass.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

namespace SpecHLS {
//
// void sliceGammaOpInputs(GammaOp op,  ArrayRef<u_int32_t> entries,
// PatternRewriter &rewriter) {
//  //
//  auto loc = op->getLoc();
//  auto type = op->getResult(0).getType();
//
//  auto innerVec = SmallVector<Value>();
//  auto outerVec = SmallVector<Value>();
//
//  auto indexInner = SmallVector<u_int32_t >();
//  auto indexOuter = SmallVector<u_int32_t >();
//
//  for (int k=0;k<op.getInputs().size() ;k++) {
//    outerVec.push_back(op->getOperand(k));
//    indexOuter.push_back(k);
//  }
//
//  for (u_int32_t e :entries) {
//    auto value = op->getOperand(e);
//    innerVec.push_back(value);
//    indexOuter.erase(&e);
//    indexInner.push_back(e);
//  }
//
//  indexOuter.erase(&e);
//
//  auto innerGamma =
//  rewriter.create<SpecHLS::GammaOp>(loc,type,op.getNameAttr(),op.getOperand(0),
//  innerVec); auto outerGamma =
//  rewriter.create<SpecHLS::GammaOp>(loc,type,op.getNameAttr(),op.getOperand(0),
//  outerVec);
//
//}

struct FactorGammaInputsPattern : OpRewritePattern<GammaOp> {

  using OpRewritePattern<GammaOp>::OpRewritePattern;
  bool verbose = false;

private:
  //
  //
  // TODO: Comment
  //
  //
  bool isMatch(mlir::Operation *a, mlir::Operation *b) const {
    if (a->getName() != b->getName()) {
      return false;
    }

    if (a->getNumOperands() != b->getNumOperands()) {
      return false;
    }

    unsigned numOp = a->getNumOperands();
    for (unsigned i = 0; i < numOp; i++) {
      if (a->getOperand(i).getType() != b->getOperand(i).getType()) {
        return false;
      }
    }

    if (b->getNumResults() == 1 && a->getNumResults() == 1) {

      return TypeSwitch<Operation *, bool>(a)
          .Case<circt::comb::AddOp>([&](auto op) { return true; })
          .Case<circt::comb::SubOp>([&](auto op) { return true; })
          .Case<circt::comb::ICmpOp>([&](auto op) { return true; })
          .Case<circt::comb::AndOp>([&](auto op) { return true; })
          .Case<circt::comb::OrOp>([&](auto op) { return true; })
          .Case<circt::comb::XorOp>([&](auto op) { return true; })
          .Case<circt::comb::ExtractOp>([&](auto op) { return true; })
          .Case<circt::comb::ConcatOp>([&](auto op) { return true; })
          .Case<circt::hw::ConstantOp>([&](auto op) {
            auto bc = dyn_cast<circt::hw::ConstantOp>(b);
            return bc.getValue() == op.getValue();
          })
          .Case<circt::comb::MuxOp>([&](auto op) { return true; })
          .Case<circt::comb::TruthTableOp>([&](auto op) {
            auto bc = dyn_cast<circt::comb::TruthTableOp>(b);
            return bc.getLookupTable() == op.getLookupTable();
          })
          .Case<SpecHLS::LookUpTableOp>([&](auto op) {
            // FIXME : could be true if both LUT content do match
            return false;
          })
          .Case<SpecHLS::GecosOp>([&](auto op) {
            // FIXME : could be true if both LUT content do match
            auto bc = dyn_cast<SpecHLS::GecosOp>(b);
            return op->getName() == bc->getName();
          })
          .Case<SpecHLS::AlphaOp>([&](auto op) { return true; })
          .Default([&](auto op) { return false; });
    };
    return false;
  }

  //
  //
  // TODO: Comment
  //
  //
  bool checkMatch(mlir::Operation::operand_range inputs, int i, int k,
                  SmallVector<int32_t> &matches) const {
    if (i >= inputs.size() || k >= inputs.size() || i < 0 || k < 0) {
      if (verbose)
        llvm::errs() << "\t- out of bounds  " << i << "," << k << " in "
                     << inputs.size() << "\n";
      return false;
    }

    Value va = inputs[i];
    Value vb = inputs[k];
    if (i == k || (va.getType() != vb.getType())) {
      return false;
    }

    auto a = va.getDefiningOp();
    auto b = vb.getDefiningOp();
    if (a == NULL || b == NULL) {
      return false;
    }
    auto match = isMatch(a, b);
    if (verbose) {
      if (match) {
        llvm::errs() << "## Match between \n\t->" << *a << "\n\t->" << *b
                     << "\n";
      }
    }
    return match;
  }

  //
  //
  //  Cette fonction construit une liste avec la position des arguments produit
  //  par des op d'une même classe d'equivalence (definie par checkmatch)
  //
  LogicalResult extractMatches(GammaOp op,
                               SmallVector<int32_t> &matches) const {
    // llvm::errs() << "analyzing  " << op << " \n";

    u_int32_t nbInputs = op.getInputs().size();
    for (int i = 0; i < nbInputs; i++) {
      auto rootValue = op.getInputs()[i];
      auto root = rootValue.getDefiningOp();
      if (root == NULL || root->getNumResults() != 1 ||
          !root->getResult(0).hasOneUse()) {
        continue;
      }

      /* build the set of nodes (at pos K>i) that match the current target
       * node */
      matches.clear();
      matches.push_back(i);
      for (int k = i + 1; k < nbInputs; k++) {
        if (checkMatch(op.getInputs(), i, k, matches)) {
          matches.push_back(k);
        }
      }

      if (matches.size() > 1) {
        if (verbose) {
          llvm::errs() << "##### match set {\n";
          for (auto m : matches) {
            auto defOp = op.getInputs()[m].getDefiningOp();
            llvm::errs() << "\tin[" << m << "] -> " << *defOp << "\n";
          }
          llvm::errs() << "}\n";
        }

        return success();
      }
    }
    return failure();
  }

  //
  //
  //
  //
  //
  SpecHLS::GammaOp createGammaForOperand(u_int32_t j, GammaOp op,
                                         LookUpTableOp innerLUT,
                                         SmallVector<int32_t> &matches,
                                         PatternRewriter &rewriter) const {
    SmallVector<Value> args;
    if (verbose)
      llvm::errs() << "-Extracting all " << j << "th args in matched ops \n";

    for (auto mid : matches) {
      if (mid >= op.getInputs().size()) {
        llvm::errs() << "Inconsistent match index " << mid << " for " << op
                     << "\n";
        return NULL;
      }
      auto value = op->getOperand(mid + 1);
      if (value == NULL) {
        llvm::errs() << "No value at " << mid << "\n";
        return NULL;
      }
      // llvm::errs() << "Value " << value << "\n";
      auto matchedOp = value.getDefiningOp();

      if (matchedOp == NULL) {
        llvm::errs() << "No defining op for " << value << " at " << mid << "\n";
        return NULL;
      } else {

        // llvm::errs() << "Matched op  " << matchedOp->getName() << "\n";

        if (j >= matchedOp->getNumOperands()) {
          if (verbose)
            llvm::errs() << "No operand " << j << " in " << *matchedOp << "\n";
          continue;
        }

        //      if (verbose)
        //        llvm::errs() << "\t-analyzing match " << *matchedOp << " at
        //        offset  "
        //                     << j << "\n";

        auto matchedArgValue = matchedOp->getOperand(j);
        if (!matchedArgValue) {
          llvm::errs() << "No valid argValue at offset " << j << "\n";
          return NULL;
        }

        //      if (verbose)
        //        llvm::errs() << "\t-extracting value " << matchedArgValue << "
        //        \n";
        args.push_back(matchedArgValue);
      }
    }
    auto gamma = rewriter.create<SpecHLS::GammaOp>(
        op->getLoc(), args[0].getType(), op.getName(), innerLUT->getResult(0),
        args);
    if (verbose)
      llvm::errs() << "- Creating inner gamma " << gamma << " at offset  " << j
                   << "\n";
    return gamma;
  }

  //
  //
  //
  //
  //
  int32_t analyzeMatchedOps(GammaOp op, SmallVector<int32_t> &matches) const {
    int32_t nbMatchInputs = -1;
    Operation *rootMatchedOp;
    u_int32_t nbInputs = op.getInputs().size();

    /* computes the number of inputs on matched ops */
    for (auto mid : matches) {
      auto matchedValue = op.getInputs()[mid];
      if (!matchedValue) {
        continue;
      }

      auto matchedOp = matchedValue.getDefiningOp();
      if (!matchedOp) {
        continue;
      }

      auto nbOperands = matchedOp->getNumOperands();
      if (nbMatchInputs < 0) {
        nbMatchInputs = nbOperands;
        // We keep track of one of the matched op
        rootMatchedOp = matchedOp;
        if (verbose)
          llvm::errs() << "Reference matched op " << *matchedOp << "\n";
      }
      if (nbOperands != nbMatchInputs) {
        llvm::errs() << "Inconsistent arity for " << *matchedOp << ", expected "
                     << nbOperands << "\n";
        return -1;
      }
    }
    return nbMatchInputs;
  }

  //
  //
  //
  //
  //
  SpecHLS::LookUpTableOp
  createInnerReindexingLUT(GammaOp op, SmallVector<int32_t> &matches,
                           PatternRewriter &rewriter) const {
    SmallVector<int> lutContent;
    auto nbInputs = op.getInputs().size();
    int pos = 0;
    if (verbose)
      llvm::errs() << "Reindexing Gamma op inputs for " << op << "\n";
    for (int k = 0; k < nbInputs; k++) {
      auto newIndex = 0;
      if (std::count_if(matches.begin(), matches.end(),
                        [&](const auto &e) { return (k == e); })) {
        newIndex = pos++;
      }
      lutContent.push_back(newIndex);
      if (verbose)
        llvm::errs() << "- input[" << k << "]" << op.getInputs()[k] << " -> "
                     << newIndex << "\n";
    }

    auto inputType = op.getSelect().getType();
    auto innerLUTInputBW = inputType.getIntOrFloatBitWidth();
    auto innerLUTOutputBW = APInt(32, matches.size() - 1).getActiveBits();

    for (int k = lutContent.size(); k < (1 << innerLUTInputBW); k++) {
      lutContent.push_back(0);
    }

    if (verbose) {
      llvm::errs() << " - Matches size = " << matches.size() << "\n";
      llvm::errs() << " - Gamma select " << op.getSelect() << "\n";
      llvm::errs() << "   - select BW = " << op.getInputs().size() << "\n";
      llvm::errs() << "   - LUT inputBW  " << innerLUTInputBW << "\n";
      llvm::errs() << "  - LUT outputBW " << innerLUTOutputBW << "\n";
      llvm::errs() << " - LUT size " << lutContent.size() << "\n";
    }

    if ((1 << innerLUTInputBW) != lutContent.size()) {
      emitError(op->getLoc(), "Inconsistent LUT content");
    }
    if (op != NULL && op.getSelect() == NULL) {
      llvm::errs() << " -null port \n";
      emitError(op->getLoc(), "Null port");
    }

    auto lut = rewriter.create<SpecHLS::LookUpTableOp>(
        op->getLoc(), rewriter.getIntegerType(innerLUTOutputBW), op.getSelect(),
        rewriter.getI32ArrayAttr(lutContent));

    if (verbose)
      llvm::errs() << " ## Created " << lut << "\n";
    verify(lut);
    return lut;
  }

public:
  LogicalResult matchAndRewrite(GammaOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<int32_t> matches;
    u_int32_t nbInputs = op.getInputs().size();
    if (!extractMatches(op, matches).succeeded()) {
      return failure();
    }
    if (verbose)
      llvm::errs() << "#### FactorGammaInputs pass for " << op << "\n";
    for (auto k = 0; k < matches.size(); k++) {
      if (matches[k] >= op.getInputs().size()) {
        llvm::errs() << "  -Inconsistent match table ! " << matches[k] << "\n";
      }
      if (verbose) {
        llvm::errs() << "  - match[" << k << "]= " << op.getInputs()[matches[k]]
                     << "\n";
      }
    }

    auto firstMatchIndex = matches[0];

    auto rootValue = op.getInputs()[firstMatchIndex];

    if (verbose)
      llvm::errs() << "- root match is " << rootValue << "\n";
    auto root = rootValue.getDefiningOp();
    if (root->getNumResults() != 1 || !root->getResult(0).hasOneUse()) {
      if (verbose)
        llvm::errs() << " cannot factor root " << rootValue << "\n";
      return failure();
    }

    auto nbMatchInputs = analyzeMatchedOps(op, matches);

    //  if (verbose)
    //    llvm::errs() << " - matched inputs have " << nbMatchInputs << "
    //    each\n";

    auto innerLUT = createInnerReindexingLUT(op, matches, rewriter);
    //  if (innerLUT != NULL)
    //    if (verbose)
    //      llvm::errs() << " - created LUT " << innerLUT << " each\n";

    SmallVector<Value> newGammas;
    for (u_int32_t j = 0; j < nbMatchInputs; j++) {
      auto gamma = createGammaForOperand(j, op, innerLUT, matches, rewriter);
      newGammas.push_back(gamma);
    }

    assert(newGammas.size() == nbMatchInputs);

    //  if (verbose)
    //    llvm::errs() << "Rewiring " << nbMatchInputs << " arguments in the
    //    root op "
    //                 << *root << "\n";
    for (u_int32_t j = 0; j < nbMatchInputs; j++) {
      //    if (verbose)
      //      llvm::errs() << "\t-replace arg[" << j << "]=" <<
      //      root->getOperand(j)
      //                   << " by  " << newGammas[j] << "\n";
      root->setOperand(j, newGammas[j]);
    }

    if (verbose)
      llvm::errs() << "Root op is now " << *root << "\n";

    //  if (verbose)
    //    llvm::errs() << "Before rewiring outer gamma " << op << "\n";
    for (auto mid : matches) {
      //    if (verbose)
      //      llvm::errs() << "\t- Update operand from " << op->getOperand(mid +
      //      1)
      //                   << " to " << rootValue << "\n";
      op->setOperand(mid + 1, rootValue);
      //    if (verbose)
      //      llvm::errs() << "\t - " << op << "\n";
    }
    //  if (verbose)
    //    llvm::errs() << "After rewiring outer gamma " << op << "\n";
    //  if (verbose)
    //    llvm::errs() << "##################################################"
    //                    "############\n\n\n";
    //  if (verbose)
    //    llvm::errs() << "##################################################"
    //                    "############\n";

    return success();
  }
};

struct FactorGammaInputsPass
    : public impl::FactorGammaInputsPassBase<FactorGammaInputsPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    patterns.insert<FactorGammaInputsPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      llvm::errs() << "partial conversion failed pattern  \n";
      signalPassFailure();
    }
    mlir::verify(getOperation(), true);
  }
};
std::unique_ptr<OperationPass<>> createFactorGammaInputsPass() {
  return std::make_unique<FactorGammaInputsPass>();
}

} // namespace SpecHLS
