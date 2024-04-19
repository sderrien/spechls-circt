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
  bool isMatch(mlir::Operation *a, mlir::Operation *b) const {
    if (a->getName() == b->getName()) {
      if (verbose)
        llvm::outs() << "\t- comparing opname " << a->getName() << " and "
                     << b->getName() << "\n";
      if (verbose)
        llvm::outs() << "\t- comparing dialect "
                     << a->getDialect()->getNamespace() << " and "
                     << b->getDialect()->getNamespace() << "\n";
      if (a->getNumOperands() == b->getNumOperands()) {
        if (verbose)
          llvm::outs() << "\t- comparing #operands " << a->getNumOperands()
                       << " and " << b->getNumOperands() << "\n";
        if (b->getNumResults() == 1 && a->getNumResults() == 1) {
          if (verbose)
            llvm::outs() << "\t- match !\n";
          return true;
        }
      }
    }
    return false;
  }

  bool checkMatch(mlir::Operation::operand_range inputs, int i, int k,
                  SmallVector<int32_t> &matches) const {
    if (i <= inputs.size() && k <= inputs.size()) {
      Value va = inputs[i];
      Value vb = inputs[k];

      if (i != k && (va.getType() == vb.getType())) {
        auto a = va.getDefiningOp();
        auto b = vb.getDefiningOp();

        if (a != NULL && b != NULL) {
          if (verbose)
            llvm::outs() << "\t- comparing  (" << i << "," << k << ") ->" << *a
                         << " and " << *b << "\n";
          return isMatch(a, b);
        }
      }

    } else {
      if (verbose)
        llvm::outs() << "\t- out of bounds  " << i << "," << k << " in "
                     << inputs.size() << "\n";
    }
    return false;
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
      if (root != NULL) {
        if (root->getNumResults() == 1) {
          if (root->getResult(0).hasOneUse()) {

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
              if (verbose)
                llvm::outs() << "match {\n";
              for (auto m : matches) {
                auto defOp = op.getInputs()[m].getDefiningOp();
                if (verbose)
                  llvm::outs() << "\tin[" << m << "] -> " << *defOp << "\n";
              }
              if (verbose)
                llvm::outs() << "}\n";

              return success();
            }
          }
        }
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
      llvm::outs() << "-Extracting all " << j << "th args in matched ops \n";
    for (auto mid : matches) {
      auto value = op.getInputs()[mid];
      if (value) {
        auto matchedOp = value.getDefiningOp();
        if (matchedOp) {
          if (j < matchedOp->getNumOperands()) {
            if (verbose)
              llvm::outs() << "\t-analyzing match " << *matchedOp
                           << " at offset  " << j << "\n";

            auto matchedArgValue = matchedOp->getOperand(j);
            if (matchedArgValue) {
              if (verbose)
                llvm::outs()
                    << "\t-extracting value " << matchedArgValue << " \n";
              args.push_back(matchedArgValue);

            } else {
              llvm::errs() << "No valid arrgValue at offset " << j << "\n";
              return NULL;
            }
          }
        } else {
          llvm::errs() << "No defining op for value " << value << " at offset "
                       << mid << "\n";
          return NULL;
        }
      } else {
        llvm::errs() << "Invalid value at " << mid << "\n";
        return NULL;
      }
    }
    auto gamma = rewriter.create<SpecHLS::GammaOp>(
        op->getLoc(), op->getResultTypes(), op.getName(),
        innerLUT->getResult(0), args);
    if (verbose)
      llvm::outs() << "- Creating inner gamma " << gamma << " at offset  " << j
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
      if (matchedValue) {
        auto matchedOp = matchedValue.getDefiningOp();
        if (matchedOp) {
          auto nbOperands = matchedOp->getNumOperands();
          if (nbMatchInputs < 0) {
            nbMatchInputs = nbOperands;
            // We keep track of one of the matched op
            rootMatchedOp = matchedOp;
            if (verbose)
              llvm::outs() << "Reference matched op " << *matchedOp << "\n";
          }
          if (nbOperands != nbMatchInputs) {
            llvm::errs() << "Inconsistent arity for " << *matchedOp
                         << ", expected " << nbOperands << "\n";
            return -1;
          }
        }
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
      llvm::outs() << "Reindexing inner Gamma op inputs \n";
    for (int k = 0; k < nbInputs; k++) {
      auto newIndex = 0;
      if (std::count_if(matches.begin(), matches.end(),
                        [&](const auto &e) { return (k == e); })) {
        newIndex = pos++;
      }
      lutContent.push_back(newIndex);
      if (verbose)
        llvm::outs() << "- " << k << " -> " << newIndex << "\n";
    }

    auto innerBW = APInt(32, matches.size()).ceilLogBase2();
    return rewriter.create<SpecHLS::LookUpTableOp>(
        op->getLoc(), rewriter.getIntegerType(innerBW), op.getSelect(),
        rewriter.getI32ArrayAttr(lutContent));
  }

public:
  LogicalResult matchAndRewrite(GammaOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<int32_t> matches;
    u_int32_t nbInputs = op.getInputs().size();
    if (extractMatches(op, matches).succeeded()) {
      auto firstMatchIndex = matches[0];
      auto rootValue = op.getInputs()[firstMatchIndex];
      auto root = rootValue.getDefiningOp();
      if (root->getNumResults() == 1) {
        if (root->getResult(0).hasOneUse()) {

          auto innerLUT = createInnerReindexingLUT(op, matches, rewriter);
          auto nbMatchInputs = analyzeMatchedOps(op, matches);

          SmallVector<Value> newGammas;
          for (u_int32_t j = 0; j < nbMatchInputs; j++) {
            auto gamma =
                createGammaForOperand(j, op, innerLUT, matches, rewriter);
            newGammas.push_back(gamma);
          }

          assert(newGammas.size() == nbMatchInputs);

          if (verbose)
            llvm::outs() << "Rewiring " << nbMatchInputs
                         << " arguments in the root op " << *root << "\n";
          for (u_int32_t j = 0; j < nbMatchInputs; j++) {
            if (verbose)
              llvm::outs() << "\t-replace arg[" << j
                           << "]=" << root->getOperand(j) << " by  "
                           << newGammas[j] << "\n";
            root->setOperand(j, newGammas[j]);
          }

          if (verbose)
            llvm::outs() << "Root op is now " << *root << "\n";

          if (verbose)
            llvm::outs() << "Before rewiring outer gamma " << op << "\n";
          for (auto mid : matches) {
            if (verbose)
              llvm::outs() << "\t- Update operand from "
                           << op->getOperand(mid + 1) << " to " << rootValue
                           << "\n";
            op->setOperand(mid + 1, rootValue);
            if (verbose)
              llvm::outs() << "\t - " << op << "\n";
          }
          if (verbose)
            llvm::outs() << "After rewiring outer gamma " << op << "\n";
          if (verbose)
            llvm::outs() << "##################################################"
                            "############\n\n\n";
          if (verbose)
            llvm::outs() << "##################################################"
                            "############\n";

          if (verbose)
            llvm::outs() << *op->getParentOp() << "\n";

          return success();
        }
      }
    }
    return failure();

    //
    //
    //
    //
    //    extractMatches(op,matches);
    //    // llvm::errs() << "analyzing  " << op << " \n";
    //    for (int i = 0; i < ; i++) {
    //      auto rootValue = op.getInputs()[i];
    //      auto root = rootValue.getDefiningOp();
    //      if (root != NULL) {
    //        if (root->getNumResults() == 1) {
    //          if (root->getResult(0).hasOneUse()) {
    //            SmallVector<int32_t> matches;
    //
    //            /* build the set of nodes (at pos K>i) that match the current
    //            target
    //             * node */
    //
    //
    //            if (verbose) llvm::outs() << "Target " << i << " -> " << *root
    //            << "\n"; matches.push_back(i); for (int k = i + 1; k <
    //            nbInputs; k++) {
    //              checkmatch(op.getInputs(), i, k, matches);
    //            }
    //
    //            if (matches.size() > 1) {
    //              if (verbose) llvm::outs() << "match " << i << " -> " <<
    //              *root << "-> {"; for (auto m : matches) {
    //                auto defOp = op.getInputs()[m].getDefiningOp();
    //                if (verbose) llvm::outs() << "" << m << " -> " << *defOp
    //                << ", ";
    //              }
    //              if (verbose) llvm::outs() << "}\n";
    //
    //              /*
    //               * creates a LUT for reindexing inner Gamma inputs
    //               */
    //              createInnerReindexingLUT(op, matches, rewriter)
    //              SmallVector<int> lutContent;
    //              int pos = 0;
    //              if (verbose) llvm::outs() << "Reindexing inner Gamma op
    //              inputs \n"; for (int k = 0; k < nbInputs; k++) {
    //                auto newIndex = 0;
    //                if (std::count_if(matches.begin(), matches.end(),
    //                                  [&](const auto &e) { return (k == e);
    //                                  })) {
    //                  newIndex = pos++;
    //                }
    //                lutContent.push_back(newIndex);
    //                if (verbose) llvm::outs() << "- " << k << " -> " <<
    //                newIndex << "\n";
    //              }
    //
    //              /*
    //               *
    //               */
    //              auto innerBW = APInt(32, matches.size()).ceilLogBase2();
    //              auto innerLut = rewriter.create<SpecHLS::LookUpTableOp>(
    //                  op->getLoc(), rewriter.getIntegerType(innerBW),
    //                  op.getSelect(), rewriter.getI32ArrayAttr(lutContent));
    //
    //              /*
    //               * creates a LUT for reindexing outer Gamma inputs, by
    //               skipping
    //               * inpouts that have hoisted out to the inner Gamma
    //               */
    //              SmallVector<int> outerLutContent;
    //              pos = 0;
    //              for (int k = 0; k <= i; k++) {
    //                outerLutContent.push_back(k);
    //              }
    //              for (int k = i + 1; k < nbInputs; k++) {
    //                if (std::count_if(
    //                        matches.begin(), matches.end(),
    //                        [&](const auto &item) { return (k == item); })) {
    //                  outerLutContent.push_back(i);
    //                } else {
    //                  outerLutContent.push_back(pos++);
    //                }
    //              }
    //
    //              if (verbose) llvm::outs() << "Outer gamma " << op << "
    //              reindexing  \n"; for (int k = 0; k < nbInputs; k++) {
    //                if (verbose) llvm::outs() << " - input " << k << "
    //                reindexed to "
    //                             << outerLutContent[k] << " \n";
    //              }
    //
    //              auto outerLut = rewriter.create<SpecHLS::LookUpTableOp>(
    //                  op->getLoc(), op->getResult(0).getType(),
    //                  op.getSelect(),
    //                  rewriter.getI32ArrayAttr(outerLutContent));
    //
    //              /*
    //               * Create a LUT for reindexing inner gamma inputs.
    //               *
    //               */
    //
    //              SmallVector<Value> newGammas;
    //
    //              u_int32_t nbMatchInputs = -1;
    //              Operation *rootMatchedOp;
    //              /* computes the number of inputs on matched ops */
    //              for (auto mid : matches) {
    //                auto matchedValue = op.getInputs()[mid];
    //                if (matchedValue) {
    //                  auto matchedOp = matchedValue.getDefiningOp();
    //                  if (matchedOp) {
    //                    auto nbOperands = matchedOp->getNumOperands();
    //                    if (nbMatchInputs < 0) {
    //                      nbMatchInputs = nbOperands;
    //                      // We keep track of one of the matched op
    //                      rootMatchedOp = matchedOp;
    //                      if (verbose) llvm::outs() << "Reference matched op "
    //                      << *matchedOp << "\n";
    //                    }
    //                    if (nbOperands != nbMatchInputs) {
    //                      llvm::errs() << "Inconsistent arity for " <<
    //                      *matchedOp
    //                                   << ", expected " << nbInputs << "\n";
    //                      return failure();
    //                    }
    //                  }
    //                }
    //              }
    //
    //              for (u_int32_t j = 0; j < nbMatchInputs; j++) {
    //                createGammaForOperand()
    //                SmallVector<Value> args;
    //                if (verbose) llvm::outs()
    //                    << "-Extracting all " << j << "th args in matched ops
    //                    \n";
    //
    //                for (auto mid : matches) {
    //                  auto matchedValue = op.getInputs()[mid];
    //                  if (matchedValue) {
    //                    auto matchedOp = matchedValue.getDefiningOp();
    //                    if (matchedOp) {
    //                      if (j < matchedOp->getNumOperands()) {
    //                        if (verbose) llvm::outs() << "\t-analyzing match "
    //                        << *matchedOp
    //                                     << " at offset  " << j << "\n";
    //
    //                        auto matchedArgValue = matchedOp->getOperand(j);
    //                        if (matchedArgValue) {
    //                          if (verbose) llvm::outs() << "\t-extracting
    //                          value "
    //                                       << matchedArgValue << " \n";
    //                          args.push_back(matchedArgValue);
    //
    //                        } else {
    //                          llvm::errs()
    //                              << "No valid arrgValue at offset " << j <<
    //                              "\n";
    //                          return failure();
    //                        }
    //                      }
    //                    } else {
    //                      llvm::errs()
    //                          << "No defining op for value " << matchedValue
    //                          << " at offset " << mid << "\n";
    //                      return failure();
    //                    }
    //                  } else {
    //                    llvm::errs() << "Invalid value at " << mid << "\n";
    //                    return failure();
    //                  }
    //                }
    //                auto gamma = rewriter.create<SpecHLS::GammaOp>(
    //                    op->getLoc(), op->getResultTypes(), op.getName(),
    //                    innerLut->getResult(0), args);
    //                if (verbose) llvm::outs() << "- Creating inner gamma " <<
    //                gamma
    //                             << " at offset  " << j << "\n";
    //                newGammas.push_back(gamma);
    //
    //                if (root->isBeforeInBlock(gamma)) {
    //                  root->moveAfter(gamma);
    //                }
    //              }
    //
    //              assert(newGammas.size() == nbMatchInputs);
    //
    //              if (verbose) llvm::outs() << "Rewiring " << nbMatchInputs
    //                           << " arguments in the root op " << *root <<
    //                           "\n";
    //              for (u_int32_t j = 0; j < nbMatchInputs; j++) {
    //                if (verbose) llvm::outs()
    //                    << "replace arg[" << j << "]=" << root->getOperand(j)
    //                    << " by  " << newGammas[j] << "\n";
    //                root->setOperand(j, newGammas[j]);
    //              }
    //
    //              if (verbose) llvm::outs() << "Root op is now " << root <<
    //              "\n";
    //
    //              for (auto mid : matches) {
    //                if (verbose) llvm::outs() << "change value  " <<
    //                op->getOperand(mid)
    //                             << " by " << rootValue << "\n";
    //                op->setOperand(mid + 1, rootValue);
    //              }
    //              if (verbose) llvm::outs() << "After rewiring outer gamma "
    //              << op << "\n"; return success();
    //            }
    //          }
    //        }
    //      }
    //    }
    //    return failure();
  }
};

struct EliminateRedundantGammaInputs : OpRewritePattern<GammaOp> {
  using OpRewritePattern<GammaOp>::OpRewritePattern;
  bool verbose = false;

private:
  //
  //
  //  Cette fonction construit une liste avec la position des arguments produit
  //  par des op d'une même classe d'equivalence (definie par checkmatch)
  //
  LogicalResult extractMatches(GammaOp op,
                               SmallVector<int32_t> &matches) const {
    // llvm::errs() << "analyzing  " << op << " \n";
    auto nbInputs = op.getInputs().size();
    for (uint32_t i = 0; i < nbInputs; i++) {
      auto rootValue = op.getInputs()[i];
      if (rootValue != NULL) {
        matches.clear();
        matches.push_back(i);
        for (uint32_t k = i + 1; k < nbInputs; k++) {
          if (op.getInputs()[k] == rootValue) {
            matches.push_back(k);
          }
        }
        if (matches.size() > 1) {
          return success();
        }
      }
    }

    return failure();
  }

  SpecHLS::LookUpTableOp
  createOuterReindexingLUT(GammaOp op, SmallVector<int32_t> &matches,
                           PatternRewriter &rewriter) const {
    /*
     * creates a LUT for reindexing outer Gamma inputs, by skipping
     * inpouts that have hoisted out to the inner Gamma
     */
    auto nbInputs = op.getInputs().size();
    auto firstMatchIndex = matches[0];
    SmallVector<int> outerLutContent;
    u_int32_t pos = 0;
    for (int k = 0; k <= firstMatchIndex; k++) {
      outerLutContent.push_back(k);
    }
    for (int k = firstMatchIndex + 1; k < nbInputs; k++) {
      if (std::count_if(matches.begin(), matches.end(),
                        [&](const auto &item) { return (k == item); })) {
        outerLutContent.push_back(firstMatchIndex);
      } else {
        outerLutContent.push_back(pos++);
      }
    }

    if (verbose)
      llvm::outs() << "Outer gamma " << op << " reindexing  \n";
    for (int k = 0; k < nbInputs; k++) {
      if (verbose)
        llvm::outs() << " - input " << k << " reindexed to "
                     << outerLutContent[k] << " \n";
    }

    return rewriter.create<SpecHLS::LookUpTableOp>(
        op->getLoc(), op.getSelect().getType(), op.getSelect(),
        rewriter.getI32ArrayAttr(outerLutContent));
  }

public:
  LogicalResult matchAndRewrite(GammaOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<int32_t> matches;
    u_int32_t nbInputs = op.getInputs().size();
    if (extractMatches(op, matches).succeeded()) {
      if (matches.size() == nbInputs) {
        llvm::errs() << "Eliminating " << op
                     << " because it has all the same inputs :\n";
        op.getResult().replaceAllUsesWith(op.getInputs()[0]);
        rewriter.eraseOp(op);
        return success();
      } else {

        auto lut = createOuterReindexingLUT(op, matches, rewriter);

        // filter out redundant input values
        SmallVector<Value> args;
        for (int32_t k = 0; k < nbInputs; k++) {
          bool found = false;
          for (u_int32_t j = 1; j < matches.size(); j++) {
            if (k == matches[j]) {
              found = true;
              break;
            }
          }
          if (!found) {
            args.push_back(op.getInputs()[k]);
          }
        }
        auto gamma = rewriter.create<SpecHLS::GammaOp>(
            op->getLoc(), op->getResultTypes(), op.getName(), lut->getResult(0),
            args);

        llvm::errs() << "Simplifying  " << op << " into  " << gamma << "\n";
        rewriter.replaceOp(op, gamma);
        return success();
      }

    } else {
      return failure();
    }
  }
};
struct FactorGammaInputsPass
    : public impl::FactorGammaInputsPassBase<FactorGammaInputsPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    patterns.insert<FactorGammaInputsPattern>(ctx);
    patterns.insert<EliminateRedundantGammaInputs>(ctx);

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

struct EliminateRedundantGammaInputsPass
    : public impl::EliminateRedundantGammaInputsPassBase<
          EliminateRedundantGammaInputsPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    patterns.insert<FactorGammaInputsPattern>(ctx);
    patterns.insert<EliminateRedundantGammaInputs>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      llvm::errs() << "partial conversion failed pattern  \n";
      signalPassFailure();
    }
    mlir::verify(getOperation(), true);
  }
};
std::unique_ptr<OperationPass<>> createEliminateRedundantGammaInputsPass() {
  return std::make_unique<EliminateRedundantGammaInputsPass>();
}

} // namespace SpecHLS