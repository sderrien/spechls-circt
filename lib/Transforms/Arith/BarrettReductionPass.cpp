//===- BarrettReductionPass.cpp - Barrett Reduction Optimization Pass -----===//
//
// This file implements a pass that rewrites certain comb.mod operations
// (mod by a constant) into Barrett reduction sequences.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"

#include "BarrettReductionPass.h" // Generated from BarrettReductionPass.td

using namespace mlir;
using namespace circt::comb;

namespace mlir {
std::unique_ptr<Pass> createBarrettReductionPass();
void registerBarrettReductionPass();
} // end namesp

/// ----------------------------------------------------------------------------
/// Explanation of Barrett Reduction:
///
/// Barrett reduction is a method to compute `x mod M` using a 'precomputed
/// constant' that depends on M. Typically, if `k` is chosen such that
/// 2^k > M, the precomputed constant is `mu = floor((2^k) / M)`.
///
/// Then:
///   q = floor((x * mu) / 2^k)
///   r = x - q * M
///   if (r >= M) then r -= M
///
/// The final `r` is `x mod M`.
///
/// In hardware, we can typically pick `k` to match the bit width of `x`.
/// For example, if we are dealing with 32-bit integers:
///   k = 32
///   mu = floor((1 << 32) / M)
///
/// Then:
///   q = (x * mu) >> 32
///   r = x - q * M
///   if (r >= M) r -= M
///
/// The pass below performs a simple version of this substitution for
/// constant M in `comb.mod %x, %M`.
///
/// Note that we do not handle special corner cases (e.g., sign issues),
/// so for a real production pass, you might refine it further.
/// ----------------------------------------------------------------------------

namespace {

/// Pattern rewriting: matches a comb.mod where the RHS is a constant
/// and replaces it with a Barrett reduction sequence.
struct BarrettReductionPattern : public OpRewritePattern<comb::ModOp> {
  using OpRewritePattern<comb::ModOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::ModOp modOp,
                                PatternRewriter &rewriter) const override {
    // 1. Check if RHS is a constant
    auto rhsCstOp = modOp.getRhs().getDefiningOp<hw::ConstantOp>();
    if (!rhsCstOp)
      return failure(); // Not a constant, skip.

    // 2. Extract the constant modulus M
    APInt modulus = rhsCstOp.getValue().cast<APInt>();
    if (modulus.isZero()) {
      // Degenerate case: mod by 0 is not well-defined, skip.
      return failure();
    }

    // For demonstration, we only handle positive moduli here.
    if (modulus.isNegative()) {
      return failure();
    }

    // 3. Compute bitwidth (k). We assume LHS and RHS have the same bitwidth
    // in comb dialect for this example. Usually, you'd double-check widths.
    unsigned bitWidth = modOp.getType().getIntOrFloatBitWidth();

    // 4. Compute mu = floor((1 << bitWidth) / M).
    //    We must ensure that (1 << bitWidth) does not overflow APInt.
    //    For simplicity, we do it by constructing an APInt of the required width.
    APInt oneShifted(bitWidth, 0, /*isSigned=*/false);
    oneShifted.setBit(bitWidth); // 2^bitWidth in APInt representation

    APInt mu = oneShifted.udiv(modulus);

    // 5. Build the Barrett sequence:
    //    q = (x * mu) >> bitWidth
    //    r = x - q * M
    //    if (r >= M) r = r - M
    //    Then r is our final result (x mod M).

    // Grab location info for new ops
    Location loc = modOp.getLoc();
    Value lhs = modOp.getLhs();

    // Create constant ops for mu and M
    auto constMu = rewriter.create<hw::ConstantOp>(
        loc, rewriter.getIntegerAttr(modOp.getType(), mu));
    auto constM = rewriter.create<hw::ConstantOp>(
        loc, rewriter.getIntegerAttr(modOp.getType(), modulus));

    // q = (x * mu) >> bitWidth
    // We'll use comb::MulOp and comb::ShrUOp (logical shift right)
    Value mulVal = rewriter.create<comb::MulOp>(loc, lhs, constMu);
    auto shiftAmount = rewriter.create<hw::ConstantOp>(
        loc, rewriter.getIntegerAttr(modOp.getType(), bitWidth));
    Value q = rewriter.create<comb::ShrUOp>(loc, mulVal, shiftAmount);

    // r = x - q * M
    Value qTimesM = rewriter.create<comb::MulOp>(loc, q, constM);
    Value r = rewriter.create<comb::SubOp>(loc, lhs, qTimesM);

    // Conditionally subtract M if r >= M
    // For simplicity, do:
    //   cmp = r >= M
    //   final = select cmp (r - M) r
    Value cmp = rewriter.create<comb::ICmpOp>(
        loc, comb::ICmpPredicate::ge, r, constM);

    Value rMinusM = rewriter.create<comb::SubOp>(loc, r, constM);
    Value finalResult = rewriter.create<comb::MuxOp>(loc, cmp, rMinusM, r);

    // 6. Replace all uses of modOp with the newly computed finalResult
    rewriter.replaceOp(modOp, finalResult);

    return success();
  }
};

/// The actual pass that runs the pattern above.
struct BarrettReductionPassImpl
    : public BarrettReductionPassBase<BarrettReductionPassImpl> {
  void runOnOperation() override {
    // We will apply the pattern to all comb::ModOp in the current operation.
    RewritePatternSet patterns(&getContext());
    patterns.add<BarrettReductionPattern>(patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

/// Creates the pass declared in BarrettReductionPass.td
std::unique_ptr<mlir::OperationPass> createBarrettReductionPass() {
  return std::make_unique<BarrettReductionPassImpl>();
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// This registerPass() call will allow us to run:
///   mlir-opt --barrett-reduction ...
/// if properly linked into an MLIR-based tool.
void mlir::registerBarrettReductionPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createBarrettReductionPass();
  });
}

}