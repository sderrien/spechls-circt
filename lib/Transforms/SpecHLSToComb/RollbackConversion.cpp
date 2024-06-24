#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Transforms/SpecHLSConversion.h"

LogicalResult
RollbackToCombConversion::matchAndRewrite(RollbackOp op,
                                        PatternRewriter &rewriter) const 
{
    // Get operator items
    auto loc = op.getLoc();
    auto stall = op.getStall();
    auto data = op.getData();
    auto depths = op.getDepths();
    auto idx = op.getIdx();

    // Create delays registers
    SmallVector<DelayOp> delays;
    SmallVector<Value> mux_operands;
    mux_operands.push_back(data);
    for (size_t i = 0; i < depths.size(); i++)
    {
        auto delay = rewriter.create<DelayOp>(
                loc, op.getType(), data, stall, data, cast<IntegerAttr>(depths[i]));
        mux_operands.push_back(delay.getResult());
        delays.push_back(delay);
    }

    if (delays.size() == 1)
    {
        auto mux = rewriter.create<MuxOp>(loc, op.getType(), idx, data, delays[0]);
        rewriter.replaceOp(op, mux);
        return success();
    }
    else
    {
        auto array = rewriter.create<ArrayCreateOp>(loc, mux_operands);
        llvm::errs() << array << "\n";
        auto result = rewriter.create<ArrayGetOp>(loc, array, idx);
        llvm::errs() << result << "\n";
        rewriter.replaceOp(op, result);
        return success();
    }
    return failure();
}
