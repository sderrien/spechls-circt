//===- UnrollInstr.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the UnrollInstr pass
//
//===----------------------------------------------------------------------===//

#include <map>
#include <stack>

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


using namespace mlir;
using namespace circt;
using namespace SpecHLS;

void topologicalSort(Operation * op, std::stack<Value> &stack, DenseMap<Operation *, bool> &visited);

namespace SpecHLS {

    struct LongestPathPattern : OpRewritePattern<hw::HWModuleOp>
    {

        LongestPathPattern(MLIRContext * ctx) :
            OpRewritePattern<hw::HWModuleOp>(ctx){}

        LogicalResult matchAndRewrite(hw::HWModuleOp top,
                                PatternRewriter &rewriter) const override
        {
            auto inits = top.getOps<InitOp>();
            InitOp starting_point;
            bool find = false;
            for (InitOp init : inits)
            {
                if (hasPragmaContaining(init, "MU_INITAL"))
                {
                    starting_point = init;
                    find = true;
                    break;
                }
            }
            if (!find)
            {
                return failure();
            }
            DenseMap<Operation *, int> dists;
            DenseMap<Operation *, bool> visited;
            std::stack<Value> stack;
            for (Operation &op : top.getBody().front().getOperations())
            {
                dists[&op] = INT_MIN;
                visited[&op] = false;
            }
            for (Operation &op : top.getBody().front().getOperations())
            {
                if (!visited[&op])
                {
                    topologicalSort(&op, stack, visited);
                }
            }
            dists[starting_point] = 0;
            while (!stack.empty())
            {
                Value v = stack.top();
                stack.pop();

                // Get the delay of the Operation
                int delay = 0;
                DelayOp delta = v.getDefiningOp<DelayOp>();
                if (delta)
                {
                    delay = delta.getDepth();
                }

                // Get the attribute
                Operation * op = v.getDefiningOp();
                auto dist = dists[op];

                // Check attribute's Value
                if (dist == INT_MIN)
                {
                    continue;
                }
            
                // Save all uses & update all uses' value
                for (Operation * use : v.getUsers())
                {
                    auto use_d = dists[use]; 
                    if (use_d < (dist + delay))
                    {
                        dists[use] = dist + delay;
                    }
                }
            }
            Operation * output = top.getBody().front().getTerminator();
            output->setAttr("#dist", rewriter.getI32IntegerAttr(dists[output]));
            removePragmaAttr(starting_point, "MU_INITAL");
            return success();
        }

    };


    struct LongestPathPass : public impl::LongestPathPassBase<LongestPathPass>
    {

        public:
            void runOnOperation() override
            {
                auto *ctx = &getContext();

                RewritePatternSet patterns(ctx);
                patterns.add<LongestPathPattern>(ctx);

                if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                std::move(patterns))))
                {
                    llvm::errs() << "failed\n";
                    signalPassFailure();
                }
            }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createLongestPathPass() 
    {
        return std::make_unique<LongestPathPass>();
    }

} // namespace SpecHLS


void topologicalSort(Operation * op, std::stack<Value> &stack, DenseMap<Operation *, bool> &visited)
{
    visited[op] = true;

    for ( Operation * use : op->getUsers())
    {
        if (!visited[use])
        {
            topologicalSort(use, stack, visited);
        }
    }
    for (size_t i = 0; i < op->getNumResults(); i++)
    {
        stack.push(op->getResult(i));
    }
}
