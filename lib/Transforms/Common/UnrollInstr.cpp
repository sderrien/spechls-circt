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

// Setup the HWModuleOp to be unrolled
bool setupHWModule(hw::HWModuleOp op, PatternRewriter &rewriter, MuOp &first_mu)
{
    // Check module validity
    if (op.getNumPorts() < 2)
    {
        return false;
    }
    if (!hasPragmaContaining(op, "UNROLL_NODE"))
    {
        return false;
    }
    
    // Remove all MuOp
    op.walk([&](MuOp mu) {
            first_mu = mu;
            // Replace the uses of the output of the mu by an op's argument
            std::pair<StringAttr, BlockArgument> arg = op.appendInput(mu.getName(), 
                    mu.getResult().getType());
            rewriter.replaceOp(mu, arg.second);
            });
    
    // Update pragma
    setPragmaAttr(op, rewriter.getStringAttr("INLINE"));
    return true;
}

namespace SpecHLS {

    struct UnrollInstrPattern : OpRewritePattern<hw::HWModuleOp>
    {
        llvm::ArrayRef<unsigned int> instrs;
        using OpRewritePattern<hw::HWModuleOp>::OpRewritePattern;

        // Constructor to save pass arguments
        UnrollInstrPattern(MLIRContext *ctx, const llvm::ArrayRef<unsigned int> intrs) : 
            OpRewritePattern<hw::HWModuleOp>(ctx)
        {
            this->instrs = intrs;
        }

        LogicalResult matchAndRewrite(hw::HWModuleOp top,
                                PatternRewriter &rewriter) const override
        {
            MuOp first_mu;
            if (!setupHWModule(top, rewriter, first_mu))
            {
                return failure();
            }

            // Create the new module with all instances
            SmallVector<hw::PortInfo> ports;

            StringAttr module_name = rewriter.getStringAttr(
                    top.getName().str() + std::string("_unroll")); 
            hw::HWModuleOp unrolled_module = rewriter.create<hw::HWModuleOp>(rewriter.getUnknownLoc(),
                    module_name, ports
                    );
            Block *body = unrolled_module.getBodyBlock();
            ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(unrolled_module.getLoc(), body);
            
            auto initial = builder.create<InitOp>(first_mu.getInit().getType(), "initial_value");
            setPragmaAttr(initial, rewriter.getStringAttr("MU_INITAL"));
            
            // Create a constOp for each instruction
            SmallVector<hw::ConstantOp> cons;
            for (size_t i = 0; i < instrs.size(); i++)
            {
                cons.push_back(builder.create<hw::ConstantOp>(builder.getI32IntegerAttr(instrs[i])));
            }

            // Constant to drive the DelayOp between each instance 
            hw::ConstantOp enable = builder.create<hw::ConstantOp>(builder.getBoolAttr(1));

            // Add the first instance with the initial value
            SmallVector<Value> inputs;
            inputs.push_back(cons[0]);
            inputs.push_back(initial);
            hw::InstanceOp inst = builder.create<hw::InstanceOp>(
                    top, top.getName(), inputs
                    );

            DelayOp delta = builder.create<DelayOp>(
                    inst.getType(0),
                    inst.getResult(0),
                    enable,
                    inst.getResult(0),
                    builder.getI32IntegerAttr(1)
                    );

            // Add the other instances
            for (size_t i = 1; i < cons.size(); i++)
            {
                inputs.clear();
                hw::ConstantOp op = cons[i];
                inputs.push_back(op);
                inputs.push_back(delta.getResult());
                inst = builder.create<hw::InstanceOp>(
                        top, top.getName(), inputs
                        );
                delta = builder.create<DelayOp>(
                        inst.getType(0),
                        inst.getResult(0),
                        enable,
                        inst.getResult(0),
                        builder.getI32IntegerAttr(1)
                        );
            }

            // Plug the last delay into the output of the module
            unrolled_module.appendOutput("out", delta.getResult());

            return success();
        }

    };


    struct UnrollInstrPass : public impl::UnrollInstrPassBase<UnrollInstrPass>
    {

        public:
            void runOnOperation() override
            {
                auto *ctx = &getContext();
                auto pm = PassManager::on<ModuleOp>(ctx);

                RewritePatternSet patterns(ctx);
                patterns.add<UnrollInstrPattern>(ctx, *instrs);

                if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                std::move(patterns))))
                {
                    llvm::errs() << "failed\n";
                    signalPassFailure();
                }

                OpPassManager dynamicPM("builtin.module");
                dynamicPM.addPass(createInlineModulesPass());
                dynamicPM.addPass(createCanonicalizerPass());
                if (failed(runPipeline(dynamicPM, getOperation())))
                {
                    signalPassFailure();
                }
            }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createUnrollInstrPass() 
    {
        return std::make_unique<UnrollInstrPass>();
    }

} // namespace SpecHLS

