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
            if (top.getNumPorts() < 2)
            {
                return failure();
            }
            if (!hasPragmaContaining(top, "UNROLL_NODE"))
            {
                return failure();
            }
            // Change tag of the module 
            removePragmaAttr(top, "UNROLL_NODE"); 
            setPragmaAttr(top, rewriter.getStringAttr("INLINE"));
            // Find all SpecHLS.mu node in the HWModuleOp
            SmallVector<MuOp> list_mu;
            top.walk([&](MuOp mu) {
                    list_mu.push_back(mu);
                    std::pair<StringAttr, BlockArgument> arg = top.appendInput(mu.getName(), 
                            mu.getResult().getType());
                    // replace SpecHLS.mu node by module arguments
                    rewriter.replaceOp(mu, arg.second);
                    });

            // Create the new module with all instances
            SmallVector<hw::PortInfo> ports;

            StringAttr module_name = rewriter.getStringAttr(
                    top.getName().str() + std::string("_unroll")); 
            hw::HWModuleOp unrolled_module = rewriter.create<hw::HWModuleOp>(rewriter.getUnknownLoc(),
                    module_name, ports
                    );
            Block *body = unrolled_module.getBodyBlock();
            ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(unrolled_module.getLoc(), body);
            
            
            SmallVector<hw::ConstantOp> cons;
            for (size_t i = 0; i < instrs.size(); i++)
            {
                cons.push_back(builder.create<hw::ConstantOp>(builder.getI32IntegerAttr(instrs[i])));
            }

            auto initial = builder.create<InitOp>(list_mu[0].getInit().getType(), "initial_value");
            setPragmaAttr(initial, rewriter.getStringAttr("MU_INITAL"));

            ArrayRef<Value> inputs = {cons[0], initial};
            hw::InstanceOp inst = builder.create<hw::InstanceOp>(
                    top, top.getName(), inputs
                    );
            hw::ConstantOp enable = builder.create<hw::ConstantOp>(builder.getBoolAttr(1));
            DelayOp delta = builder.create<DelayOp>(
                    inst.getType(0), inst.getResult(0), enable, inst.getResult(0), 
                    builder.getI32IntegerAttr(1));

            for (size_t i = 1; i < cons.size(); i++)
            {
                hw::ConstantOp op = cons[i];
                SmallVector<Value> in;
                in.push_back(op);
                in.push_back(delta.getResult());
                inst = builder.create<hw::InstanceOp>(
                        top, top.getName(), ArrayRef(in)
                        );
                delta = builder.create<DelayOp>(
                        inst.getType(0), inst.getResult(0), enable, inst.getResult(0), 
                        builder.getI32IntegerAttr(1));
            }
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
