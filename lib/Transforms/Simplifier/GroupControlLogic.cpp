//===- GroupControlNode.cpp - SV Simulation Extraction Pass --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass extracts simulation constructs to sunewHTaskules.
// It will take simulation operations, write, finish, assert, assume, and cover
// and extract them and the dataflow into them into a separate module.  This
// module is then instantiated in the original module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"

#include "Common/OutliningUtils.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SetVector.h"

#include <set>

using namespace mlir;
using namespace circt;
using namespace circt::comb;

//===----------------------------------------------------------------------===//
// StubExternalModules Pass
//===----------------------------------------------------------------------===//

struct GroupControlNodePass
        : public SpecHLS::impl::GroupControlNodePassBase<GroupControlNodePass> {
    int max_bitwidth = 4;

    function_ref<bool(Operation *)> opfilter = [&](Operation *op) {
        return false;
    };

    GroupControlNodePass() {}

public:
    void runOnOperation() override;

private:
    bool checkResultIntBitwidth(Operation *op);

    virtual bool filter(Operation *op);

    Operation *gammaFilter(SpecHLS::GammaOp gamma);

    mlir::LogicalResult sliceGammaControlOps(Operation *op, Block *body);

    mlir::LogicalResult groupAsHTask(SpecHLS::GammaOp gamma, StringRef name, SetVector<Operation *> slice, SetVector<Value *> inputs, SetVector<Value *> outputs);

    mlir::LogicalResult groupAsHWModule(SpecHLS::GammaOp gamma, StringRef name, SetVector<Operation *> slice, SetVector<Value *> inputs, SetVector<Value *> outputs);
};


void GroupControlNodePass::runOnOperation() {
    auto top = getOperation();
    auto *topLevelModule = top.getBody();
    if (topLevelModule) {
        for (auto &op: llvm::make_early_inc_range(topLevelModule->getOperations())) {
            TypeSwitch<Operation *>(&op)
                    .Case<SpecHLS::HKernelOp>([&](auto op) {
                        sliceGammaControlOps(op, op.getBody());
                    })
                    .Case<hw::HWModuleOp>([&](auto op) {
                        sliceGammaControlOps(op, op.getBodyBlock());
                    });
        }
    }
}

mlir::LogicalResult GroupControlNodePass::sliceGammaControlOps(Operation *op, Block *body) {
    auto gammas = body->getOps<SpecHLS::GammaOp>();
    auto gammaId = 0;

    for (auto gamma: gammas) {
        llvm::outs() << " - analyzing gamma   " << gamma << "\n";
        auto controlOp = gammaFilter(gamma);
        if (controlOp) {
            /*
             * builds the slice starting from controlOp
             */
            SetVector<Operation *> slice = {};
            SetVector<Value *> inputs = {};
            SetVector<Value *> outputs = {};
            for (auto res: controlOp->getResults()) {
                outputs.insert(&res);
            }
            getBackwardSlice(*controlOp, slice, inputs, opfilter);

            /*
             * building new HTAsk from sliced ops
             */
            auto newName = "ctrl_" + std::to_string(gammaId++);
            if (target == "htask") {
                groupAsHTask(gamma, newName, slice, inputs, outputs);
            } else {
                groupAsHWModule(gamma, newName, slice, inputs, outputs);
            }
            mlir::verify(op, true);
        }
    }
}

bool GroupControlNodePass::checkResultIntBitwidth(Operation *op) {
    return (op->getResult(0).getType().getIntOrFloatBitWidth()) < max_bitwidth;
}

bool GroupControlNodePass::filter(Operation *op) {
    return TypeSwitch<Operation *, bool>(op)
            .Case<AddOp>([&](auto op) { return checkResultIntBitwidth(op); })
            .Case<SubOp>([&](auto op) { return checkResultIntBitwidth(op); })
            .Case<ICmpOp>([&](auto op) { return checkResultIntBitwidth(op); })
            .Case<circt::comb::AndOp>([&](auto op) { return true; })
            .Case<circt::comb::OrOp>([&](auto op) { return true; })
            .Case<circt::comb::XorOp>([&](auto op) { return true; })
            .Case<circt::comb::ExtractOp>([&](auto op) { return true; })
            .Case<circt::comb::ConcatOp>([&](auto op) { return true; })
            .Case<circt::hw::ConstantOp>([&](auto op) { return true; })
            .Case<circt::comb::MuxOp>([&](auto op) { return true; })
            .Case<circt::comb::TruthTableOp>([&](auto op) { return true; })
            .Case<SpecHLS::CastOp>([&](auto op) { return true; })
            .Case<SpecHLS::LookUpTableOp>([&](auto op) { return true; })
            .Default([&](auto op) { return false; });
}

Operation *GroupControlNodePass::gammaFilter(SpecHLS::GammaOp gamma) {
    if (!(gamma->getNumOperands() > 0)) {
        auto controlValue = gamma->getOperand(0);
        auto controlOp = controlValue.getDefiningOp();
        if (filter(controlOp)) return controlOp;
    }
    return NULL;

}

mlir::LogicalResult
GroupControlNodePass::groupAsHTask(SpecHLS::GammaOp gamma, StringRef name, SetVector<Operation *> slice, SetVector<Value *> inputs, SetVector<Value *> outputs) {
    auto builder = OpBuilder(gamma.getContext());
    builder.setInsertionPoint(gamma);

    auto newHTask = outlineSliceAsHTask(gamma->getParentOp(), slice, inputs, outputs, name);

    if (newHTask) {
        gamma.setOperand(0, newHTask.getResult(0));
        newHTask->setAttr(builder.getStringAttr("#pragma"), builder.getStringAttr("CONTROL_NODE"));
        return LogicalResult::success();
    }
    return LogicalResult::failure();
}

mlir::LogicalResult
GroupControlNodePass::groupAsHWModule(SpecHLS::GammaOp gamma, StringRef name, SetVector<Operation *> slice, SetVector<Value *> inputs, SetVector<Value *> outputs) {
    auto builder = OpBuilder(gamma.getContext());
    builder.setInsertionPoint(gamma);
    auto newHwModule = outlineSliceAsHwModule(gamma->getParentOp(), slice, inputs, outputs, name);

    if (newHwModule) {
        SmallVector<Value, 8> operands;
        for (auto i: inputs) {
            operands.push_back(*i);
        }
        auto inst = builder.create<hw::InstanceOp>(
                gamma->getLoc(), newHwModule,
                builder.getStringAttr(newHwModule.getName()), operands,
                ArrayAttr());

        gamma.setOperand(0, inst.getResult(0));

        newHwModule->setAttr(builder.getStringAttr("#pragma"),
                             builder.getStringAttr("CONTROL_NODE"));
        return LogicalResult::success();
    }
    return LogicalResult::failure();


}



namespace SpecHLS {
    std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
    createGroupControlNodePass() {
        // llvm::outs() << "GroupControlNodeImplPass created " << "\n";
        return std::make_unique<GroupControlNodePass>();
    }
} // namespace SpecHLS
