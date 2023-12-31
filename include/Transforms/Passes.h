//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for CIRCT transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef SPECHLS_TRANSFORMS_PASSES_H
#define SPECHLS_TRANSFORMS_PASSES_H

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <limits>

#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace SpecHLS {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<>> createMergeGammasPass();
std::unique_ptr<mlir::OperationPass<>> createMergeLookUpTablesPass();
std::unique_ptr<mlir::OperationPass<>> createFactorGammaInputsPass();
std::unique_ptr<mlir::OperationPass<>> createGenerateCPass();
std::unique_ptr<mlir::Pass> createYosysOptimizer();
std::unique_ptr<mlir::Pass> createGecosSchedulePass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createGroupControlNodePass();

void registerYosysOptimizerPipeline();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DEF_MERGEGAMMASPASS
#define GEN_PASS_DEF_MERGELOOKUPTABLESPASS
#define GEN_PASS_DEF_FACTORGAMMAINPUTSPASS
#define GEN_PASS_DEF_GROUPCONTROLNODEPASS
#define GEN_PASS_DEF_GENERATECPASS
#define GEN_PASS_DEF_YOSYSOPTIMIZER
#define GEN_PASS_DECL_GECOSSCHEDULEPASS

#include "Transforms/Passes.h.inc"

} // namespace SpecHLS

#endif // CIRCT_TRANSFORMS_PASSES_H
