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
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SSP/SSPDialect.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

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
std::unique_ptr<mlir::OperationPass<>>
createEliminateRedundantGammaInputsPass();
std::unique_ptr<mlir::OperationPass<>> createGenerateCPass();
std::unique_ptr<mlir::Pass> createYosysOptimizer();
std::unique_ptr<mlir::Pass> createSchedulePass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createInlineModulesPass();
std::unique_ptr<mlir::Pass> createMobilityPass();
std::unique_ptr<mlir::Pass> createLocalMobilityPass();
// std::unique_ptr<mlir::Pass> createControlOptimizer();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createGroupControlNodePass();

std::unique_ptr<mlir::OperationPass<circt::hw::HWModuleOp>>
createConvertSpecHLSToCombPass();
std::unique_ptr<mlir::OperationPass<SpecHLS::LookUpTableOp>>
createConvertSpecHLSLUTToCombPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExportVitisHLS();

void registerYosysOptimizerPipeline();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION

#define GEN_PASS_DEF_MERGEGAMMASPASS

#define GEN_PASS_DEF_MERGELOOKUPTABLESPASS

#define GEN_PASS_DEF_FACTORGAMMAINPUTSPASS
#define GEN_PASS_DECL_FACTORGAMMAINPUTSPASS

#define GEN_PASS_DEF_ELIMINATEREDUNDANTGAMMAINPUTSPASS
#define GEN_PASS_DECL_ELIMINATEREDUNDANTGAMMAINPUTSPASS

#define GEN_PASS_DEF_GROUPCONTROLNODEPASS
#define GEN_PASS_DEF_GENERATECPASS
#define GEN_PASS_DECL_YOSYSOPTIMIZER
#define GEN_PASS_DEF_YOSYSOPTIMIZER
#define GEN_PASS_DECL_GECOSSCHEDULEPASS
#define GEN_PASS_DEF_GECOSSCHEDULEPASS
#define GEN_PASS_DEF_YOSYSOPTIMIZER

#define GEN_PASS_DEF_INLINEMODULES
#define GEN_PASS_DECL_INLINEMODULES

#define GEN_PASS_DECL_EXPORTVITISHLS
#define GEN_PASS_DEF_EXPORTVITISHLS

#define GEN_PASS_DECL_SPECHLSLUTTOCOMB
#define GEN_PASS_DEF_SPECHLSLUTTOCOMB

#define GEN_PASS_DECL_SPECHLSTOCOMB
#define GEN_PASS_DEF_SPECHLSTOCOMB

#define GEN_PASS_DECL_MOBILITYPASS
#define GEN_PASS_DEF_MOBILITYPASS

#define GEN_PASS_DECL_LOCALMOBILITYPASS
#define GEN_PASS_DEF_LOCALMOBILITYPASS

#define GEN_PASS_DECL_SCHEDULEPASS
#define GEN_PASS_DEF_SCHEDULEPASS

#include "Transforms/Passes.h.inc"

} // namespace SpecHLS

#endif // CIRCT_TRANSFORMS_PASSES_H