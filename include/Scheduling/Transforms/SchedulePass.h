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

#ifndef SPECHLS_SCHEDULING_PASSES_H
#define SPECHLS_SCHEDULING_PASSES_H

#include "circt/Dialect/SSP/SSPOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace Scheduling {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<>> createGecosSchedulePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DEF_GECOSSCHEDULEPASS

// #include "Scheduling/Transforms/Scheduling.h.inc"

} // namespace Scheduling

#endif // SPECHLS_SCHEDULING_PASSES_H