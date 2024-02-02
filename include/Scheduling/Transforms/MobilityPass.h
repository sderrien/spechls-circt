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

#ifndef SPECHLS_MOBILITY_PASS_H
#define SPECHLS_MOBILITY_PASS_H

#include "circt/Dialect/SSP/SSPOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace SpecHLS {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
[[maybe_unused]] std::unique_ptr<mlir::Pass> createMobilityPass();

#define GEN_PASS_DECL_MOBILITYPASS
#define GEN_PASS_DEF_MOBILITYPASS

#include "Scheduling/Transforms/Passes.h.inc"

} // namespace SpecHLS

#endif // SPECHLS_MOBILITY_PASS_H