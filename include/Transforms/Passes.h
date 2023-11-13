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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <limits>

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "SpecHLS/SpecHLSOps.h"
#include "SpecHLS/SpecHLSDialect.h"

namespace SpecHLS {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<>> createMergeGammasPass() ;


//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DEF_MERGEGAMMASPASS

#include "Transforms/Passes.h.inc"

} // namespace circt

#endif // CIRCT_TRANSFORMS_PASSES_H
