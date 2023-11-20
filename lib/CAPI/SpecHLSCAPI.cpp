//===- Seq.cpp - C Interface for the Seq Dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
//#include "circt/Dialect/Comb/CombOps.h"
//#include "circt/Dialect/HW/HWOpInterfaces.h"
//#include "circt/Dialect/HW/HWOps.h"
//#include "mlir/Dialect/Arith/IR/Arith.h"
//#include "mlir/IR/PatternMatch.h"
//#include "mlir/Transforms/DialectConversion.h"
//#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/CAPI/Registration.h"
#include "SpecHLS/SpecHLSOpsDialect.cpp.inc"


extern "C" {
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SpecHLS, spechls, SpecHLS::SpecHLSDialect)
}

//void registerSeqPasses() { SpecHLS::registerTransformsPasses(); }

