//===-- Passes.td - Conversion pass definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for all dialect conversions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_PASSES_TD
#define CIRCT_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"
include "SpecHLS/SpecHLSDialect.td"

//===----------------------------------------------------------------------===//
// SpecHLSToComb
//===----------------------------------------------------------------------===//

def SpecHLSToComb : Pass<"lower-spechls-to-comb", "mlir::ModuleOp"> {
  let summary = "Lower SPECHLS to comb";
  let constructor = "createConvertSpecHLSToCombPass()";
  let dependentDialects = [
    "circt::hw::HWDialect,circt::comb::CombDialect,SpecHLS::SpecHLSDialect"
  ];
}

def SpecHLSLUTToComb : Pass<"lower-SpecHLSLUT-to-comb", "SpecHLS::LookUpTableOp"> {
  let summary = "Lower SPECHLS to comb";
  let constructor = "createConvertSpecHLSLUTToCombPass()";
  let dependentDialects = [
    "circt::hw::HWDialect,circt::comb::CombDialect,SpecHLS::SpecHLSDialect"
  ];
}


#endif // CIRCT_CONVERSION_PASSES_TD
