//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Transforms/ directory.
//
//===----------------------------------------------------------------------===//

#ifndef SPECHLS_TRANSFORMS_PASSES
#define SPECHLS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def MergeGammasPass : Pass<"merge-gammas"> {
  let summary = "Merge connected gamma ops";
  let description = [{
    Merge consecutive `gammas` ops

  }];
  let constructor = "SpecHLS::createMergeGammasPass()";
  let dependentDialects = ["circt::hw::HWDialect, mlir::arith::ArithDialect, circt::comb::CombDialect,circt::seq::SeqDialect,SpecHLS::SpecHLSDialect"];
}




def MergeLookUpTablesPass : Pass<"merge-luts"> {
  let summary = "Merge connected LUTs ops";
  let description = [{
    Merge consecutive `LUT` ops

  }];
  let constructor = "SpecHLS::createMergeLookUpTablesPass()";
  let dependentDialects = ["circt::hw::HWDialect, mlir::arith::ArithDialect, circt::comb::CombDialect,circt::seq::SeqDialect,SpecHLS::SpecHLSDialect"];
}

#endif // SPECHLS_TRANSFORMS_PASSES
