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
include "SpecHLS/SpecHLSDialect.td"

def MergeGammasPass : Pass<"merge-gammas"> {
  let summary = "Merge connected gamma ops";
  let description = [{
    Merge consecutive `gammas` ops

  }];
  let constructor = "SpecHLS::createMergeGammasPass()";
  let dependentDialects = ["circt::hw::HWDialect, mlir::arith::ArithDialect, circt::comb::CombDialect,circt::seq::SeqDialect,SpecHLS::SpecHLSDialect"];
}


def FactorGammaInputsPass : Pass<"factor-gamma-inputs"> {
  let summary = "Merge connected gamma ops";
  let description = [{ }];
  let constructor = "SpecHLS::createFactorGammaInputsPass()";
  let dependentDialects = ["circt::hw::HWDialect, mlir::arith::ArithDialect, circt::comb::CombDialect,circt::seq::SeqDialect,SpecHLS::SpecHLSDialect"];
}

def GenerateCPass : Pass<"generate-c"> {
  let summary = "Merge connected LUTs ops";
  let description = [{
  }];
  let constructor = "SpecHLS::createGenerateCPass()";
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



def GroupControlNodePass : Pass<"group-control","mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    statement.
  }];
  let constructor = "SpecHLS::createGroupControlNodePass()";
  let dependentDialects = ["circt::hw::HWDialect, mlir::arith::ArithDialect, circt::comb::CombDialect,circt::seq::SeqDialect,SpecHLS::SpecHLSDialect"];
/*  let options = [
    Option<"disableInstanceExtraction", "disable-instance-extraction", "bool",
           "false", "Disable extracting instances only that feed test code">,
    Option<"disableRegisterExtraction", "disable-register-extraction", "bool",
           "false", "Disable extracting registers only that feed test code">,
    Option<"disableModuleInlining", "disable-module-inlining", "bool",
           "false", "Disable inlining modules that only feed test code">
  ];*/
  let statistics = [
    Statistic<"numOpsExtracted", "num-ops-extracted", "Number of ops extracted">,
    Statistic<"numOpsErased", "num-ops-erased", "Number of ops erased">
  ];
}




def YosysOptimizer : Pass<"yosys-optimizer"> {
  let summary = "Invoke Yosys to perform circuit optimization.";

  let constructor = "SpecHLS::createYosysOptimizer()";
//  let constructor = "SpecHLS::createYosysOptimizer()";

  let dependentDialects = ["circt::hw::HWDialect, mlir::arith::ArithDialect, circt::comb::CombDialect,circt::seq::SeqDialect,SpecHLS::SpecHLSDialect"];

/*  let options = [
    Option<"yosysFilesPath", "yosys-files-path", "std::string",
           "\"?\"", "Yosys file path">,
    Option<"abcPath", "abc-path", "std::string",
           "\"?\"", "ABC file path">,
  ];
*/
  let description = [{
    This pass invokes Yosys to convert an arithmetic circuit to an optimized
    boolean circuit that uses the arith and comb dialects.

    Note that booleanization changes the function signature: multi-bit integers
    are transformed to a tensor of booleans, for example, an `i8` is converted
    to `tensor<8xi1>`.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "circt::comb::CombDialect",
    "SpecHLS::SpecHLSDialect"
  ];
}




#endif // SPECHLS_TRANSFORMS_PASSES