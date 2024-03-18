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

def AutoPipelinePass : Pass<"auto-pipeline"> {
  let summary = "Schedule pipeline draft";
  let description = [{
    Perform operation scheduling on the

  }];
  let constructor = "SpecHLS::createAutoPipelinePass()";
  let dependentDialects = ["circt::ssp::SSPDialect","circt::hw::HWDialect, mlir::arith::ArithDialect, circt::comb::CombDialect,circt::seq::SeqDialect,SpecHLS::SpecHLSDialect"];
}


def EliminateRedundantGammaInputsPass : Pass<"eliminiate-redundant-gamma-inputs"> {
  let summary = "Simplify Gamma ops which have multiple identical inputs";

  let description = [{
   Simplify Gamma ops which have multiple identical inputs, e.g.

       %r = SpecHLS.gamma @x %s ?  %0, %1, %0, %1, %2 :i32";

     is simplified into a two input gamma op

       %_s = SpecHLS.lookUpTable [%s] :i1= {0,1,0,1}
       %r = SpecHLS.gamma @x %_s ?  %0, %1 :i32

   }];
  let constructor = "SpecHLS::createEliminateRedundantGammaInputsPass()";
  let dependentDialects = ["circt::hw::HWDialect, mlir::arith::ArithDialect, circt::comb::CombDialect,circt::seq::SeqDialect,SpecHLS::SpecHLSDialect"];
}

def FactorGammaInputsPass : Pass<"factor-gamma-inputs"> {
  let summary = "Merge connected gamma ops";
  let description = [{ }];
  let constructor = "SpecHLS::createFactorGammaInputsPass()";
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




def YosysOptimizer : Pass<"yosys-optimizer","mlir::ModuleOp"> {
  let summary = "Invoke Yosys to perform circuit optimization.";

  let constructor = "SpecHLS::createYosysOptimizerPass()";


  let options = [
    Option<"replace", "replace-with-optimized-module", "bool",
           "false", "Disable inlining modules that only feed test code">,
    Option<"abc", "use-abc", "bool",
           "false", "Enable ABC synthesis">
  ];

  let description = [{ }];

   let dependentDialects = ["circt::hw::HWDialect, circt::sv::SVDialect,SpecHLS::SpecHLSDialect, mlir::arith::ArithDialect, circt::comb::CombDialect,circt::seq::SeqDialect"];

}
/*
def ControlOptimizer : Pass<"control-optimizer","circt::hw::HWModuleOp"> {
  let summary = "Invoke Yosys to perform circuit optimization.";
  let constructor = "SpecHLS::createControlOptimizer()";
  let description = [{ }];
  let dependentDialects = ["circt::hw::HWDialect, circt::sv::SVDialect,SpecHLS::SpecHLSDialect, mlir::arith::ArithDialect, circt::comb::CombDialect,circt::seq::SeqDialect"];
}
*/

def InlineModules : Pass<"inline-opt-modules", "mlir::ModuleOp"> {
  let summary = "Inline Yosys optimized modules";
  let description = [{

  }];
   let dependentDialects = [
      "mlir::arith::ArithDialect",
      "circt::comb::CombDialect",
      "SpecHLS::SpecHLSDialect"
    ];
  let constructor = "SpecHLS::createInlineModulesPass()";
}


def ExportVitisHLS : Pass<"export-vitis-hls", "mlir::ModuleOp"> {
  let summary = "Export C code for Vitis HLS.";

  let constructor = "createExportVitisHLS()";

  let description = [{ }];

  let dependentDialects = ["circt::hw::HWDialect, circt::sv::SVDialect,SpecHLS::SpecHLSDialect, mlir::arith::ArithDialect, circt::comb::CombDialect,circt::seq::SeqDialect"];

}

def SpecHLSToComb : Pass<"lower-spechls-to-comb", "circt::hw::HWModuleOp"> {
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

def MobilityPass : Pass<"mobility", "mlir::ModuleOp"> {
  let summary = "Compute mobility informations.";
  let constructor = "SpecHLS::createMobilityPass()";
  let dependentDialects = ["circt::ssp::SSPDialect"];
}

def LocalMobilityPass : Pass<"local-mobility", "mlir::ModuleOp"> {
  let summary = "Compute local mobility informations.";
  let constructor = "SpecHLS::createLocalMobilityPass()";
  let dependentDialects = ["circt::ssp::SSPDialect"];
}

 def SchedulePass : Pass<"schedule", "mlir::ModuleOp"> {
   let summary = "A circt::schedule pass wrapper for gecos.";
   let description = [{
     A circt::schedule pass wrapper for gecos.
   }];

   let constructor = "SpecHLS::createSchedulePass()";
   let dependentDialects = ["circt::ssp::SSPDialect"];
 }


#endif // SPECHLS_TRANSFORMS_PASSES