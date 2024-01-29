//===- spechls-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOpsDialect.cpp.inc"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"

#include "SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "InitAllPasses.h"
#include "InitAllTranslations.h"

#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Interop/InteropDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Pipeline/PipelineDialect.h"
#include "circt/Dialect/SSP/SSPDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "mlir/IR/Dialect.h"

int main(int argc, char **argv) {

  mlir::DialectRegistry registry;

  //registerAllDialects(registry);
  registry.insert<SpecHLS::SpecHLSDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  circt::hwarith::HWArithDialect, circt::comb::CombDialect,
                  circt::seq::SeqDialect, circt::hw::HWDialect, circt::sv::SVDialect,
                  circt::ssp::SSPDialect,
                  //      circt::firrtl::FIRRTLDialect,
                  circt::fsm::FSMDialect>();

  mlir::registerAllPasses();
  // TODO: Register SpecHLS passes here.
  SpecHLS::registerAllTranslations();
  SpecHLS::registerAllPasses();

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SpecHLS optimizer driver\n", registry));
}