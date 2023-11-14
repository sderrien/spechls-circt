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

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"

#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOpsDialect.cpp.inc"
#include "InitAllPasses.h"
#include "InitAllTranslations.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // TODO: Register SpecHLS passes here.
  SpecHLS::registerAllTranslations();
  SpecHLS::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<SpecHLS::SpecHLSDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<circt::comb::CombDialect>();
  registry.insert<circt::seq::SeqDialect>();
  registry.insert<circt::hw::HWDialect>();

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  //registerAllDialects(registry);
  
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "SpecHLS optimizer driver\n", registry));
}