//===- Passes.h - Conversion Pass Construction and Registration -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This fle contains the declarations to register conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_PASSES_H
#define CIRCT_CONVERSION_PASSES_H

#include "circt/Conversion/AffineToLoopSchedule.h"
#include "circt/Conversion/ArcToLLVM.h"
#include "circt/Conversion/CFToHandshake.h"
#include "circt/Conversion/CalyxToFSM.h"
#include "circt/Conversion/CalyxToHW.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/ConvertToArcs.h"
#include "circt/Conversion/DCToHW.h"
#include "circt/Conversion/ExportChiselInterface.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/FIRRTLToHW.h"
#include "circt/Conversion/FSMToSV.h"
#include "circt/Conversion/HWArithToHW.h"
#include "circt/Conversion/HWToLLHD.h"
#include "circt/Conversion/HWToLLVM.h"
#include "circt/Conversion/HWToSystemC.h"
#include "circt/Conversion/HandshakeToDC.h"
#include "circt/Conversion/HandshakeToHW.h"
#include "circt/Conversion/LLHDToLLVM.h"
#include "circt/Conversion/LoopScheduleToCalyx.h"
#include "circt/Conversion/MooreToCore.h"
#include "circt/Conversion/PipelineToHW.h"
#include "circt/Conversion/SCFToCalyx.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "SpecHLS/SpecHLSDialect.h"

#include "SpecHLS/SpecHLSDialect.h"

namespace SpecHLS {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertSpecHLSToCombPass();
std::unique_ptr<mlir::OperationPass<SpecHLS::LookUpTableOp>> createConvertSpecHLSLUTToCombPass();

// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DEF_MERGEGAMMASPASS
#define GEN_PASS_DEF_SPECHLSTOCOMB
#define GEN_PASS_DEF_SPECHLSLUTTOCOMB
#define GEN_PASS_DECL_MERGEGAMMASPASS
#define GEN_PASS_DECL_SPECHLSTOCOMB
#define GEN_PASS_DECL_SPECHLSLUTTOCOMB

#include "Conversion/Passes.h.inc"

} // namespace SpecHLS

#endif // CIRCT_CONVERSION_PASSES_H
