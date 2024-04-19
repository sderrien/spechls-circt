//===- ExportVitisHLS.cpp - Arith-to-comb mapping pass ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the ExportVitisHLS pass.
//
//===----------------------------------------------------------------------===//
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/Pass/Pass.h"

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"

#include "Transforms/Passes.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "Transforms/VitisExport/CFileContent.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

using namespace std;

using namespace mlir;
using namespace circt;
using namespace SpecHLS;
using namespace circt::hw;

void printOperation(CFileContent *p, Operation *op);
void printHWModule(CFileContent *p, HWModuleOp op);
// void printTestbench(CFileContent *p, HWModuleOp op);

namespace SpecHLS {

struct ExportVitisHLSPass
    : public impl::ExportVitisHLSBase<ExportVitisHLSPass> {

public:
  void runOnOperation() {
    auto *ctx = &getContext();
    auto module = this->getOperation();

    llvm::outs() << "In  Vitis-HLS Export C code pass ";
    std::ostringstream out;

    for (auto hwop : module.getOps<circt::hw::HWModuleOp>()) {
      llvm::outs() << "Exporting Vitis-HLS C code for " << hwop.getName()
                   << "\n";
      auto moduleName = hwop.getNameAttr().str();
      auto file = CFileContent("./", moduleName);
      printHWModule(&file, hwop);
      // printTestbench(&file,hwop);
      file.save();
    }
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExportVitisHLS() {
  return std::make_unique<ExportVitisHLSPass>();
}
} // namespace SpecHLS