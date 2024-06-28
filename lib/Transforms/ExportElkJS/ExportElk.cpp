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

namespace SpecHLS {

struct ExportElkPass
    : public impl::ExportVitisHLSBase<ExportElkPass> {

public:
  void runOnOperation() {
    auto *ctx = &getContext();
    auto module = this->getOperation();

    llvm::outs() << "In ElkJS export pass ";
    std::ostringstream out;

    for (auto hwop : module.getOps<circt::hw::HWModuleOp>()) {
      auto moduleName = hwop.getNameAttr().str();
      for (Operation* op : hwop.getBodyBlock()->getOperations()) {

         llvm::outs() <<  "node "<< op.getName() << " {\n"                            ;
         for (auto arg : op.getOperands()) {
           llvm::outs() <<  "    port " << arg.getName() << " { ^port.side: WEST }\n"            ;
         }
         llvm::outs() <<  "    node n1 {\n"                               ;
         llvm::outs() <<  "        portConstraints: FIXED_SIDE\n"         ;
         llvm::outs() <<  "        port p1 { ^port.side: WEST }\n"        ;
         llvm::outs() <<  "        port p2 { ^port.side: EAST }\n"        ;
         llvm::outs() <<  "    }\n"                                       ;
         llvm::outs() <<  "}"                                             ;
      }
    }
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExportVitisHLS() {
  return std::make_unique<ExportElkPass>();
}
} // namespace SpecHLS