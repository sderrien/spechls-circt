//===- CustomCEmitter.h - MLIR to LLVM conversion ------------*- C++ -*-===//
//
// Copyright 2019 Jakub Lichman
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements the translation between an MLIR GPU dialect module and
// the corresponding CUDA C file.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_CUDAC_CustomCEmitter_H
#define MLIR_TARGET_CUDAC_CustomCEmitter_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

//#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

namespace SpecHLS {
using namespace mlir;

template <typename T>
class OpPassBase;

class CustomCEmitter {
public:
  void translateModule(mlir::ModuleOp module, llvm::raw_ostream &output);

private:
  // TODO(limo1996): reset all 3 per new function (Global variables are not
  // expected for now I guess)
  int var_counter = 0;

  std::map<Operation *, std::string> opToName;
  std::map<Value *, std::string> argToName;

  std::vector<std::string> globals;

  std::vector<std::string> declarations;

  std::vector<std::string> init;
  std::vector<std::string> syncUpdate;
  std::vector<std::string> combUpdate;
  std::vector<std::string> exit;

  std::set<std::string> includes;

  std::set<std::string> setup;

  std::set<std::string> launchFuncDeviceVars;


  std::string appendModulePrefix(std::string fname, std::string moduleName) {
    return moduleName + "_" + fname + "_mlir";
  }

  std::string getFuncName(Operation *op) {
    std::string moduleName;
    auto module = op->getParentOfType<ModuleOp>();
    if (module.getName())
      moduleName = module.getName()->str();

    if (auto attr =
            op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      return appendModulePrefix(attr.getValue().str(), moduleName);
    return appendModulePrefix(op->getName().getStringRef().str(), moduleName);
  }

  inline std::string getFreshVar(Operation *op) {
    std::string fresh_var = "i" + std::to_string(var_counter++);
    if (op != nullptr)
      opToName[op] = fresh_var;
    return fresh_var;
  }

  inline std::string getFreshVar(Value *v) {
    std::string fresh_var = "i" + std::to_string(var_counter++);
    argToName[v] = fresh_var;
    return fresh_var;
  }






  inline bool isStaticArray(MemRefType memref) {
    return memref.hasStaticShape();
  }

  inline std::string getPointer(MemRefType memref, int less = 0, int min = 1) {
    return type2str(memref.getElementType()) +
           std::string(std::max((int)memref.getShape().size() - less, min),
                       '*');
  }

  std::string getId(Operation* op);
  std::string type2str(Type type);
  std::string attr2str(Attribute attr);
  template <typename T>
  std::string op2str(T *v);
  std::string value2str(Value *v);
  std::string bin2str(Value lhs, char operand, Value rhs);

  void processModuleOrFunc(Operation *op, std::ostringstream &out);
  void printOperation(Operation *op);

  std::string valueList(OperandRange range, std::string sep);
  std::string argList(OperandRange range, std::string sep);

  void printBinaryOp(Operation* args, std::string op) ;
  void printUnaryOp(Operation* args, std::string op) ;

  void printAlpha(SpecHLS::AlphaOp op);
  void printDelay(SpecHLS::DelayOp op);
  void printLUT(SpecHLS::LookUpTableOp op);
  void printGamma(SpecHLS::GammaOp op);
  void printArrayRead(SpecHLS::ArrayReadOp op);
  void printRollback(SpecHLS::RollbackOp op);
  void printRollback(SpecHLS::InitOp op);
  void printEncoder(SpecHLS::EncoderOp op);
  void printDecoder(SpecHLS::DecoderOp op);
  void printMu(SpecHLS::MuOp op);
  void printPrint(SpecHLS::PrintOp op);
  void printExit(SpecHLS::ExitOp op);

  void printMux(circt::comb::MuxOp op);
  void printCompare(circt::comb::ICmpOp op);
  void printExtract(circt::comb::ExtractOp op);
  void printConcat(circt::comb::ConcatOp op);
  void printCast(mlir::UnrealizedConversionCastOp op);

  void printInstance(circt::hw::InstanceOp op);
  void printConstant(circt::hw::ConstantOp op);
  void printOutput(circt::hw::OutputOp op) ;
  void printHWModule(circt::hw::HWModuleOp op) ;
  void printExtern(circt::hw::HWModuleExternOp op) ;

  void printDefault(Operation *op);


};
} // namespace gpu

#endif // MLIR_TARGET_CUDAC_CustomCEmitter_H
