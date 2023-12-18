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
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/Optional.h"
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

namespace mlir {
namespace gpu {

template <typename T>
class OpPassBase;

class CustomCEmitter {
public:
  void translateModule(mlir::ModuleOp module, llvm::raw_ostream &output);

private:
  // TODO(limo1996): reset all 3 per new function (Global variables are not
  // expected for now I guess)
  int var_counter = 0;

  std::vector<std::string> funcDeclarations;
  std::map<Operation *, std::string> opToName;
  std::map<Value *, std::string> argToName;
  std::set<std::string> includes;
  std::set<std::string> launchFuncDeviceVars;

  bool isKernelFunc(Operation *op) { return gpu::GPUDialect::isKernel(op); }

  bool isKernelModule(Operation *op) {
    UnitAttr isKernelModuleAttr =
        op->getAttrOfType<UnitAttr>(gpu::GPUDialect::getKernelModuleAttrName());
    return static_cast<bool>(isKernelModuleAttr);
  }

  std::string appendModulePrefix(std::string fname, std::string moduleName) {
    return moduleName + "_" + fname + "_mlir";
  }

  std::string getFuncName(Operation *op) {
    std::string moduleName;
    auto module = op->getParentOfType<ModuleOp>();
    if (module.getName())
      moduleName = module.getName().getValue();

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

  inline std::string dim2str(mlir::gpu::KernelDim3 dim) {
    return value2str(dim.x) + "," + value2str(dim.y) + "," + value2str(dim.z);
  }

  inline bool isHostFunc(Operation *op) {
    bool has = false;
    op->walk([&](LaunchFuncOp op) {
      has = true;
      return WalkResult::interrupt();
    });
    return has;
  }

  inline std::string getShapeAt(AllocOp *op, unsigned int i) {
    auto shapeAt = static_cast<MemRefType>(op->getType()).getShape()[i];
    return shapeAt == -1 ? value2str(op->getOperand(i))
                         : std::to_string(shapeAt);
  }

  template <typename T>
  std::string bin2str(T op, char operand) {
    return this->value2str(op.lhs()) + " " + operand + " " +
           this->value2str(op.rhs());
  }

  inline bool isStaticArray(MemRefType memref) {
    return memref.hasStaticShape();
  }

  inline std::string getPointer(MemRefType memref, int less = 0, int min = 1) {
    return type2str(memref.getElementType()) +
           std::string(std::max((int)memref.getShape().size() - less, min),
                       '*');
  }

  std::string type2str(Type type);
  std::string attr2str(Attribute attr);
  template <typename T>
  std::string op2str(T *v);
  std::string value2str(Value *v);

  void processModuleOrFunc(Operation *op, std::ostringstream &out);

  bool printKernelVar(Operation *op, std::ostringstream &out,
                      std::string indent);

  void printLaunchFuncOp(gpu::LaunchFuncOp *op, std::ostringstream &out,
                         std::string indent);
  void printConstantOp(ConstantOp *op, std::ostringstream &out,
                       std::string indent);
  void printSIToFPOp(SIToFPOp *fpOp, std::ostringstream &out,
                     std::string indent);
  void printMemRefCastOp(MemRefCastOp *memCastOp, std::ostringstream &out,
                         std::string indent);
  void printCmpFOp(CmpFOp *cmpOp, std::ostringstream &out, std::string indent);
  void printSelectOp(SelectOp *selOp, std::ostringstream &out,
                     std::string indent);
  void printSqrtfOp(stencil::SqrtfOp *sqrtOp, std::ostringstream &out,
                    std::string indent);
  void printFabsOp(stencil::FabsOp *fabsOp, std::ostringstream &out,
                   std::string indent);
  void printExpOp(stencil::ExpOp *expOp, std::ostringstream &out,
                  std::string indent);
  void printPowOp(stencil::PowOp *powOp, std::ostringstream &out,
                  std::string indent);
  void printReturnOp(Operation *op, std::ostringstream &out,
                     std::string indent);
  void printLoadOp(LoadOp *op, std::ostringstream &out, std::string indent);
  void printStoreOp(StoreOp *op, std::ostringstream &out, std::string indent);
  void printAllocOp(AllocOp *op, std::ostringstream &out, std::string indent);
  void printDeallocOp(DeallocOp *op, std::ostringstream &out,
                      std::string indent);
  void printIfOp(loop::IfOp ifOp, std::ostringstream &out, std::string indent);
  void printForLoop(loop::ForOp *op, std::ostringstream &out,
                    std::string indent);
  bool printArithmetics(Operation *op, std::ostringstream &out,
                        std::string indent);
  void printCallOp(CallOp *callOp, std::ostringstream &out, std::string indent);
  void printDefaultOp(Operation *op, std::ostringstream &out,
                      std::string indent);
  void printOperation(Operation *op, std::ostringstream &out,
                      std::string indent);
  void printFunction(Operation *op, std::ostringstream &out,
                     std::string indent);
};
} // namespace gpu
} // namespace mlir

#endif // MLIR_TARGET_CUDAC_CustomCEmitter_H
