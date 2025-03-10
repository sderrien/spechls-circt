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

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
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

struct ExportCppPass : public impl::ExportVitisHLSBase<ExportCppPass> {

  llvm::DenseMap<Operation *, int> operationIDMap;
  int idCounter = 0;

public:
  std::string getCTypeString(mlir::Type mlirType) {
    return mlir::TypeSwitch<mlir::Type, std::string>(mlirType)
        .Case<mlir::IntegerType>([](mlir::IntegerType intType) -> std::string {
          return "ac_int<" + std::to_string(intType.getWidth()) + ", false>";
        })
        .Default([](mlir::Type) -> std::string { return "unknown_type"; });
  }

  // FIXME move into abstract codegen class
  std::string getConstantValueExpression(hw::ConstantOp constantOp) {
    auto attr = constantOp.getValue();
    auto constType = constantOp.getResult().getType();
    if (constType.isUnsignedInteger()) {
      return llvm::formatv("(ap_uint<{0}>({1}))",
                           constType.getIntOrFloatBitWidth(),
                           attr.getZExtValue());
    } else {
      return llvm::formatv("(ap_int<{0}>({1}))",
                           constType.getIntOrFloatBitWidth(),
                           attr.getSExtValue());
    }
  }
  std::string getCOperandString(mlir::Value operand) {

    if (auto definingOp = operand.getDefiningOp()) {
      if (auto constantOp = llvm::dyn_cast<hw::ConstantOp>(definingOp)) {
        return getConstantValueExpression(constantOp);
      }
      return definingOp->getName().getStringRef().str();
    }
    return "unknown";
  }

  std::string getCOperationString(mlir::Operation *op) {
    return mlir::TypeSwitch<mlir::Operation *, std::string>(op)
        .Case<circt::comb::AddOp>(
            [this](circt::comb::AddOp addOp) -> std::string {
              std::string result;
              for (unsigned i = 0; i < addOp.getNumOperands(); ++i) {
                if (i > 0)
                  result += " + ";
                result += getCOperandString(addOp.getOperand(i));
              }
              return result;
            })
        .Case<circt::comb::SubOp>(
            [this](circt::comb::SubOp subOp) -> std::string {
              std::string lhs = getCOperandString(subOp.getOperand(0));
              std::string rhs = getCOperandString(subOp.getOperand(1));
              return lhs + " - " + rhs;
            })
        .Case<circt::comb::AndOp>(
            [this](circt::comb::AndOp andOp) -> std::string {
              std::string result;
              for (unsigned i = 0; i < andOp.getNumOperands(); ++i) {
                if (i > 0)
                  result += " & ";
                result += getCOperandString(andOp.getOperand(i));
              }
              return result;
            })
        .Case<circt::comb::OrOp>([this](circt::comb::OrOp orOp) -> std::string {
          std::string result;
          for (unsigned i = 0; i < orOp.getNumOperands(); ++i) {
            if (i > 0)
              result += " | ";
            result += getCOperandString(orOp.getOperand(i));
          }
          return result;
        })
        .Case<circt::comb::XorOp>(
            [this](circt::comb::XorOp xorOp) -> std::string {
              std::string result;
              for (unsigned i = 0; i < xorOp.getNumOperands(); ++i) {
                if (i > 0)
                  result += " ^ ";
                result += getCOperandString(xorOp.getOperand(i));
              }
              return result;
            })
        .Case<circt::comb::ConcatOp>(
            [this](circt::comb::ConcatOp concatOp) -> std::string {
              std::string result;
              for (unsigned i = 0; i < concatOp.getNumOperands(); ++i) {
                if (i > 0)
                  result += " | ";
                result += getCOperandString(concatOp.getOperand(i));
              }
              return result;
            })
        .Case<hw::ConstantOp>([this](hw::ConstantOp constantOp) -> std::string {
          return getConstantValueExpression(constantOp);
        })
        .Case<circt::comb::MuxOp>(
            [this](circt::comb::MuxOp muxOp) -> std::string {
              std::string cond = getCOperandString(muxOp.getOperand(0));
              std::string trueVal = getCOperandString(muxOp.getOperand(1));
              std::string falseVal = getCOperandString(muxOp.getOperand(2));
              return cond + " ? " + trueVal + " : " + falseVal;
            })
        .Case<SpecHLS::GammaOp>(
            [this](SpecHLS::GammaOp gammaOp) -> std::string {
              std::string select = getCOperandString(gammaOp.getSelect());
              std::string result = "mux(" + select + ", ";
              for (unsigned i = 0; i < gammaOp.getInputs().size(); ++i) {
                if (i > 0)
                  result += ", ";
                result += getCOperandString(gammaOp.getInputs()[i]);
              }
              result += ")";
              return result;
            })
        .Case<SpecHLS::CastOp>([this](SpecHLS::CastOp castOp) -> std::string {
          std::string operand = getCOperandString(castOp.getOperand());
          std::string targetType = getCTypeString(castOp.getType());
          return "static_cast<" + targetType + ">(" + operand + ")";
        })
        .Case<hw::InstanceOp>([this](hw::InstanceOp instanceOp) -> std::string {
          std::string result = instanceOp.getInstanceNameAttr().str() + "(";
          for (unsigned i = 0; i < instanceOp.getNumOperands(); ++i) {
            if (i > 0)
              result += ", ";
            result += getCOperandString(instanceOp.getOperand(i));
          }
          result += ")";
          return result;
        })
        .Default([](mlir::Operation *) -> std::string {
          return "unsupported_operation";
        });
  }

  std::string getLHS(Operation *op) {
    if (op->getNumResults() > 1) {
      llvm::errs() << "Unsupported op " << *op << "\n";
      return "error";
    } else {
      return getCTypeString(op->getResult(0).getType());
    }
  }

  std::string generateCpp(HWModuleOp op) {
    std::string str = "";

    str.append("#include <stdio.h>\n");
    str.append("#include <string.h>\n");
    str.append("#include <math.h>\n");
    str.append("#include <ap_int.h>\n");
    str.append("#include <ap_uint.h>\n");
    str.append("\n");
    str.append("\n");

    str.append(llvm::formatv("void f_{0}(){\n", op.getSymName().str()));
    bool first = true;
    for (auto input : llvm::zip(op.getInputNames(), op.getInputTypes())) {
      if (!first) {
        str.append(",\n");
        first = false;
      }
      auto name = dyn_cast<StringAttr>(std::get<0>(input));
      auto type = std::get<1>(input);
      str.append(llvm::formatv("{0} {1}", type, name));
    }

    if (op->getNumRegions() > 0) {
      for (auto &region : op->getRegions()) {
        for (auto &block : region.getBlocks()) {
          for (auto &childOp : block.getOperations()) {

            str.append(llvm::formatv("{0} = {1} ;", getLHS(&childOp),
                                     getCOperationString(&childOp)));
          }
        }
      }

      str.append("}");
      return str;
    }
  }
  void runOnOperation() {
    auto *ctx = &getContext();
    auto module = this->getOperation();

    for (auto hwop : module.getOps<circt::hw::HWModuleOp>()) {
      // Create an output file stream object
      auto filename = hwop.getName() + ".Cppt";
      std::ofstream outFile(filename.str());
      // Check if the file was opened successfully
      if (!outFile) {
        llvm::errs() << "Error: Could not open the file " << filename << "\n";
      } else {
        // Write the string to the file
        outFile << generateCpp(hwop);
        llvm::outs() << "Content written to " << filename << "\n";
        // Close the file stream
        outFile.close();
      }
    }
  }

  std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExportCpp() {
    return std::make_unique<ExportCppPass>();
  }
}; }// namespace SpecHLS}