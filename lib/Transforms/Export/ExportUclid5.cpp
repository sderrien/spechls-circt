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

struct ExportUclid5Pass : public impl::ExportUclid5Base<ExportUclid5Pass> {

private:
  llvm::DenseMap<Operation *, int> operationIDMap;
  llvm::DenseMap<void *, int> valueIdMap;
  int idValueCounter = 0;
  int idCounter = 100;

  int getOperationId(Operation *op) {
    if (!operationIDMap.lookup(op)) {
      operationIDMap[op] = idCounter++;
    }
    return operationIDMap[op];
  }

  void printValueIdMap(const llvm::DenseMap<void *, int> &valueIdMap) {
    for (const auto &pair : valueIdMap) {
      void* value = pair.first;
      int id = pair.second;

      // Afficher la valeur (clé)
      llvm::errs() << "Value: ";
      if (value!=NULL) {
        llvm::errs() << value;
        //value->print(llvm::errs());  // Affiche la représentation de la valeur
      } else {
        llvm::errs() << "NULL";
      }
      llvm::errs() << ", ID: " << id << "\n";
    }
  }

  int getValueId(Value *op) {
    //printValueIdMap(valueIdMap);
    auto ptr =op->getAsOpaquePointer();
    if (!valueIdMap.contains(ptr)) {
      valueIdMap[ptr] = idCounter++;
    }
    return valueIdMap[ptr];
  }

  std::string getOperandString(Value *v) {
    std::ostringstream oss;
    oss << "t" <<getValueId(v);
    return oss.str();
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

  //
  //
  //
  //

  std::string translateHThread(SpecHLS::HTaskOp hthread,std::ostringstream &oss) {

    llvm::errs() << "#### converting Htask " << hthread.getName().str() << "(";

    // Start the Uclid5 module
    oss << "procedure " << hthread.getName().str() << "(";

    auto bodyBlock = &hthread.getRegion().getBlocks().front();

    bool first = true;
    for (auto arg : bodyBlock->getArguments()) {
      oss << getOperandString(&arg) << " : " <<getUclid5TypeString(arg.getType());
      if (!first) oss << ",";
    }
    oss << ") modifies ";

    for (Value output : hthread->getResults()) {
      if (!first) {
        oss << (",");
        first = false;
      }
      oss << getOperandString(&output);
    }
    oss << ";\n { \n";



    // Translate operations within the module
    for (auto &op : bodyBlock->getOperations()) {
      std::string opString = getUclid5OperationString(&op);
      oss << "\t" << opString << "\n";
    }
    oss << "}\n";

    llvm::errs() << oss.str() << "\n";
    return oss.str();
  }

  std::string translateToplevel(SpecHLS::HKernelOp toplevel) {

    std::ostringstream oss;

    auto bodyBlock = &toplevel.getRegion().getBlocks().front();

    llvm::errs() << "#### converting Htasks\n\n";
    for (auto task : bodyBlock->getOps<SpecHLS::HTaskOp>()) {
      translateHThread(task,oss);
    }

    llvm::errs() << "\n#### converting types \n\n";

    // Translate operations within the module

    oss << " // converting types\n";
    for (auto &op : bodyBlock->getOperations()) {
      for (auto k = 0; k < op.getNumResults(); k++) {
        auto res = op.getResult(k);
        oss << getUclid5TypeString(res.getType()) << " t" << getValueId(&res) << ";\n";
      }
    }

    oss << " // converting types\n";
    for (auto &op : bodyBlock->getOperations()) {
      for (auto k = 0; k < op.getNumResults(); k++) {
        auto res = op.getResult(k);
        oss << getUclid5TypeString(res.getType()) << " t" << getValueId(&res) << ";\n";
      }
    }

    llvm::errs() << "\n#### converting elementary ops \n";
    oss << "init { \n";

    oss << " // converting types\n";
    for (auto &op : bodyBlock->getOperations()) {
         mlir::TypeSwitch<mlir::Operation*>(&op)
          .Case<SpecHLS::MuOp>(
              [this,&oss](SpecHLS::MuOp mu)  {
                //           auto res = mu.getResult(k);
                auto res = mu.getResult();
                auto defOp = mu->getOperand(0).getDefiningOp();
                assert(defOp!=NULL);
                oss << assign(defOp," = havoc;\n");
                oss << getOperandString(&res) << " = havoc;\n";
              }
              )
          .Case<SpecHLS::DelayOp>(
              [this,&oss](SpecHLS::DelayOp delay)  {
                //           auto res = mu.getResult(k);
                auto res = delay.getResult();
                auto defOp = delay->getOperand(0).getDefiningOp();
                assert(defOp!=NULL);
                oss << assign(defOp," = havoc;\n");
                oss << getOperandString(&res) << " = havoc;\n";
              }
              )
          .Case<SpecHLS::DelayOp>(
                 [&op](SpecHLS::DelayOp delay) -> std::string {
                   return "";
                 }).Default([this](mlir::Operation *op) -> std::string {
                       llvm::errs() << "\n->unsupported_operation :" << *op << " \n";
                       return "";
                     });

      }

    oss << " }\n";

    oss << "next {\n";

    for (auto &op : bodyBlock->getOperations()) {
      auto opStr =getUclid5OperationString(&op);
        oss << "\t"<<  opStr << "\n";
        llvm::errs() << op << " ==> " << opStr <<"\n";
    }

    oss << "}\n";
    llvm::errs() << "#### done \n";
    return oss.str();
  }

  /**************
   *
   *
   *
   *
   * @param mlirType
   * @return
   */
  const std::string getUclid5TypeString(mlir::Type mlirType) {
    return mlir::TypeSwitch<mlir::Type, std::string>(mlirType)
        .Case<mlir::IntegerType>([](mlir::IntegerType intType) -> std::string {
          if (intType.getWidth() == 1) {
            return "bool";
          } else {
            return llvm::formatv("bv[{0}]", intType.getWidth());
          }
        })
        .Case<mlir::FloatType>([](mlir::FloatType floatType) -> std::string {
          return "bv[32]"; // Assuming a 32-bit floating point, adjust as needed
        })
        .Case<mlir::MemRefType>([this](mlir::MemRefType memRefType) {
          auto elementType = memRefType.getElementType();
          return "array [" + getUclid5TypeString(elementType) + "]";
          return "array [" + getUclid5TypeString(elementType) + "]";
        })
        .Case<mlir::TupleType>(
            [this](mlir::TupleType tupleType) -> std::string {
              std::string result = "record {";
              for (unsigned i = 0, e = tupleType.size(); i != e; ++i) {
                if (i != 0)
                  result += ", ";
                result += "field" + std::to_string(i) + " : " +
                          getUclid5TypeString(tupleType.getType(i));
              }
              result += "}";
              return result;
            })
        .Default([](mlir::Type) -> std::string { return "uninterpreted"; });
  }


  std::string assign(mlir::Operation *op, string rhs) {
    std::ostringstream oss;
    if (op->getNumResults()>0) {
      Value res = op->getResult(0);
      oss << getUclid5TypeString(res.getType()) << " " << getOperandString(&res) << " = " << rhs << ";";
      return oss.str();
    } else {
      return rhs + ";";
    }
  }

  std::string reduce(mlir::Operation *op,const string separator=",",const int start=0, const int end=-1) {
    std::ostringstream oss;
    for (size_t i = start; i < op->getNumOperands(); ++i) {
      if (end<0 || i<end) {
        auto operand = op->getOperand(i);
        oss << getOperandString(&operand);
        if (i != op->getNumOperands() - 1) {
          oss << " " << separator << " ";
        }
      }
    }
    return oss.str();
  }

  std::string nAryOp(const string oper, mlir::Operation *op) {
    return assign(op, reduce(op,oper));
  }

  std::string getUclid5OperationString(mlir::Operation *op) {
    return mlir::TypeSwitch<mlir::Operation *, std::string>(op)
        .Case<circt::comb::AddOp>(
            [this,op](circt::comb::AddOp addOp) -> std::string {
              return nAryOp("+", op);
            })
        .Case<circt::comb::SubOp>(
            [this,op](circt::comb::SubOp subOp) -> std::string {
              return nAryOp("-", op);
            })
        .Case<circt::comb::AndOp>(
            [this,op](circt::comb::AndOp andOp) -> std::string {
              return nAryOp("&", op);
            })
        .Case<circt::comb::OrOp>(
            [this,op](circt::comb::OrOp orOp) -> std::string {
              return nAryOp("|", op);
          })
        .Case<circt::comb::XorOp>(
            [this,op](circt::comb::XorOp xorOp) -> std::string {
              return nAryOp("^", op);
            })
        .Case<circt::comb::ExtractOp>(
            [this,op](circt::comb::ExtractOp extractOp) -> std::string {

              int loBit = extractOp.getLowBit();
              auto resultType  = extractOp.getResult().getType();
              int hiBit = loBit + resultType.getIntOrFloatBitWidth() - 1;
              string rhs = "[" + to_string(hiBit) + ":" + std::to_string(loBit) + "]";

              return assign(op, rhs);
            })
        .Case<circt::comb::ConcatOp>(
            [this,op](circt::comb::ConcatOp concatOp) -> std::string {
              std::ostringstream oss;
              oss << "{" << nAryOp(",", op) << "}";
              return assign(op, oss.str());
            })
        .Case<hw::ConstantOp>([this](hw::ConstantOp constantOp) -> std::string {
            return getConstantValueExpression(constantOp);
        })
        .Case<circt::comb::MuxOp>(
            [this](circt::comb::MuxOp muxOp) -> std::string {
              auto operand = muxOp.getOperand(0);
              std::string cond = getOperandString(&operand);
              operand = muxOp.getOperand(1);
              std::string trueVal = getOperandString(&operand);
               operand = muxOp.getOperand(2);
              std::string falseVal = getOperandString(&operand);
              return cond + " ? " + trueVal + " : " + falseVal;
            })
        .Case<SpecHLS::GammaOp>(
            [this,op](SpecHLS::GammaOp gammaOp) -> std::string {
              std::ostringstream oss;
              auto res  = op->getResult(0);
              oss <<  getOperandString(&res) << " = { ";
              auto select = gammaOp.getOperand(0);
              std::string selectStr = getOperandString(&select);
              std::string result = "n_to_1_mux(" + selectStr+ ", {";
              for (unsigned i = 0; i < gammaOp.getInputs().size(); ++i) {
                if (i > 0)
                  result += ", ";
                auto operand = gammaOp.getOperand(i);

                result += getOperandString(&operand);
              }
              result += ")";
              oss << result;
              return oss.str();
            })
        .Case<SpecHLS::InitOp>(
            [this,op](SpecHLS::InitOp gammaOp) -> std::string {
              return assign(op, " havoc()");
            })
        .Case<SpecHLS::CommitOp>(
            [this,op](SpecHLS::CommitOp commitOp) -> std::string {
              if (auto parent = dyn_cast<SpecHLS::HTaskOp>(commitOp->getParentOp())) {
                std::ostringstream oss;
                auto guard = commitOp.getOperands().back();
                oss << "if (" << getOperandString(&guard) << ") { \n";
                for (auto i=0;i<parent->getNumResults();i++) {
                    auto nbCommits = commitOp.getNumOperands();
                    auto nbResults = parent->getNumResults();
                    commitOp.dump();
                    //assert(commitOp.getNumOperands()==(parent->getNumResults()+1));
                    for (unsigned i = 0; i < commitOp.getNumOperands()-1; ++i) {
                      auto rhs = commitOp.getOperand(i);
                      auto lhs = parent->getResult(i);
                      oss << "\t\t" << getOperandString(&lhs) << " = " << getOperandString(&rhs) << ";\n";
                    }
                }
                oss << "\tendif; \n";
                return oss.str();

              }
//              if (commitOp.getNumOperands()>1) {
//                auto operand = commitOp.getOperand(0);
//                return oss.str();
//              } else {
              }
            )
        .Case<SpecHLS::ExitOp>(
            [this,op](SpecHLS::ExitOp exitOp) -> std::string {
              return llvm::formatv("exit({0})",nAryOp(",", exitOp));
            })
        .Case<SpecHLS::HTaskOp>(
            [this](SpecHLS::HTaskOp taskOp) -> std::string {
              string rhs = llvm::formatv("call {0} ({1});",taskOp.getName(),reduce(taskOp));
              return rhs;
            })
        .Default([this](mlir::Operation *op) -> std::string {
          llvm::errs() << "\n->unsupported_operation :" << *op << " \n";
        });
  }

public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    auto module = this->getOperation();

    for (auto toplevel : module.getOps<SpecHLS::HKernelOp>()) {
      // Create an output file stream object
      auto filename = toplevel.getName() + ".ucl";
      std::ofstream outFile(filename.str());
      // Check if the file was opened successfully
      if (!outFile) {
        llvm::errs() << "Error: Could not open the file " << filename << "\n";
      } else {
        // Write the string to the file
        auto res= translateToplevel(toplevel);
        llvm::errs() << res;
        outFile << "module main {\n";

        outFile << res;
        llvm::outs() << "Content written to " << filename << "in "<< std::filesystem::current_path() <<"\n";
        // Close the file stream
        outFile << "}\n";
        outFile.close();
      }
    }
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExportUclid5Pass() {
  return std::make_unique<ExportUclid5Pass>();
}
} // namespace SpecHLS