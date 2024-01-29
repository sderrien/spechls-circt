//
// Created by Steven on 04/07/2023.
//

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
// This file implements a translation between MLIR GPU dialect module and
// corresponding CUDA C file.
//
//===----------------------------------------------------------------------===//


#include "SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"
#include "Transforms/VitisExport/CFileContent.h"

#include "llvm/ADT/TypeSwitch.h"
#include <fstream>


using namespace std;
using namespace mlir;
using namespace SpecHLS;

using namespace circt::hw;
using namespace circt::comb;

std::string quote(string s);
string parent(string s);
string replace_all(string str, const string &r, const string &ins) ;
string assign(CFileContent *p,Value lhs, string rhs);

string op2str(Operation *v);
string value2str(Value *v);
string type2str(Type type);
string attr2str(Attribute attr);
string predicate2str(circt::comb::ICmpPredicate predicate);

string valueList(CFileContent *p,OperandRange range, string sep);
string argList(CFileContent *p,OperandRange range, string sep) ;


void printBinaryOp(CFileContent *p, Operation *op, string c) {
  if (op->getNumOperands() == 2 and op->getNumResults() == 1) {
    auto res = op->getResult(0);
    llvm::outs() << "In binop " << *op << "\n";
    auto resVar = p->getValueId(&res);
    auto decl = type2str(res.getType()) + " " + resVar + ";";
    llvm::outs() << decl << "for " << *op << "\n";
    p->appendDeclarations(decl);
    auto rhs = op->getOperand(0);
    auto lhs = op->getOperand(1);

    p->appendCombUpdate("// " + op2str(op));
    p->appendCombUpdate(resVar + "=" + p->getValueId(&rhs) + " " + c + " " + p->getValueId(&lhs) + ";\n");
  } else {
    llvm::errs() << "Operation " << op << " is not a binary op\n";
  }
}

void printAlpha(CFileContent *p, SpecHLS::AlphaOp op) {
  auto arrayType = type2str(op->getResultTypes()[0]);
  // auto arrayType = type2str(op->getOperand(0).getType());

  auto res = op.getResult();
  p->appendDeclarations(type2str(res.getType())+" "+p->getValueId(&res));

  p->appendDeclarations(arrayType + " update_" + op.getNameAttrName().str() +
                        "(" + argList(p,op->getOperands(), ",") + ");\n");
  p->appendCombUpdate("\tupdate_" + op.getNameAttrName().str() + "(" +
                      valueList(p,op->getOperands(), ",") + ");\n");
}

void printDelay(CFileContent *p, SpecHLS::DelayOp op) {
  auto type = type2str(op->getResultTypes()[0]);
  mlir::Operation *pop = op.getOperation();
  auto res = op.getResult();
  p->appendDeclarations(type2str(op.getType())+" "+p->getValueId(&res)+";");

  p->appendDeclarations("DelayLine <" + type + "," +
                        to_string(op.getDepth()) + "> " + p->getOpId(pop) +
                        ";\n");
  p->appendSyncUpdate("// "+ op2str(op.getOperation()));
  p->appendSyncUpdate(p->getValueId(&res) +"="+ p->getOpId(pop) + ".pop();\n");
  p->appendCombUpdate("// "+ op2str(op.getOperation()));
  p->appendCombUpdate(p->getOpId(pop) + ".push(" +
                      valueList(p,op->getOperands(), ",") + ");");
}

void printLUT(CFileContent *p, SpecHLS::LookUpTableOp op) {
  auto res = op.getResult();
  auto lutName= "LUT_"+p->getOpId(op.getOperation());
  auto baseTypeName = type2str(op.getType());
  auto lutDepth = op.getContent().size();

  string lutElts = "";

  for (u_int32_t i = 0; i < op.getContent().size(); i++) {
    auto elt  = cast<IntegerAttr>(op.getContent()[i]).getInt();
    lutElts += (i!=0?",":"") + to_string(elt)  ;
  }

  p->appendDeclarations(baseTypeName+" "+lutName+"["+ to_string(lutDepth)+"]={"+lutElts+"}");
  p->appendDeclarations(type2str(op.getType())+" "+p->getValueId(&res)+";");
  p->appendCombUpdate("\t" + p->getValueId(&res) +"="+ lutName +"[" +
                      valueList(p,op->getOperands(), "][") + "];\n");
}

void printGamma(CFileContent *p, SpecHLS::GammaOp op) {
  auto control = op->getOperand(0);
  auto res = op.getResult();
  p->appendDeclarations(type2str(op.getType())+" "+p->getValueId(&res)+";");

  p->appendCombUpdate("switch(" + p->getValueId(&control) + ") {\n");
  uint32_t item = 0;

  for (auto data : op->getOperands().drop_front(1)) {
    p->appendCombUpdate("case " + to_string(item++) + ": " +
                        p->getValueId(&res) + "=" + p->getValueId(&data) + "; break; ");
  }
  auto default_value = op->getOperands().back();
  p->appendCombUpdate("default : " + p->getValueId(&res) + "=" +
                      p->getValueId(&default_value) + "; break; ");
  p->appendCombUpdate("}\n");
}

void printArrayRead(CFileContent *p, SpecHLS::ArrayReadOp op) {
  auto res = op.getResult();
  auto type = type2str(op->getResultTypes()[0]);
  mlir::Operation *pop = op.getOperation();

  //p->appendDeclarations("SpecRead<" + type + ",0,0> " + p->getOpId(pop) + ";");
  p->appendDeclarations(type2str(op.getType())+" "+p->getValueId(&res)+";");
  p->appendCombUpdate(p->getOpId(pop) + " read(" +
                      valueList(p,op->getOperands(), ",") + ");");
}

void printMu(CFileContent *p, SpecHLS::MuOp op) {
  auto res = op.getResult();
  auto next = op.getNext();
  auto init = op.getInit();
  p->appendDeclarations(type2str(op.getType())+" "+p->getValueId(&res)+";");

  p->appendInitUpdate("// init Mu "+op.getNameAttr().getValue().str());
  p->appendInitUpdate(p->getValueId(&res) + "= " + p->getValueId(&init) + ";");
  p->appendInitUpdate("// update Mu "+op.getNameAttr().getValue().str());
  p->appendSyncUpdate(p->getValueId(&res) + "= " + p->getValueId(&next) + ";");
}

void printExit(CFileContent *p, SpecHLS::ExitOp op) {
  auto in = op->getOperand(0);
  p->appendCombUpdate("// "+ op2str(op.getOperation()));

  p->appendCombUpdate("\texit = " + p->getValueId(&in) + ";\n");
}

void printCast(CFileContent *p, mlir::UnrealizedConversionCastOp op) {
  auto res = op.getResult(0);
  auto in = op->getOperand(0);
  auto type = op.getResultTypes()[0];
  p->appendInitUpdate("// " + op2str(op.getOperation()));
  p->appendCombUpdate(p->getValueId(&res) + "= (" + type2str(type) + ") " +
                      p->getValueId(&in) + ";\n");
}

void printInstance(CFileContent *p, circt::hw::InstanceOp op) {
  p->appendInitUpdate("// " + op2str(op.getOperation()));
  // p->appendCombUpdate("\t"+ p->getValueId(&res)+"=
  // "+to_string(op.getValue().getSExtValue())+";\n");
}

void printConstant(CFileContent *p, circt::hw::ConstantOp op) {
  auto res = op.getResult();
  p->appendInitUpdate("// " + op2str(op.getOperation()));
  if (op.getType().isSignedInteger()) {

    p->appendInitUpdate(type2str(op.getType())+" "+p->getValueId(&res) + "= " +
                        to_string(op.getValue().getSExtValue()) + ";\n");
  } else {
    p->appendInitUpdate(type2str(op.getType())+" "+p->getValueId(&res) + "= " +
                        to_string(op.getValue().getZExtValue()) + ";\n");
  }
}
void printOutput(CFileContent *p, circt::hw::OutputOp op) {
  p->appendCombUpdate("// "+ op2str(op.getOperation()));
//  if (op->getNumOperands()>1) {
    p->appendCombUpdate("return {" + valueList(p,op->getOperands(), ",") + "};");
//  } else {
//    auto arg = op->getOperand(0);
//    p->appendCombUpdate("return " + p->getValueId(&arg)+";" );
//  }
}

void printExtern(CFileContent *p, circt::hw::HWModuleExternOp op) {
  p->appendInitUpdate("// " + op2str(op.getOperation()));
}

void printRollback(CFileContent *p, SpecHLS::RollbackOp op) {
  p->appendInitUpdate("// " + op2str(op.getOperation()));
}

void printConcat(CFileContent *p, circt::comb::ConcatOp op) {
  auto res = op->getResult(0);
  string current = "";
  p->appendDeclarations(type2str(op.getType())+" "+p->getValueId(&res)+";");
  for (u_int32_t k = 1; k < op->getNumOperands(); k++) {
    auto argk = op->getResult(k);
    auto bw = argk.getType().getIntOrFloatBitWidth();
    if (k == 0) {
      current = p->getValueId(&argk);
    }
    current =
        "(" + current + "<<" + to_string(bw) + ") | " + p->getValueId(&argk);
  }
  p->appendCombUpdate("// "+ op2str(op.getOperation()));
  p->appendCombUpdate(p->getValueId(&res) + "= " + current + ";\n");
}

void printExtract(CFileContent *p, circt::comb::ExtractOp op) {
  auto res = op->getResult(0);
  auto a = op->getOperand(0);
  auto mask = to_string((1 << getBitWidth(op.getType())) - 1);

  p->appendDeclarations(type2str(op.getType())+" "+p->getValueId(&res)+";");

  p->appendCombUpdate("// " + op2str(op.getOperation()));
  p->appendCombUpdate(p->getValueId(&res) + "= (" + p->getValueId(&a) + ">>" +
                      to_string(op.getLowBit()) + ")&" + mask + " ;\n");
}
void printCompare(CFileContent *p, circt::comb::ICmpOp op) {
  printBinaryOp(p,op.getOperation(), predicate2str(op.getPredicate()));
}

void printPrint(CFileContent *p, SpecHLS::PrintOp op) {
  auto res = op->getResult(0);
  auto format = op.getFormat().str();
  p->appendDeclarations(type2str(op.getType())+" "+p->getValueId(&res)+";");
  p->appendCombUpdate("// "+ op2str(op.getOperation()));
  p->appendCombUpdate(
      assign(p,op->getResult(0), " io_printf(" + quote(format) + "," +
                                   valueList(p,op->getOperands(), ",") + ")"));
}

void printMux(CFileContent *p, circt::comb::MuxOp op) {
  auto res = op->getResult(0);
  auto c = op->getOperand(0);
  auto t = op->getOperand(1);
  auto f = op->getOperand(2);
  p->appendDeclarations(type2str(op.getType())+" "+p->getValueId(&res)+";");

  p->appendCombUpdate("// "+ op2str(op.getOperation()));
  p->appendCombUpdate(
      assign(p,res,
             p->getValueId(&c) + "?" + p->getValueId(&t) + ":" + p->getValueId(&f)));
}

string templateString(string pattern, map<string, string> m) {
  string res = pattern;
  for (auto iter = m.begin(); iter != m.end(); ++iter) {
    auto k = iter->first;
    auto v = iter->second;
    replace_all(res, "$" + k, v);
  }
  return res;
}

void printEncoder(CFileContent *p, SpecHLS::EncoderOp op) {
  auto res = op->getResult(0);
  auto bw = op.getType().getWidth();
  p->appendCombUpdate("// " + op2str(op.getOperation()));

  string pattern = R"(
       $elseif ($x}&(1<<$id)) {
          $res = $id;
       }
      )";

  for (u_int32_t k = bw - 1; k >= 0; k--) {
    auto in = op->getOperand(k);
    map<string, string> args = {
        {"res", p->getValueId(&res)},
        {"else", (k == (bw - 1)) ? "" : "else "},
        {"x", p->getValueId(&in)},
        {"id", to_string(k)}};
    p->appendCombUpdate(templateString(pattern, args));
  }
}

void printDecoder(CFileContent *p, SpecHLS::DecoderOp op) {
  auto res = op->getResult(0);
  auto bw = op.getType().getWidth();
  p->appendCombUpdate("// "+ op2str(op.getOperation()));
  string pattern = R"(
       $elseif ($x}&(1<<$id)) {
          $res = $id;
       }
      )";

  for (u_int32_t k = bw - 1; k >= 0; k--) {
    auto in = op->getOperand(k);
    map<string, string> args = {
        {"res", p->getValueId(&res)},
        {"else", (k == (bw - 1)) ? "" : "else "},
        {"x", p->getValueId(&in)},
        {"id", to_string(k)}};
    p->appendCombUpdate(templateString(pattern, args));
  }
}

void printDefault(CFileContent *p, Operation *op) {

  op->emitWarning("Unsupported");
  llvm_unreachable("Default operation printing reached - UNSUPPORTED");
}

void printUnaryOp(Operation args, string op);

void printOperation(CFileContent *p, Operation *op) {

  TypeSwitch<Operation *>(op)
      //  .Case<SpecHLS::InitOp>([&](auto op) {printInit(op);})

      .Case<ConstantOp>([&](auto op) { printConstant(p, op); })
      .Case<SpecHLS::MuOp>([&](auto op) { printMu(p, op); })
      .Case<InstanceOp>([&](auto op) {printInstance(p,op);})
      .Case<OutputOp>([&](auto op) {printOutput(p,op);})

      .Case<SpecHLS::DelayOp>([&](auto op) {printDelay(p,op);})
      .Case<SpecHLS::GammaOp>([&](auto op) {printGamma(p,op);})
      .Case<SpecHLS::RollbackOp>([&](auto op) {printRollback(p,op);})
      .Case<SpecHLS::AlphaOp>([&](auto op) {printAlpha(p,op);})
      .Case<SpecHLS::ArrayReadOp>([&](auto op)  {printArrayRead(p,op);})
      .Case<SpecHLS::EncoderOp>([&](auto op)  {printEncoder(p,op);})
      .Case<SpecHLS::DecoderOp>([&](auto op)  {printDecoder(p,op);})
      .Case<SpecHLS::LookUpTableOp>([&](auto op)  {printLUT(p,op);})

      .Case<MulOp>([&](auto op) { printBinaryOp(p, op, "*"); })
      .Case<AddOp>([&](auto op) { printBinaryOp(p, op, "+"); })
      .Case<SubOp>([&](auto op) { printBinaryOp(p, op, "-"); })
      .Case<AndOp>([&](auto op) { printBinaryOp(p, op, "&"); })
      .Case<OrOp>([&](auto op) { printBinaryOp(p, op, "|"); })
      .Case<XorOp>([&](auto op) { printBinaryOp(p, op, "^"); })
      .Case<DivSOp>([&](auto op) { printBinaryOp(p, op, "/"); })
      .Case<DivUOp>([&](auto op) { printBinaryOp(p, op, "/"); })
      .Case<ModSOp>([&](auto op) { printBinaryOp(p, op, "%"); })
      .Case<ModUOp>([&](auto op) { printBinaryOp(p, op, "%"); })
      .Case<ShrUOp>([&](auto op) { printBinaryOp(p, op, ">>"); })
      .Case<ShrSOp>([&](auto op) { printBinaryOp(p, op, ">>"); })
      .Case<ShlOp>([&](auto op) { printBinaryOp(p, op, "<<"); })

      .Case<mlir::UnrealizedConversionCastOp>(
          [&](auto op) { printCast(p, op); })

      .Case<ConcatOp>([&](ConcatOp op) { printConcat(p, op); })
      .Case<ExtractOp>([&](ExtractOp op) { printExtract(p, op); })
      .Case<ICmpOp>([&](auto op) { printCompare(p, op); })
      .Case<MuxOp>([&](auto op) { printMux(p, op); })

      //.Case<SpecHLS::PrintOp>([&](auto op)  {printPrint(op);})
      .Case<SpecHLS::ExitOp>([&](auto op) {printExit(p,op);})

      .Default([&](auto op) {
        // llvm::outs() << " default  " << op->getOperationName() << "\n";
        return string(" ");
      });
}

void printHWModule(CFileContent *p, circt::hw::HWModuleOp hwop) {

  auto bodyBlock = hwop.getBodyBlock();
  string moduleDecl;
  llvm::outs() << "Generating C code for " << hwop.getSymName().str() << "\n";

  p->appendIncludesUpdate("#include<ac_int.h>");
  p->appendIncludesUpdate("");
//  if (hwop->getNumResults() == 0) {
//    moduleDecl += "void ";
//  } else {
//    moduleDecl += type2str(hwop->getResult(0).getType()) + " ";
//  }

  auto name = hwop.getSymName().str();
  moduleDecl += "struct "+name+"_res {\n";
  u_int32_t nbout  =0;
  for (auto portInfo : hwop.getPortList()) {
    if (portInfo.isOutput()) {
      moduleDecl += "\t"+type2str(portInfo.type)+" "+portInfo.getName().str() +";\n";
      nbout++;
    }
  }

  moduleDecl += "};\n\n ";

  moduleDecl += "struct "+name+"_res "+ name + "(";

  if (!hwop.getBody().getBlocks().empty()) {
    auto nargs = hwop.getBody().getArguments().size();
    for (u_int32_t i = 0; i < nargs; i++) {
      auto arg = hwop.getBody().getArguments()[i];
      Type arg_type = arg.getType();
      moduleDecl += (i == 0 ? "" : ", ") + type2str(arg_type) + " " + p->getValueId(&arg);
    }
  }
  moduleDecl += ") {";
  p->appendDeclarations(moduleDecl);

  llvm::outs() << moduleDecl << " {\n";

  for (Operation &op : *bodyBlock) {
    llvm::outs() << "\t- print for " << op << "\n";
    printOperation(p, &op);
  }
}
