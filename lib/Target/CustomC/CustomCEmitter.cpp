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

#include "Target/CustomC/CustomCEmitter.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <fstream>


using namespace mlir;
using namespace SpecHLS;
using namespace mlir::builtin;


void SpecHLS::CustomCEmitter::translateModule(mlir::ModuleOp module,
                                     llvm::raw_ostream &output) {
  std::ostringstream out;

  for (auto &op : module) {
    processModuleOrFunc(&op, out);
  }

  for (auto include : includes)
    output << "#include <" << include << ">\n";

  output << "\n";
  for (auto decl : declarations)
    output << decl << ";\n";

  output << "\n" << out.str();
}


std::string CustomCEmitter::type2str(Type type) {
  // TODO(limo1996): pointers, char

  if (type.isInteger(1))
    return "bool";
  if (type.isInteger(16))
    return "short";
  if (type.isInteger(32))
    return "int";
  if (type.isInteger(64))
    return "long long";
  if (type.isF16() || type.isF32())
    return "float";
  if (type.isF64())
    return "double";
  if (type.isIndex())
    return "int";
  if (MemRefType::classof(type)) {
    auto memref = type.cast<MemRefType>();
    return getPointer(memref, 0, 0);
  }
  if (ShapedType::classof(type)) { // VectorType, TensorType
    includes.insert("array");
    auto vector = type.cast<ShapedType>();
    std::string res = "";
    for (size_t i = 0; i < vector.getShape().size(); i++)
      res += "std::array<";
    res += type2str(vector.getElementType());
    for (auto it : vector.getShape())
      res += "," + std::to_string(it) + ">";
    return res;
  }
  llvm_unreachable("Unsupported type for CUDA C translation");
}

std::string CustomCEmitter::attr2str(Attribute attr) {
  // TODO(limo1996): AffineMapAttr, Dictionary, IntegerSet
  TypeSwitch<Attribute>(attr)
      .Case<ArrayAttr>([&](ArrayAttr array ) {
            std::string res = "{";
            llvm::interleave(
                array.begin(), array.end(), [&](Attribute a) { res += attr2str(a); },
                [&] { res += ","; });
            return res + "}";
          })
      .Case<BoolAttr>([&](BoolAttr t) { return t.getValue()? "true" : "false"; })
      .Case<FloatAttr>([&](FloatAttr f) { return std::to_string(f.getValueAsDouble()); })
      .Case<IntegerAttr>([&](IntegerAttr attr) { return std::to_string(attr.cast<IntegerAttr>().getInt());})
      .Case<StringAttr>([&](StringAttr attr) { return attr.getValue().str(); })
      .Case<TypeAttr>([&](TypeAttr attr) { return type2str(attr.getValue()); })
      .Case<SymbolRefAttr>([&](SymbolRefAttr attr) { return attr.getLeafReference().str(); })
      .Case<DenseElementsAttr>([&](DenseElementsAttr attr) {

        std::string res = "{";
            auto dense = attr;
//            interleave(
//                dense.attr_value_begin(), dense.attr_value_end(),
//                [&](Attribute a) { res += attr2str(a); }, [&] { res += ","; });
            return res + "}";

      });
//  switch (attr.getKind()) {
//  case StandardAttributes::Kind::Array: {
//    std::string res = "{";
//    auto array = attr.cast<ArrayAttr>();
//    llvm::interleave(
//        array.begin(), array.end(), [&](Attribute a) { res += attr2str(a); },
//        [&] { res += ","; });
//    return res + "}";
//  }
//  case StandardAttributes::Kind::Bool:
//    return attr.cast<BoolAttr>().getValue() ? "true" : "false";
//  case StandardAttributes::Kind::Float:
//    return std::to_string(attr.cast<FloatAttr>().getValueAsDouble());
//  case StandardAttributes::Kind::Integer:
//    return std::to_string(attr.cast<IntegerAttr>().getInt());
//  case StandardAttributes::Kind::Opaque:
//    return attr.cast<OpaqueAttr>().getAttrData().str();
//  case StandardAttributes::Kind::String:
//    return attr.cast<OpaqueAttr>().getValue().str();
//  case StandardAttributes::Kind::SymbolRef:
//    return attr.cast<SymbolRefAttr>().getLeafReference().str();
//  case StandardAttributes::Kind::Type:
//    return type2str(attr.cast<TypeAttr>().getValue());
//  case StandardAttributes::Kind::Unit:
//    llvm_unreachable("Unit attribute cannot be translated to CUDA C. Has "
//                     "meaning only by existance.");
//  case StandardAttributes::Kind::DenseElements:
//    // TODO(limo1996): Handle multi dimensiional denses
//    std::string res = "{";
//    auto dense = attr.cast<DenseElementsAttr>();
//    interleave(
//        dense.attr_value_begin(), dense.attr_value_end(),
//        [&](Attribute a) { res += attr2str(a); }, [&] { res += ","; });
//    return res + "}";
//  }
   llvm_unreachable("Unsupported attribute ");
}

template <typename T>
std::string CustomCEmitter::op2str(T *v) {
  std::string s;
  llvm::raw_string_ostream r(s);
  v->print(r);
  return r.str();
}

std::string CustomCEmitter::value2str(Value *v) {
  if (argToName.find(v) != argToName.end())
    return argToName[v];

  if (auto dop = v->getDefiningOp()) {
    if (opToName.find(dop) != opToName.end())
      return opToName[dop];
  }
  return "unknown_value";
}




std::string CustomCEmitter::bin2str(Value lhs, char operand, Value rhs) {
  return this->value2str(&lhs) + " " + operand + " " +
         this->value2str(&rhs);
}

static std::string predicate2str(circt::comb::ICmpPredicate predicate) {
  switch (predicate) {
  case circt::comb::ICmpPredicate::slt:
    return "<";
  case circt::comb::ICmpPredicate::sgt:
    return ">";
  case circt::comb::ICmpPredicate::sge:
    return "<=";
  case circt::comb::ICmpPredicate::ult:
    return "<";
  case circt::comb::ICmpPredicate::ugt:
    return ">";
  case circt::comb::ICmpPredicate::uge:
    return "<=";
  case circt::comb::ICmpPredicate::sle:
    return ">=";
  case circt::comb::ICmpPredicate::eq:
    return "==";
  case circt::comb::ICmpPredicate::ne:
    return "!=";
  default:

    llvm_unreachable("Unsupported ICmpPredicate");
  }
  llvm_unreachable("This should be never reached!");
}

//  std::string right_side = "";
//
//  auto pattern = R"(
//                if ({{expr}}) {
//                    {}
//                } else {
//                    {}
//                )";
//
//  std::string str = replace_all(replace_all(replace_all(pattern,
//                                                        "{{first_name}}", "Homer"),
//                                            "{{last_name}}", "Simpson"),"{{location}}", "Springfield");

std::string CustomCEmitter::valueList(OperandRange range, std::string sep) {
  std::string res = "";
  if (range.size()>0) {
    auto arrayVar =  range[0];
    res = value2str(&arrayVar);
    for (size_t i=1;i<range.size(); i++) {
      auto var = range[i];
      res = res +  sep + value2str(&var) ;
      return "";
    }
  }
  return res;
}

std::string CustomCEmitter::argList(OperandRange range, std::string sep) {
  std::string res = "";
  if (range.size()>0) {
    auto arrayVar =  range[0];
    res = value2str(&arrayVar);
    for (size_t i=1;i<range.size(); i++) {
      auto var = range[i];
      res = res + type2str(var.getType())+ " " + value2str(&var) + ";\n";
      return "";
    }
  }
  return res;
}

void CustomCEmitter::printBinaryOp(Operation* op, std::string c) {

  if (op->getNumOperands()==2 and op->getNumResults()==1) {
    auto res = op->getResult(0);
    auto rhs= op->getOperand(0);
    auto lhs= op->getOperand(1);
    this->combUpdate.push_back("\t"+ value2str(&res)+"="+ value2str(&rhs) + " "+c+" " + value2str(&lhs) + ";\n");
  } else {
    llvm::errs() << "Operation " << op << " is not a binary op\n";
  }

}



void  CustomCEmitter::printAlpha(SpecHLS::AlphaOp op) {
    auto arrayType = type2str(op->getResultTypes()[0]);
    //auto arrayType = type2str(op->getOperand(0).getType());

    this->globals.push_back(arrayType+" update_"+op.getNameAttrName().str()+"(" + argList(op->getOperands(), ",") + ");\n");
    this->combUpdate.push_back("\tupdate_"+op.getNameAttrName().str()+"(" + valueList(op->getOperands(), ",") + ");\n");
}

void  CustomCEmitter::printDelay(SpecHLS::DelayOp op) {
  auto res = op.getResult();
  auto type = type2str(op->getResultTypes()[0]);
  this->globals.push_back("DelayLine <"+type+","+std::to_string(op.getDepth())+"> "+ getId(op)+ ";\n");

  this->syncUpdate.push_back(getId(op)+".pop();\n");
  this->combUpdate.push_back(getId(op)+".push(" + valueList(op->getOperands(), ",") + ");\n");
}

void  CustomCEmitter::printLUT(SpecHLS::LookUpTableOp op) {
  auto res = op.getResult();
  this->combUpdate.push_back("\t"+ value2str(&res)+"=LUT_[" + valueList(op->getOperands(), "][") + "];\n");
}


void  CustomCEmitter::printGamma(SpecHLS::GammaOp op) {
  auto control = op->getOperand(0);
  this->combUpdate.push_back("switch("+value2str(&control)+") {\n");
  uint32_t item =0;
  auto res = op.getResult();
  for (auto data : op->getOperands().drop_front(1)) {
    this->combUpdate.push_back( "case " + std::to_string(item++) + ": "+ value2str(&res)+"="+ value2str(&data)+  "; break; ");
  }
  auto default_value= op->getOperands().back();
  this->combUpdate.push_back( "default : "+ value2str(&res)+"="+ value2str(&default_value)+  "; break; ");
  this->combUpdate.push_back("}\n");
}

void  CustomCEmitter::printArrayRead(SpecHLS::ArrayReadOp op) {
  auto res = op.getResult();
  auto type = type2str(op->getResultTypes()[0]);

  this->globals.push_back("SpecRead<"+type+",0,0> "+ getId(op)+ ";\n");

  this->combUpdate.push_back(getId(op)+ " read(" + valueList(op->getOperands(), ",") + ");\n");
}

void  CustomCEmitter::printMu(SpecHLS::MuOp op) {
  auto res = op.getResult();
  auto next = op.getNext();
  auto init = op.getInit();
  this->init.push_back("\t"+ value2str(&res)+"= "+value2str(&init)+"();\n");
  this->syncUpdate.push_back("\t"+ value2str(&res)+"= "+value2str(&next)+"();\n");
}

void  CustomCEmitter::printExit(SpecHLS::ExitOp op) {
  auto in = op->getOperand(0);
  this->combUpdate.push_back("\texit = " +value2str(&in)+";\n");

}

void CustomCEmitter::printCast(mlir::UnrealizedConversionCastOp op) {
  auto res = op.getResult(0);
  auto in = op->getOperand(0);
  auto type =  op.getResultTypes()[0];
  this->combUpdate.push_back("\t"+ value2str(&res)+"= ("+type2str(type)+") "+value2str(&in)+";\n");
}

void  CustomCEmitter::printInstance(circt::hw::InstanceOp op) {

}

void  CustomCEmitter::printConstant(circt::hw::ConstantOp op) {
  auto res = op.getResult();
  if (op.getType().isSignedInteger()) {
    this->init.push_back("\t"+ value2str(&res)+"= "+std::to_string(op.getValue().getSExtValue())+";\n");
  } else {
    this->init.push_back("\t"+ value2str(&res)+"= "+std::to_string(op.getValue().getZExtValue())+";\n");
  }
}
void  CustomCEmitter::printOutput(circt::hw::OutputOp op)  {
  this->init.push_back("return {"+valueList(op->getOperands(),",")+"}");
}
void  CustomCEmitter::printHWModule(circt::hw::HWModuleOp op)  {
  auto hwop = static_cast<circt::hw::HWModuleOp>(op);

  auto funcName = getFuncName(op);




  using namespace std;


  std::string declaration;

  if (hwop->getNumResults() == 0) {
    declaration += "void ";
  } else {
    declaration += type2str(hwop->getResult(0).getType()) + " ";
  }

  declaration += funcName + "(";

  for (unsigned int i = 0; i < hwop->getNumOperands(); i++) {
    Type arg_type;
    std::string arg_name = "arg" + std::to_string(i);
    if (!hwop.getBody().getBlocks().empty()) {
      auto a = hwop.getBody().getArguments()[i];
      //argToName[a] = arg_name;
      arg_type = a.getType();
    } else {
      arg_type = hwop->getOperands()[i].getType();
    }
    declaration += (i == 0 ? "" : ", ") + type2str(arg_type) + " " + arg_name;
  }
  declaration += ")";
  this->declarations.push_back(declaration);

  llvm::outs() << declaration << " {\n";

  for (auto &block : hwop.getBody()) {
    for (auto &inner_op : block.getOperations()) {
      if (op == &inner_op)
        return;
      printOperation(&inner_op);
    }
  }

  ofstream oFile;
  oFile.open(funcName+".cpp");

  for (auto &i : includes)
    oFile << i;

  for (auto &g : globals)
    oFile << g;

  for (auto &d : declarations)
    oFile << d;
  oFile << "bool exit;";
  for (auto &i : init)
    oFile << i;

  oFile << "do {";
  for (auto &c : combUpdate)
    oFile << c;
  for (auto &s : syncUpdate)
    oFile << s;
  oFile << "while (!exit);";
  oFile << "}";

}
void  CustomCEmitter::printExtern(circt::hw::HWModuleExternOp op)  {}

void CustomCEmitter::printDefault(Operation *op) {
  op->emitWarning("Unsupported");
  llvm_unreachable("Default operation printing reached - UNSUPPORTED");
}


void printUnaryOp(Operation args, std::string op) ;


void CustomCEmitter::printOperation(Operation *op) {

  TypeSwitch<Operation *>(op)


      //  .Case<SpecHLS::InitOp>([&](auto op) {printInit(op);})
      .Case<circt::hw::ConstantOp>([&](auto op) {printConstant(op);})
      .Case<SpecHLS::MuOp>([&](auto op) {printMu(op);})
      .Case<circt::hw::InstanceOp>([&](auto op) {printInstance(op);})
      .Case<circt::hw::OutputOp>([&](auto op) {printOutput(op);})

      .Case<SpecHLS::DelayOp>([&](auto op) {printDelay(op);})
      .Case<SpecHLS::GammaOp>([&](auto op) {printGamma(op);})
      .Case<SpecHLS::RollbackOp>([&](auto op) {printRollback(op);})
      .Case<SpecHLS::AlphaOp>([&](auto op) {printAlpha(op);})
      .Case<SpecHLS::ArrayReadOp>([&](auto op)  {printArrayRead(op);})
      .Case<SpecHLS::EncoderOp>([&](auto op)  {printEncoder(op);})
      .Case<SpecHLS::DecoderOp>([&](auto op)  {printDecoder(op);})

      .Case<circt::comb::MulOp>([&](auto op) {printBinaryOp(op,"*");})
      .Case<circt::comb::AddOp>([&](auto op) {printBinaryOp(op,"+");})
      .Case<circt::comb::SubOp>([&](auto op) {printBinaryOp(op,"-");})
      .Case<circt::comb::AndOp>([&](auto op) {printBinaryOp(op,"&");})
      .Case<circt::comb::OrOp >([&](auto op)   {printBinaryOp(op,"|");})
      .Case<circt::comb::XorOp>([&](auto op) {printBinaryOp(op,"^");})
      .Case<circt::comb::DivSOp>([&](auto op) {printBinaryOp(op,"/");})
      .Case<circt::comb::DivUOp>([&](auto op) {printBinaryOp(op,"/");})
      .Case<circt::comb::ModSOp>([&](auto op) {printBinaryOp(op,"%");})
      .Case<circt::comb::ModUOp>([&](auto op) {printBinaryOp(op,"%");})
      .Case<circt::comb::ShrUOp>([&](auto op) {printBinaryOp(op,">>");})
      .Case<circt::comb::ShrSOp>([&](auto op) {printBinaryOp(op,">>");})
      .Case<circt::comb::ShlOp >([&](auto op)   {printBinaryOp(op,"<<");})

      .Case<mlir::UnrealizedConversionCastOp>([&](auto op) {printCast(op);})

      .Case<circt::comb::ConcatOp>([&](circt::comb::ConcatOp op) {printConcat(op);})
      .Case<circt::comb::ExtractOp>([&](circt::comb::ExtractOp op) {printExtract(op);})
      .Case<circt::comb::ICmpOp>([&](auto op) {printCompare(op);})
      .Case<circt::comb::MuxOp>([&](auto op) {printMux(op);})

      .Case<SpecHLS::PrintOp>([&](auto op)  {printPrint(op);})
      .Case<SpecHLS::ExitOp>([&](auto op) {printExit(op);})

      .Default([&](auto op) {
        llvm::outs() << " default filter  " << *op << "\n";
        return std::string (" ");
      });
}
