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

using namespace mlir;

namespace mlir {
namespace customc {

void CustomCEmitter::translateModule(mlir::ModuleOp module,
                                     llvm::raw_ostream &output) {
  std::ostringstream out;

  for (auto &op : module) {
    processModuleOrFunc(&op, out);
  }

  for (auto include : includes)
    output << "#include <" << include << ">\n";

  output << "\n";
  for (auto decl : funcDeclarations)
    output << decl << ";\n";

  output << "\n" << out.str();
}

void CustomCEmitter::processModuleOrFunc(Operation *op,
                                         std::ostringstream &out) {
  if (auto module = dyn_cast<ModuleOp>(op)) {
    for (auto &inner_op : *module.getBody()) {
      processModuleOrFunc(&inner_op, out);
    }
  } else if (isa<FuncOp>(op)) {
    printFunction(op, out, "");
  }
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
  switch (attr.getKind()) {
  case StandardAttributes::Kind::Array: {
    std::string res = "{";
    auto array = attr.cast<ArrayAttr>();
    interleave(
        array.begin(), array.end(), [&](Attribute a) { res += attr2str(a); },
        [&] { res += ","; });
    return res + "}";
  }
  case StandardAttributes::Kind::Bool:
    return attr.cast<BoolAttr>().getValue() ? "true" : "false";
  case StandardAttributes::Kind::Float:
    return std::to_string(attr.cast<FloatAttr>().getValueAsDouble());
  case StandardAttributes::Kind::Integer:
    return std::to_string(attr.cast<IntegerAttr>().getInt());
  case StandardAttributes::Kind::Opaque:
    return attr.cast<OpaqueAttr>().getAttrData().str();
  case StandardAttributes::Kind::String:
    return attr.cast<StringAttr>().getValue().str();
  case StandardAttributes::Kind::SymbolRef:
    return attr.cast<SymbolRefAttr>().getLeafReference().str();
  case StandardAttributes::Kind::Type:
    return type2str(attr.cast<TypeAttr>().getValue());
  case StandardAttributes::Kind::Unit:
    llvm_unreachable("Unit attribute cannot be translated to CUDA C. Has "
                     "meaning only by existance.");
  case StandardAttributes::Kind::DenseElements:
    // TODO(limo1996): Handle multi dimensiional denses
    std::string res = "{";
    auto dense = attr.cast<DenseElementsAttr>();
    interleave(
        dense.attr_value_begin(), dense.attr_value_end(),
        [&](Attribute a) { res += attr2str(a); }, [&] { res += ","; });
    return res + "}";
  }
  llvm_unreachable("Unsupported attribute for CUDA C translation");
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

bool CustomCEmitter::printKernelVar(Operation *op, std::ostringstream &out,
                                    std::string indent) {
  std::string op_name = "";
  if (gpu::ThreadIdOp::classof(op)) {
    op_name = "threadIdx";
  } else if (gpu::BlockDimOp::classof(op)) {
    op_name = "blockDim";
  } else if (gpu::BlockIdOp::classof(op)) {
    op_name = "blockIdx";
  } else if (gpu::GridDimOp::classof(op)) {
    op_name = "gridDim";
  } else
    return false;

  auto dim = static_cast<gpu::BlockIdOp>(op).dimension().str();
  out << indent << "int " << getFreshVar(op) << " = " << op_name << "." << dim
      << ";\n";
  return true;
}

void CustomCEmitter::printLaunchFuncOp(gpu::LaunchFuncOp *launchOp,
                                       std::ostringstream &out,
                                       std::string indent) {
  std::vector<std::vector<std::string>> operands;
  auto kernel_name = appendModulePrefix(launchOp->kernel().str(),
                                        launchOp->getKernelModuleName().str());
  auto stencilFunc = launchOp->getParentOfType<FuncOp>();
  assert(stencilFunc && "expected parent function operation");
  for (unsigned int i = 0; i < launchOp->getNumKernelOperands(); i++) {
    auto operand = launchOp->getKernelOperand(i);
    auto op_name = value2str(operand);
    auto type = operand->getType();
    std::string size = "";
    if (MemRefType::classof(type)) {
      op_name += "_device";
      auto memref = type.cast<MemRefType>();
      if (memref.getShape().size() == 0)
        size = "1*sizeof(" + type2str(memref.getElementType()) + ")";
      else {
        auto isize = stencilFunc.getArgument(0);
        auto jsize = stencilFunc.getArgument(1);
        auto ksize = stencilFunc.getArgument(2);
        size = value2str(isize) + "*" + value2str(jsize) + "*" +
               value2str(ksize) + "*sizeof(" +
               type2str(memref.getElementType()) + ")";
      }
      if (launchFuncDeviceVars.find(op_name) == launchFuncDeviceVars.end()) {
        launchFuncDeviceVars.insert(op_name);
        out << indent << type2str(type) << " " << op_name << ";\n";
        out << indent << "cudaMalloc((void **)&" << op_name << ", ";
        out << size + ");\n";
      }

      out << indent << "cudaMemcpy(" << op_name << ", " << value2str(operand)
          << ", " << size << ", cudaMemcpyHostToDevice);\n";
    }
    operands.push_back({op_name, value2str(operand), size});
  }

  out << indent << kernel_name << "<<<dim3("
      << dim2str(launchOp->getGridSizeOperandValues()) << "),dim3("
      << dim2str(launchOp->getBlockSizeOperandValues()) << ")>>>"
      << "(";
  for (unsigned int i = 0; i < launchOp->getNumKernelOperands(); i++) {
    out << (i == 0 ? "" : ", ") << operands[i][0];
  }
  out << ");\n";
  for (unsigned int i = 0; i < launchOp->getNumKernelOperands(); i++) {
    if (operands[i][2] == "")
      continue;
    out << indent << "cudaMemcpy(" << operands[i][1] << ", " << operands[i][0]
        << ", " << operands[i][2] << ", cudaMemcpyDeviceToHost);\n";
  }
}

void CustomCEmitter::printConstantOp(ConstantOp *cop, std::ostringstream &out,
                                     std::string indent) {
  out << indent << type2str(cop->getType()) << " "
      << getFreshVar(cop->getOperation()) << " = "
      << attr2str(cop->getAttr("value")) << ";\n";
}

void CustomCEmitter::printSIToFPOp(SIToFPOp *fpOp, std::ostringstream &out,
                                   std::string indent) {
  auto type = type2str(fpOp->getResult()->getType());
  out << indent << type << " " << getFreshVar(fpOp->getOperation()) << " = ("
      << type << ")" << value2str(fpOp->in()) << ";\n";
}

void CustomCEmitter::printMemRefCastOp(MemRefCastOp *memCastOp,
                                       std::ostringstream &out,
                                       std::string indent) {
  auto type = type2str(memCastOp->getResult()->getType());
  out << indent << type << " " << getFreshVar(memCastOp->getOperation())
      << " = (" << type << ")" << value2str(memCastOp->source()) << ";\n";
}

void CustomCEmitter::printSqrtfOp(comb::SqrtfOp *sqrtOp,
                                  std::ostringstream &out, std::string indent) {
  out << indent << type2str(sqrtOp->getResult()->getType()) << " "
      << getFreshVar(sqrtOp->getOperation()) << " = sqrt("
      << value2str(sqrtOp->value()) << ");\n";
  includes.insert("math.h");
}

void CustomCEmitter::printFabsOp(comb::FabsOp *fabsOp, std::ostringstream &out,
                                 std::string indent) {
  out << indent << type2str(fabsOp->getResult()->getType()) << " "
      << getFreshVar(fabsOp->getOperation()) << " = fabs("
      << value2str(fabsOp->value()) << ");\n";
  includes.insert("math.h");
}

void CustomCEmitter::printExpOp(comb::ExpOp *expOp, std::ostringstream &out,
                                std::string indent) {
  out << indent << type2str(expOp->getResult()->getType()) << " "
      << getFreshVar(expOp->getOperation()) << " = exp("
      << value2str(expOp->value()) << ");\n";
  includes.insert("math.h");
}

void CustomCEmitter::printPowOp(comb::*powOp, std::ostringstream &out,
                                std::string indent) {
  out << indent << type2str(powOp->getResult()->getType()) << " "
      << getFreshVar(powOp->getOperation()) << " = pow("
      << value2str(powOp->value()) << ", " << value2str(powOp->exponent())
      << ");\n";
  includes.insert("math.h");
}

void CustomCEmitter::printReturnOp(Operation *op, std::ostringstream &out,
                                   std::string indent) {
  if (op->getNumOperands() == 1)
    out << indent << "return " << value2str(op->getOperand(0)) << ";\n";
  if (op->getNumOperands() > 1)
    llvm_unreachable("multi-return is not supported in CUDA C translation");
}

void CustomCEmitter::printLoadOp(LoadOp *op, std::ostringstream &out,
                                 std::string indent) {
  out << indent << type2str(op->getResult()->getType()) << " "
      << getFreshVar(op->getOperation()) << " = ";
  if (op->getNumOperands() == 1 &&
      op->getMemRef()->getType().cast<MemRefType>().getShape().size() > 0)
    out << "*";
  out << value2str(op->getOperand(0));
  for (unsigned int i = 1; i < op->getNumOperands(); i++)
    out << "[" << value2str(op->getOperand(i)) << "]";
  out << ";" << std::endl;
}

void CustomCEmitter::printStoreOp(StoreOp *op, std::ostringstream &out,
                                  std::string indent) {
  out << indent;
  if (auto memRef = op->getOperand(1)->getType().cast<MemRefType>())
    if (op->getNumOperands() == 2 && memRef.getShape().size() > 0)
      out << "*";
  out << value2str(op->getOperand(1));
  for (unsigned int i = 2; i < op->getNumOperands(); i++)
    out << "[" << value2str(op->getOperand(i)) << "]";
  out << " = " << value2str(op->getOperand(0)) << ";" << std::endl;
}

void CustomCEmitter::printDeallocOp(DeallocOp *op, std::ostringstream &out,
                                    std::string indent) {
  out << indent << "delete " << value2str(op->getOperand()) << ";\n";
}

void CustomCEmitter::printAllocOp(AllocOp *allocOp, std::ostringstream &out,
                                  std::string indent) {
  auto memref = allocOp->getType().cast<MemRefType>();
  auto shapeSize = memref.getShape().size();
  auto allocVar = getFreshVar(allocOp->getOperation());

  out << indent;

  if (shapeSize == 0)
    out << type2str(allocOp->getResult()->getType()) << " " << allocVar
        << ";\n";
  else if (isStaticArray(memref)) { // size known at compile time
    out << type2str(memref.getElementType()) << " " << allocVar;
    for (auto s : memref.getShape())
      out << "[" << s << "]";
    out << ";\n";
  } else { // dynamic allocation
    out << type2str(allocOp->getResult()->getType()) << " ";
    std::vector<std::string> loopVars;
    std::string curr_indent = indent;

    for (unsigned int i = 0; i < shapeSize; i++) {
      out << (i == 0 ? "" : curr_indent) << allocVar;
      for (auto lvar : loopVars)
        out << "[" << lvar << "]";
      out << " = new " << getPointer(memref, i + 1, 0) << "["
          << getShapeAt(allocOp, i) << "];" << std::endl;
      if (i == shapeSize - 1)
        continue; // in most inner loop only emit alloc
      auto loopVar = getFreshVar(static_cast<Operation *>(nullptr));
      loopVars.push_back(loopVar);
      out << curr_indent << "for (int " << loopVar << " = 0; " << loopVar
          << " < " << getShapeAt(allocOp, i) << "; " << loopVar << "++) {"
          << std::endl;
      curr_indent += "    ";
    }

    for (unsigned long i = 0; i < loopVars.size(); i++) {
      curr_indent = curr_indent.substr(0, curr_indent.size() - 4);
      out << curr_indent << "}" << std::endl;
    }
  }
}

bool CustomCEmitter::printArithmetics(Operation *op, std::ostringstream &out,
                                      std::string indent) {
  std::string right_side = "";
  if (MulFOp::classof(op)) {
    right_side = bin2str(static_cast<MulFOp>(op), '*');
  } else if (MulIOp::classof(op)) {
    right_side = bin2str(static_cast<MulIOp>(op), '*');
  } else if (AddFOp::classof(op)) {
    right_side = bin2str(static_cast<AddFOp>(op), '+');
  } else if (AddIOp::classof(op)) {
    right_side = bin2str(static_cast<AddIOp>(op), '+');
  } else if (DivFOp::classof(op)) {
    right_side = bin2str(static_cast<DivFOp>(op), '/');
  } else if (DivISOp::classof(op)) {
    right_side = bin2str(static_cast<DivISOp>(op), '/');
  } else if (DivIUOp::classof(op)) {
    right_side = bin2str(static_cast<DivIUOp>(op), '/');
  } else if (RemFOp::classof(op)) {
    right_side = bin2str(static_cast<RemFOp>(op), '%');
  } else if (RemISOp::classof(op)) {
    right_side = bin2str(static_cast<RemISOp>(op), '%');
  } else if (RemIUOp::classof(op)) {
    right_side = bin2str(static_cast<RemIUOp>(op), '%');
  } else if (ExpOp::classof(op)) {
    llvm_unreachable("TODO e^n");
  } else if (AndOp::classof(op)) {
    right_side = bin2str(static_cast<AndOp>(op), '&');
  } else if (OrOp::classof(op)) {
    right_side = bin2str(static_cast<OrOp>(op), '|');
  } else if (SubFOp::classof(op)) {
    right_side = bin2str(static_cast<SubFOp>(op), '-');
  } else if (SubIOp::classof(op)) {
    right_side = bin2str(static_cast<SubIOp>(op), '-');
  } else if (XOrOp::classof(op)) {
    right_side = bin2str(static_cast<XOrOp>(op), '^');
  } else {
    return false;
  }

  out << indent << type2str(op->getResult(0)->getType()) << " "
      << getFreshVar(op) << " = " << right_side << ";\n";
  return true;
}

static std::string predicate2str(CmpFPredicate predicate) {
  switch (predicate) {
  case CmpFPredicate::ULT:
    return "<";
  case CmpFPredicate::UGT:
    return ">";
  case CmpFPredicate::ULE:
    return "<=";
  case CmpFPredicate::UGE:
    return ">=";
  case CmpFPredicate::UEQ:
    return "==";
  case CmpFPredicate::UNE:
    return "!=";
  case CmpFPredicate::FirstValidValue:
  case CmpFPredicate::OEQ:
  case CmpFPredicate::OGT:
  case CmpFPredicate::OGE:
  case CmpFPredicate::OLT:
  case CmpFPredicate::OLE:
  case CmpFPredicate::ONE:
  case CmpFPredicate::ORD:
  case CmpFPredicate::UNO:
  case CmpFPredicate::AlwaysTrue:
  case CmpFPredicate::NumPredicates:
    llvm_unreachable("Unsupported CmpFPredicate");
  }
  llvm_unreachable("This should be never reached!");
}

void CustomCEmitter::printCmpFOp(CmpFOp *cmpOp, std::ostringstream &out,
                                 std::string indent) {
  out << indent << type2str(cmpOp->getResult()->getType()) << " "
      << getFreshVar(cmpOp->getOperation()) << " = " << value2str(cmpOp->lhs())
      << " " << predicate2str(cmpOp->getPredicate()) << " "
      << value2str(cmpOp->rhs()) << ";\n";
}

void CustomCEmitter::printSelectOp(SelectOp *selOp, std::ostringstream &out,
                                   std::string indent) {
  out << indent << type2str(selOp->getResult()->getType()) << " "
      << getFreshVar(selOp->getOperation()) << " = "
      << value2str(selOp->condition()) << " ? "
      << value2str(selOp->true_value()) << " : "
      << value2str(selOp->false_value()) << ";\n";
}

void CustomCEmitter::printIfOp(loop::IfOp ifOp, std::ostringstream &out,
                               std::string indent) {
  out << indent << "if (" << value2str(ifOp.condition()) << ") {\n";
  for (auto &block : ifOp.thenRegion()) {
    for (auto &op : block)
      printOperation(&op, out, indent + "    ");
  }
  out << indent << "}\n" << indent << "else {\n";
  for (auto &block : ifOp.elseRegion()) {
    for (auto &op : block)
      printOperation(&op, out, indent + "    ");
  }
  out << indent << "}\n";
}

void CustomCEmitter::printForLoop(loop::ForOp *forOp, std::ostringstream &out,
                                  std::string indent) {
  auto var = forOp->getInductionVar();
  auto loopVar = getFreshVar(var);
  auto upperVar = value2str(forOp->upperBound());
  auto lowerVar = value2str(forOp->lowerBound());
  auto step = value2str(forOp->step());

  out << indent << "for (" << type2str(var->getType()) << " " << loopVar
      << " = " << lowerVar << "; " << loopVar << " < " << upperVar << "; "
      << loopVar << " += " << step << ") {\n";

  for (auto &iop : *forOp->getBody()) {
    printOperation(&iop, out, indent + "    ");
  }

  out << indent << "}\n";
}

void CustomCEmitter::printCallOp(CallOp *callOp, std::ostringstream &out,
                                 std::string indent) {
  auto results = callOp->getNumResults();
  assert(results <= 1);
  out << indent;
  if (results == 1) {
    out << type2str(callOp->getResult(0)->getType()) << " "
        << getFreshVar(callOp->getOperation()) << " = ";
  }
  out << callOp->callee().str() << "(";
  interleave(
      callOp->operands().begin(), callOp->operands().end(),
      [&](Value *operand) { out << value2str(operand); }, [&] { out << ","; });
  out << ");\n";
}

void CustomCEmitter::printDefaultOp(Operation *op, std::ostringstream &out,
                                    std::string indent) {
  op->emitWarning("Unsupported");
  llvm_unreachable("Default operation printing reached - UNSUPPORTED");
}

void CustomCEmitter::printOperation(Operation *op, std::ostringstream &out,
                                    std::string indent) {
  if (isa<gpu::ReturnOp>(op) || isa<mlir::ReturnOp>(op))
    printReturnOp(op, out, indent);
  else if (auto c = dyn_cast<ConstantOp>(op))
    printConstantOp(&c, out, indent);
  else if (auto fpOp = dyn_cast<SIToFPOp>(op))
    printSIToFPOp(&fpOp, out, indent);
  else if (auto castOp = dyn_cast<MemRefCastOp>(op))
    printMemRefCastOp(&castOp, out, indent);
  else if (auto cmpOp = dyn_cast<CmpFOp>(op))
    printCmpFOp(&cmpOp, out, indent);
  else if (auto selOp = dyn_cast<SelectOp>(op))
    printSelectOp(&selOp, out, indent);
  else if (auto sqrt = dyn_cast<comb::SqrtfOp>(op))
    printSqrtfOp(&sqrt, out, indent);
  else if (auto fabs = dyn_cast<comb::FabsOp>(op))
    printFabsOp(&fabs, out, indent);
  else if (auto exp = dyn_cast<comb::ExpOp>(op))
    printExpOp(&exp, out, indent);
  else if (auto pow = dyn_cast<comb::PowOp>(op))
    printPowOp(&pow, out, indent);
  else if (auto ifOp = dyn_cast<loop::IfOp>(op))
    printIfOp(ifOp, out, indent);
  else if (auto launchFunc = dyn_cast<gpu::LaunchFuncOp>(op))
    printLaunchFuncOp(&launchFunc, out, indent);
  else if (auto allocOp = dyn_cast<AllocOp>(op))
    printAllocOp(&allocOp, out, indent);
  else if (auto deallocOp = dyn_cast<DeallocOp>(op))
    printDeallocOp(&deallocOp, out, indent);
  else if (auto storeOp = dyn_cast<StoreOp>(op))
    printStoreOp(&storeOp, out, indent);
  else if (auto loadOp = dyn_cast<LoadOp>(op))
    printLoadOp(&loadOp, out, indent);
  else if (auto forOp = dyn_cast<loop::ForOp>(op))
    printForLoop(&forOp, out, indent);
  else if (auto callOp = dyn_cast<CallOp>(op))
    printCallOp(&callOp, out, indent);
  else if (!printKernelVar(op, out, indent) &&
           !printArithmetics(op, out, indent) && !isa<loop::TerminatorOp>(op))
    printDefaultOp(op, out, indent);
}

void CustomCEmitter::printFunction(Operation *op, std::ostringstream &out,
                                   std::string indent) {
  launchFuncDeviceVars.clear();
  auto func_op = static_cast<FuncOp>(op);
  auto nresults = func_op.getType().getNumResults();

  if (nresults > 1) {
    llvm_unreachable("Function has more than 1 result.");
  }

  auto funcName = getFuncName(op);

  out << indent;
  std::string declaration;
  if (isKernelFunc(op))
    declaration += "__global__ ";

  if (isHostFunc(op))
    declaration += "__host__ ";

  if (nresults == 0) {
    declaration += "void ";
  } else {
    declaration += type2str(func_op.getType().getResult(0)) + " ";
  }

  declaration += funcName + "(";

  for (unsigned int i = 0; i < func_op.getNumArguments(); i++) {
    Type arg_type;
    std::string arg_name = "arg" + std::to_string(i);
    if (!func_op.getBody().getBlocks().empty()) {
      Value *a = func_op.getArgument(i);
      argToName[a] = arg_name;
      arg_type = a->getType();
    } else {
      arg_type = func_op.getType().getInput(i);
    }
    declaration += (i == 0 ? "" : ", ") + type2str(arg_type) + " " + arg_name;
  }
  declaration += ")";
  funcDeclarations.push_back(declaration);
  out << declaration << " {\n";

  for (auto &block : func_op.getBody()) {
    for (auto &inner_op : block.getOperations()) {
      if (op == &inner_op)
        return;
      printOperation(&inner_op, out, indent + "    ");
    }
  }
  for (auto var : launchFuncDeviceVars)
    out << indent << "    cudaFree(" << var << ");\n";
  out << "}\n";
}
} // namespace customc
} // namespace mlir

static TranslateFromMLIRRegistration
    registration("mlir-to-cudac",
                 [](ModuleOp module, llvm::raw_ostream &output) {
                   auto translation = new gpu::CustomCEmitter();
                   translation->translateModule(module, output);
                   return success();
                 });
