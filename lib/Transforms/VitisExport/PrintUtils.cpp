//
// Created by Steven on 24/01/2024.
//
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "Transforms/VitisExport/CFileContent.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace SpecHLS;

using namespace circt::hw;
using namespace circt::comb;

string op2str(Operation *v) {
  std::string s;
  llvm::raw_string_ostream r(s);
  v->print(r);
  return r.str();
}

string value2str(Value *v) {
  std::string s;
  llvm::raw_string_ostream r(s);
  v->print(r);
  return r.str();
}

std::string quote(std::string s) { return "\"" + s + "\""; }
std::string parent(std::string s) { return "(" + s + ")"; }

std::string replace_all(std::string str, const std::string &remove,
                        const std::string &insert) {
  std::string::size_type pos = 0;
  while ((pos = str.find(remove, pos)) != std::string::npos) {
    str.replace(pos, remove.size(), insert);
    pos++;
  }

  return str;
}

std::string assign(CFileContent *p, Value lhs, std::string rhs) {
  return p->getValueId(&lhs) + " = " + rhs + ";\n";
}

std::string type2str(Type type) {
  // TODO(limo1996): pointers, char

  if (type.isInteger(1))
    return "bool";
  if (type.isInteger(16))
    return "short";
  if (type.isInteger(32))
    return "int";
  if (type.isInteger(64))
    return "long long";
  if (type.isUnsignedInteger()) {
    return "ac_int<" + to_string(type.getIntOrFloatBitWidth()) + ",false>";
  }
  if (type.isSignedInteger()) {
    return "ac_int<" + to_string(type.getIntOrFloatBitWidth()) + ",true>";
  }
  if (isa<IntegerType>(type)) {
    if (type.isSignlessIntOrFloat()) {
      return "ac_int<" + to_string(type.getIntOrFloatBitWidth()) + ",false>";
    } else {
      return "ac_int<" + to_string(type.getIntOrFloatBitWidth()) + ",true>";
    }
  }
  if (type.isInteger(2)) {
    return "ac_int<" + to_string(type.getIntOrFloatBitWidth()) + ",true>";
  }
  if (type.isF16() || type.isF32())
    return "float";
  if (type.isF64())
    return "double";
  if (type.isIndex())
    return "int";

  if (MemRefType::classof(type)) {
    auto memref = type.cast<MemRefType>();
    return type2str(memref.getElementType()) + "*";
  }

  //  if (ShapedType::classof(type)) { // VectorType, TensorType
  //    includes.insert("array");
  //    auto vector = type.cast<ShapedType>();
  //    std::string res = "";
  //    for (size_t i = 0; i < vector.getShape().size(); i++)
  //      res += "std::array<";
  //    res += type2str(vector.getElementType());
  //    for (auto it : vector.getShape())
  //      res += "," + std::to_string(it) + ">";
  //    return res;
  //  }
  llvm::errs() << "Unsupported type " << type << "\n";
  llvm_unreachable("Unsupported type for Vitis C translation");
}

std::string attr2str(Attribute attr) {
  // TODO(limo1996): AffineMapAttr, Dictionary, IntegerSet
  TypeSwitch<Attribute>(attr)
      .Case<ArrayAttr>([&](ArrayAttr array) {
        std::string res = "{";
        llvm::interleave(
            array.begin(), array.end(),
            [&](Attribute a) { res += attr2str(a); }, [&] { res += ","; });
        return res + "}";
      })
      .Case<BoolAttr>(
          [&](BoolAttr t) { return t.getValue() ? "true" : "false"; })
      .Case<FloatAttr>(
          [&](FloatAttr f) { return std::to_string(f.getValueAsDouble()); })
      .Case<IntegerAttr>([&](IntegerAttr attr) {
        return std::to_string(attr.cast<IntegerAttr>().getInt());
      })
      .Case<StringAttr>([&](StringAttr attr) { return attr.getValue().str(); })
      .Case<TypeAttr>([&](TypeAttr attr) { return type2str(attr.getValue()); })
      .Case<SymbolRefAttr>(
          [&](SymbolRefAttr attr) { return attr.getLeafReference().str(); })
      .Case<DenseElementsAttr>([&](DenseElementsAttr attr) {
        std::string res = "{";
        //        auto dense = attr;
        //        //            interleave(
        //        //                dense.attr_value_begin(),
        //        dense.attr_value_end(),
        //        //                [&](Attribute a) { res += attr2str(a); },
        //        [&] { res +=
        //        //                ","; });
        return res + "}";
      });
  llvm::errs() << attr << "\n";

  // Get the TypeID of the attribute
  mlir::TypeID typeID = attr.getTypeID();

  // Print the TypeID as a string
  llvm::outs() << "Attribute " << attr << "\n";
  //" : " << typeID << "\n";

  llvm_unreachable("Unsupported attribute ");
}

string valueList(CFileContent *p, OperandRange range, std::string sep) {
  std::string res = "";
  if (range.size() > 0) {
    auto arrayVar = range[0];
    res = p->getValueId(&arrayVar);
    for (size_t i = 1; i < range.size(); i++) {
      auto var = range[i];
      res = res + sep + p->getValueId(&var);
      return "";
    }
  }
  return res;
}

std::string predicate2str(circt::comb::ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::slt:
    return "<";
  case ICmpPredicate::sgt:
    return ">";
  case ICmpPredicate::sge:
    return "<=";
  case ICmpPredicate::ult:
    return "<";
  case ICmpPredicate::ugt:
    return ">";
  case ICmpPredicate::uge:
    return "<=";
  case ICmpPredicate::sle:
    return ">=";
  case ICmpPredicate::eq:
    return "==";
  case ICmpPredicate::ne:
    return "!=";
  default:

    llvm_unreachable("Unsupported ICmpPredicate");
  }
  llvm_unreachable("This should be never reached!");
}

std::string argList(CFileContent *p, OperandRange range, std::string sep) {
  std::string res = "";
  if (range.size() > 0) {
    auto arrayVar = range[0];
    res = p->getValueId(&arrayVar);
    for (size_t i = 1; i < range.size(); i++) {
      auto var = range[i];
      res = res + type2str(var.getType()) + " " + p->getValueId(&var) + ";\n";
      return "";
    }
  }
  return res;
}