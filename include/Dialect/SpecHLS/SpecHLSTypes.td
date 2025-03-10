
#ifndef SpecHLS_TYPES
#define SpecHLS_TYPES

include "Dialect/SpecHLS/SpecHLSDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/EnumAttr.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"


include "mlir/IR/AttrTypeBase.td"


class SpecHLS_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SpecHLS_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}


def TypeArrayParam : ArrayRefParameter<"::mlir::Type", "Array of Types"> {
  let printer = [{ $_printer << $_self; }];

  let parser = [{
    [&]() -> ::mlir::FailureOr<::llvm::SmallVector<::mlir::Type>> {
      ::llvm::SmallVector<::mlir::Type> fields;
      if ($_parser.parseTypeList(fields)) return ::mlir::failure();
      return fields;
    }()
  }];
}

def StringArrayParam : ArrayRefParameter<"::std::string", "Array of string"> {
  let printer = [{
    bool first = true;
    for (auto &elt: $_self) {
      if (!first) $_printer << ", ";
      first = false;
      $_printer << "\"" << elt << "\"";
    }
  }];

  let parser = [{
    [&]() -> ::mlir::FailureOr<::llvm::SmallVector<::std::string>> {
      ::llvm::SmallVector<std::string> names;
      if ($_parser.parseCommaSeparatedList([&]() -> ::mlir::ParseResult {
          ::std::string name;
          if ($_parser.parseString(&name)) return ::mlir::failure();
          names.push_back(name);
          return ::mlir::success();
        }))
        return ::mlir::failure();
      return names;
    }()
  }];
}


def Struct : SpecHLS_Type<"Struct", "struct"> {
  let parameters = (ins StringRefParameter<>:$type_name,
                        TypeArrayParam:$fieldTypes,
                        StringArrayParam:$fieldNames);
  let assemblyFormat = "`<` $type_name `,` $fieldTypes `:` $fieldNames `>`";
}
#endif  // SpecHLS_TYPES