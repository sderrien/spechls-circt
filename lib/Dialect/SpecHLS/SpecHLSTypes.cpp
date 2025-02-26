
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSTypes.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace llvm;

#include "Dialect/SpecHLS/SpecHLSOpsTypes.cpp.inc"


namespace SpecHLS {


//
//
///// This class represents the internal storage of the Toy `StructType`.
//struct StructTypeStorage : public mlir::TypeStorage {
//  /// The `KeyTy` is a required type that provides an interface for the storage
//  /// instance. This type will be used when uniquing an instance of the type
//  /// storage. For our struct type, we will unique each instance structurally on
//  /// the elements that it contains.
//  using KeyTy = STRUCT_DATA_TYPE;
//
//  /// A constructor for the type storage instance.
//  StructTypeStorage(STRUCT_DATA_TYPE data)
//      : data(data) {}
//
//  /// Define the comparison function for the key type with the current storage
//  /// instance. This is used when constructing a new instance to ensure that we
//  /// haven't already uniqued an instance of the given key.
//  bool operator==(const KeyTy &key) const { return key == data; }
//
//  /// Define a hash function for the key type. This is used when uniquing
//  /// instances of the storage.
//  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
//  /// have hash functions available, so we could just omit this entirely.
//  static llvm::hash_code hashKey(const KeyTy &key) {
//    return llvm::hash_value(key);
//  }
//
//  /// Define a construction function for the key type from a set of parameters.
//  /// These parameters will be provided when constructing the storage instance
//  /// itself, see the `StructType::get` method further below.
//  /// Note: This method isn't necessary because KeyTy can be directly
//  /// constructed with the given parameters.
//  static KeyTy getKey(STRUCT_DATA_TYPE data) {
//    return KeyTy(data);
//  }
//
//  /// Define a construction method for creating a new instance of this storage.
//  /// This method takes an instance of a storage allocator, and an instance of a
//  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
//  /// allocations used to create the type storage and its internal.
//  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
//                                      const KeyTy &key) {
//    // Copy the elements from the provided `KeyTy` into the allocator.
//
//
//    STRUCT_DATA_TYPE data= allocator.copyInto(key);
//
//    // Allocate the storage instance and construct it.
//    return new (allocator.allocate<StructTypeStorage>())
//        StructTypeStorage(data);
//  }
//
//  /// The following field contains the element types of the struct.
//  STRUCT_DATA_TYPE data;
//};
//
//
//
//// This class defines the Toy struct type. It represents a collection of
///// element types. All derived types in MLIR must inherit from the CRTP class
///// 'Type::TypeBase'. It takes as template parameters the concrete type
///// (StructType), the base class to use (Type), and the storage class
///// (StructTypeStorage).
//class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
//                                               StructTypeStorage> {
//public:
//  /// Inherit some necessary constructors from 'TypeBase'.
//  using Base::Base;
//
//  /// Create an instance of a `StructType` with the given element types. There
//  /// *must* be at least one element type.
//  static StructType get(STRUCT_DATA_TYPE info) {
//    assert(!info.empty() && "expected at least 1 element type");
//
//    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
//    // of this type. The first parameter is the context to unique in. The
//    // parameters after are forwarded to the storage instance.
//    mlir::MLIRContext *ctx = info.front().second.getContext();
//    return Base::get(ctx, info);
//  }
//
//  /// Returns the element types of this struct type.
//  StringRef getName() {
//    // 'getImpl' returns a pointer to the internal storage instance.
//    return getImpl()->data[0].first;
//  }
//
//  STRUCT_DATA_TYPE getFields() {
//    // 'getImpl' returns a pointer to the internal storage instance.
//    return getImpl()->data.drop_front(1);
//  }
//
//};
//

#ifdef USE_SPECARRAY_TYPE
mlir::Type SpecArrayType::parse(mlir::AsmParser &parser) {

  int64_t size;
  Type baseType;
  int64_t nbDiscardedWrites;
  int64_t maxPendingAddresses;
  int64_t maxPendingWrites;

  SmallVector<int64_t, 16> nbpendingWrites;
  ParseResult nok = parser.parseLess();
  nok = parser.parseLess();

  nok = parser.parseDimensionList(nbpendingWrites);

  nok = parser.parseType(baseType);

  nok = parser.parseGreater();

  nok = parser.parseLBrace();
  nok = parser.parseInteger(nbDiscardedWrites);
  nok = parser.parseComma();

  nok = parser.parseLBrace();
  do {
    int64_t tmp;
    nok = parser.parseInteger(tmp);
    nbpendingWrites.push_back(tmp);
    nok = parser.parseOptionalComma();
  } while (!nok);
  nok = parser.parseRBrace();

  nok = parser.parseComma();
  nok = parser.parseInteger(maxPendingWrites);

  nok = parser.parseComma();
  nok = parser.parseInteger(maxPendingAddresses);

  nok = parser.parseRBrace();
// SpecArrayType SpecArrayType::get(::mlir::MLIRContext *context,
  // int64_t size,
  // Type elementType,
  // int64_t nbDiscardedWrites,
  // ::llvm::ArrayRef<int64_t> nbPendingWrites,
  // int64_t maxPendingWrites,
  // int64_t maxPendingAddresses) {
  return parser.getBuilder().getType<SpecHLS::SpecArrayType>(0,baseType,nbDiscardedWrites,maxPendingWrites,maxPendingWrites,maxPendingAddresses);

  // parser.getBuilder()
}

/*
 "int64_t":$size,
"Type":$elementType,
"int64_t":$nbDiscardedWrites,
ArrayRefParameter<"int64_t">:$nbPendingWrites,
"int64_t":$maxPendingWrites,
"int64_t":$maxPendingAddresses
 */

// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void SpecArrayType::print(mlir::AsmPrinter &printer) const {

  printer << "<" << getSize() << "x" << getElementType() << ">";

  printer << "{" << (getNbDiscardedWrites());

  printer << "{" << (getNbDiscardedWrites());
  for (size_t k = 0; k < getNbPendingWrites().size(); k++) {
    if (k > 0)
      printer << ", ";
    printer << getNbPendingWrites()[k];
  }
  printer << "},";

  printer << (getMaxPendingWrites()) << ",";
  printer << (getMaxPendingAddresses()) << "}";
}
#endif

#ifdef USE_SPECSTRUCT_TYPE
mlir::Type SpecHLSDialect::parseType(mlir::DialectAsmParser &parser) const {

  int64_t size;
  Type baseType;

  std::string name;
  SmallVector<std::pair<StringRef,Type>, 128> fields;
  ParseResult nok;

  llvm::errs() << "Parsing \n";
  if (parser.parseKeyword("struct")) {
    parser.emitError(parser.getCurrentLocation(),"syntax error");
  }
  if (parser.parseString(&name)) return NULL;

  llvm::errs() << "struct name is "<< name<< "\n";
  // FIXME
  fields.push_back(std::pair(std::string(name),parser.getBuilder().getI8Type()));

  if (parser.parseLess()) return NULL;

  llvm::errs() << "Parsing 2\n";

  //parser.emitError(parser.getCurrentLocation(),"syntax error");
  do {
    std::string fieldName;
    Type fieldType;

    nok = parser.parseString(&fieldName);
    nok = parser.parseColon();
    nok = parser.parseType(fieldType);
    auto p = new std::pair(StringRef(std::string(fieldName)),fieldType);

    fields.push_back(*p);

    llvm::errs() << "Parsing " << fieldName << ":"<< StringRef(fieldName) <<" " << fieldType << "\n";
    llvm::errs() << " ->  " << p->first << ":"<< p->second << "\n";
  } while (succeeded(parser.parseOptionalComma()));
  nok = parser.parseGreater();


  llvm::errs() << "Parsing end\n";

  llvm::errs() << "fields\n";
  for (int k=0;k<fields.size();k++) {
    llvm::errs() << fields[k].first.str() << "->" << fields[k].second << "\n";
  }
  llvm::ArrayRef<std::pair<StringRef,Type>> arrayRef(fields);

  llvm::errs() << "arrayRef\n";
  for (int k=0;k<arrayRef.size();k++) {
    llvm::errs() << arrayRef[k].first.str() << "->" << arrayRef[k].second << "\n";
  }


  StructType typeresult= parser.getBuilder().getType<SpecHLS::StructType>();
  for (int k=0;k<typeresult.getFields().size();k++) {
    llvm::errs() << typeresult.getFields()[k].first.str() << "->" << typeresult.getFields()[k].second << "\n";
  }

  llvm::errs() << "Parsed "<< typeresult << "\n";
  return typeresult;// parser.getBuilder()
}
/// strings, attributes, operands, types, etc.
void SpecHLSDialect::printType(mlir::Type type,
               mlir::DialectAsmPrinter &printer) const  {

  StructType structType = type.cast<StructType>();
  printer << "struct " << structType.getName() << " <";

  for (size_t k = 0; k <structType.getFields().size(); k++) {
    auto field = structType.getFields()[k];
    if (k > 0)
      printer << ", ";
    printer << field.first.str() << ":" << field.second;
  }
  printer << ">";

}
#endif
} // namespace SpecHLS