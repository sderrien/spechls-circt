
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

#include "SpecHLS/SpecHLSOps.h"
#include "SpecHLS/SpecHLSTypes.h"
#include "SpecHLS/SpecHLSUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "SpecHLS/SpecHLSOpsTypes.h.inc"
#include "SpecHLS/SpecHLSOpsTypes.cpp.inc"
using namespace mlir;

namespace SpecHLS {

/*
 "int64_t":$size,
"Type":$elementType,
"int64_t":$nbDiscardedWrites,
ArrayRefParameter<"int64_t">:$nbPendingWrites,
"int64_t":$maxPendingWrites,
"int64_t":$maxPendingAddresses
 */

//  let assemblyFormat = "`specarray` `<` $size `x`  $elementType `>` `{`
//  $$nbDiscardedWrites `,`  {$nbPendingWrites, $nbPendingWrites*},
//  $maxPendingWrites, $maxPendingAddresses `}`";

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

} // namespace SpecHLS