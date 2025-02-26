//===- SpecHLSOps.cpp - SpecHLS dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSTypes.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Verifier.h"

#define GET_OP_CLASSES
#include "Dialect/SpecHLS/SpecHLSOps.cpp.inc"

using namespace mlir;

namespace SpecHLS {

ParseResult parseTypeListInParens(OpAsmParser &parser, SmallVectorImpl<Type> &types) {
  // Use Delimiter::Paren to parse types within parentheses
  return parser.parseCommaSeparatedList(mlir::OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
    Type type;
    if (parser.parseType(type))
      return parser.emitError(parser.getCurrentLocation(), "expected a type");
    types.push_back(type);
    return success();
  });
}

mlir::ParseResult parseOperandList(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  int nbargs = 0;
  ParseResult nok;
  mlir::Type type;

  do {
    auto dataop = OpAsmParser::UnresolvedOperand();
    nok = parser.parseOperand(dataop);
    if (nok)
      return mlir::failure();

    nok = parser.parseColon();
    if (nok)
      return mlir::failure();

    nok = parser.parseType(type);
    if (nok)
      return mlir::failure();

    if (parser.resolveOperand(dataop, type, result.operands)) {
      return mlir::failure();
    }
    nbargs++;
    nok = parser.parseOptionalComma();
  } while(!nok);

//  for (auto op : result.operands) {
//    llvm::errs() << op << "\n";
//
//  }

  return mlir::success();
}

/*
 *  ExitOP
 *
 *
 */

/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void GammaOp::print(mlir::OpAsmPrinter &printer) {
  //         %res = SpecHLS.gamma [i32 -> i32] %a ? %b:%c:%d
  printer << " @" << this->getName();
  printer << " " << this->getSelect() << ":" << this->getSelect().getType()
          << " ? ";
  int size = this->getInputs().size();
  for (int i = 0; i < (size - 1); i++) {
    printer << this->getInputs()[i] << ",";
  }
  printer << this->getInputs()[size - 1];

  printer << " :" << this->getResult().getType();
  printer.printOptionalAttrDict(this->getOperation()->getAttrs(), {"name"});
}

/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void LookUpTableOp::print(mlir::OpAsmPrinter &printer) {
  //         %res = SpecHLS.gamma [i32 -> i32] %a ? %b:%c:%d
  printer << " [" << this->getInput() << " : " << this->getInput().getType()
          << "] :" << this->getResult().getType() << "= {";

  ArrayAttr content = this->getContent();

  // printer.printAttribute();

  for (uint32_t i = 0; i < content.size(); i++) {
    if (i > 0)
      printer << ",";
    printer << content[i].cast<IntegerAttr>().getInt();
  }

  printer << " }";

  printer.printOptionalAttrDict(this->getOperation()->getAttrs(),
                                {this->getContentAttrName()});
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.

LogicalResult LookUpTableOp::verify() {
  auto content = getContent();
  auto resultType = dyn_cast<IntegerType>(getResult().getType());
  if (!resultType) {
    return emitOpError("Inconsistent LUT result type ");
  }
  auto inputType = dyn_cast<IntegerType>(getInput().getType());
  if (!inputType) {
    return emitOpError("Inconsistent LUT result type ");
  }

  auto outbw = resultType.getIntOrFloatBitWidth();
  auto inbw = inputType.getIntOrFloatBitWidth();

  if (getContent().size() != (1 << inbw)) {
    auto input = getInput();

    return emitOpError(
        "Inconsistent LUT content with " + std::to_string(content.size()) +
        " entries, with address on " + std::to_string(inbw) + " bits");
  }
  if (outbw > 32) {
    return emitOpError("Inconsistent LUT entry wordlength (max is 32)" + outbw);
  }
  auto max = (1l << outbw) - 1l;
  auto different = false;
  for (int k = 1; k < content.size(); k++) {
    auto intAttr = dyn_cast<IntegerAttr>(content[k]);
    if (intAttr) {
      if (intAttr.getValue().getZExtValue() > max) {
        auto s0 = std::to_string(intAttr.getValue().getZExtValue());
        auto s1 = std::to_string(outbw);
        // llvm::errs() << *getOperation() << "\n";
        return emitOpError("LUT entry value " + s0 +
                           " cannot fit target bitwidth " + s1);
      }
    }
  }
  return success();
}

mlir::ParseResult LookUpTableOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<int, 1024> content;
  ParseResult res;
  mlir::Type dataType;
  mlir::OpAsmParser::UnresolvedOperand selectOperand;

  // [ op ]
  ParseResult nok = parser.parseLSquare();
  if (nok)
    return mlir::failure();
  nok = parser.parseOperand(selectOperand);
  if (nok)
    return mlir::failure();

  nok = parser.parseColon();
  if (nok)
    return mlir::failure();

  mlir::Type addrType;
  parser.parseType(addrType);

  nok = parser.parseRSquare();
  if (nok)
    return mlir::failure();

  // llvm::errs() << "LUT index " << selectOperand.name <<"\n" ;
  // [ op ] : type
  nok = parser.parseColon();
  if (nok)
    return mlir::failure();
  nok = parser.parseType(dataType);
  if (nok)
    return mlir::failure();
  // llvm::errs() << "LUT type " << dataType <<"\n" ;

  // [ op ] : type
  if (parser.parseEqual())
    return mlir::failure();
  if (parser.parseLBrace())
    return mlir::failure();
  int nbelt = 0;

  //
  // Parse comma separated integers
  //
  while (true) {
    int value;
    if (parser.parseInteger(value)) {
      // llvm::errs() << " LUT["<< nbelt <<"] = " <<value <<"\n" ;
      return mlir::failure();
    }
    if (value >= (1l << (dataType.getIntOrFloatBitWidth()))) {
      llvm::errs() << " value " << value << " does not fit on " << dataType
                   << "\n";
      return mlir::failure();
    }
    content.push_back(value);
    // llvm::errs() << " LUT["<< nbelt <<"] = " <<value <<"\n" ;
    nbelt++;
    if (parser.parseOptionalComma())
      break;
  }
  if (parser.parseRBrace())
    return mlir::failure();

  result.addAttribute("content", parser.getBuilder().getI32ArrayAttr(content));
  int depth = int(ceil(log(nbelt) / log(2)));
  if ((1 << depth) != nbelt) {
    llvm::errs() << " Inconsistent number of values in LookUpTable (should be "
                    "a power of two), but currently is  "
                 << nbelt;
    return mlir::failure();
  }

  if (parser.resolveOperand(selectOperand, addrType, result.operands))
    return mlir::failure();
  result.addTypes({dataType});

  NamedAttrList attrs;
  parser.parseOptionalAttrDict(attrs);
  result.addAttributes(attrs);

  return mlir::success();
}

mlir::ParseResult GammaOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::UnresolvedOperand selectOperand;
  SmallVector<mlir::OpAsmParser::UnresolvedOperand> dataOperands;

  mlir::Type dataType;
  mlir::Type userSelType;
  mlir::Type selType;
  bool userDefinedSelType = false;
  StringAttr id = parser.getBuilder().getStringAttr("\"undef\"");

  ParseResult nok = parser.parseOptionalSymbolName(id);

  FlatSymbolRefAttr symbolAttr = FlatSymbolRefAttr::get(id);
  result.addAttribute("name", symbolAttr);

  nok = parser.parseOperand(selectOperand);
  if (nok) return mlir::failure();

  nok= parser.parseOptionalColon();
  if (nok) return mlir::failure();

  nok = parser.parseType(userSelType);
  if (nok) return mlir::failure();

  if (parser.parseQuestion())
    return mlir::failure();
  // llvm::errs() << "Parsed select operand \n" ;
  OpAsmParser::UnresolvedOperand *dataop = new OpAsmParser::UnresolvedOperand();
  int nbargs = 0;
  while (true) {
    dataop = new OpAsmParser::UnresolvedOperand();
    dataOperands.push_back(*dataop);
    nok = parser.parseOperand(dataOperands[nbargs]);
    nbargs++;
    // llvm::errs() << "parsed operand "<< nbargs <<"\n" ;
    if (nok)
      return mlir::failure();
    if (parser.parseOptionalComma()) {
      // llvm::errs() << "no comma "<< nbargs <<"\n" ;
      break;
    }
  }
  if (nbargs < 2) {
    // llvm::errs() << "invalid umber of operands \n" ;
    return mlir::failure();
  }
  nok = parser.parseColon();
  if (nok)
    return mlir::failure();

  nok = parser.parseType(dataType);
  if (nok)
    return mlir::failure();

  selType = userSelType;

  // Resolve the input operand to the type we parsed in.
  if (parser.resolveOperand(selectOperand, selType, result.operands))
    return mlir::failure();
  // llvm::errs() << "parsed select operands\n" ;

  for (int k = 0; k < nbargs; k++) {
    if (parser.resolveOperand(dataOperands[k], dataType, result.operands))
      return mlir::failure();
  }
  result.addTypes({dataType});
  // result.
  NamedAttrList attrs;
  parser.parseOptionalAttrDict(attrs);
  result.addAttributes(attrs);

  return mlir::success();
}

/*
 *  DelayOP
 *
 *
 */

mlir::ParseResult SyncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::UnresolvedOperand selectOperand;
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 16> dataOperands;
  SmallVector<mlir::Type, 16> dataTypes;

  mlir::Type dataType;
  mlir::Type resultType;

  // llvm::errs() << "Parsed select operand \n" ;
  OpAsmParser::UnresolvedOperand *dataop = new OpAsmParser::UnresolvedOperand();

  int nbargs = 0;
  while (true) {
    dataop = new OpAsmParser::UnresolvedOperand();
    auto nok = parser.parseOperand(*dataop);
    if (nok)
      return mlir::failure();
    nok = parser.parseColonType(dataType);
    if (nok)
      return mlir::failure();

    if (nbargs == 0) {
      resultType = dataType;
    }

    dataTypes.push_back(dataType);
    dataOperands.push_back(*dataop);
    nbargs++;
    // llvm::errs() << "parsed operand "<< nbargs <<"\n" ;
    if (parser.parseOptionalComma()) {
      // llvm::errs() << "no comma "<< nbargs <<"\n" ;
      break;
    }
  }

  for (int k = 0; k < dataOperands.size(); k++) {
    if (parser.resolveOperand(dataOperands[k], dataTypes[k], result.operands))
      return mlir::failure();
  }
  result.addTypes({resultType});
  // result.
  NamedAttrList attrs;
  parser.parseOptionalAttrDict(attrs);
  result.addAttributes(attrs);

  // llvm::errs() << "parsed data operands\n" ;

  return mlir::success();
}

// %t28 = SpecHLS.sync %t8 : memref<16xui32>, %t17 : ui32,%t17 : ui32

/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void SyncOp::print(mlir::OpAsmPrinter &printer) {
  //         %res = SpecHLS.delay [i32 -> i32] %a ? %b:%c:%d

  for (int k = 0; k < this->getNumOperands(); k++) {
    if (k > 0)
      printer << ",";
    printer << " " << this->getOperand(k) << " : "
            << this->getOperand(k).getType();
  }
}

mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 10> dataOperands;
  SmallVector<mlir::Type, 10> typeOperands;

  mlir::Type dataType;
  std::string format;
  StringAttr id = parser.getBuilder().getStringAttr("\"undef\"");

  ParseResult nok = parser.parseOptionalSymbolName(id);

  // Parse the quote string attribute
  StringAttr quoteStringAttr;
  if (parser.parseAttribute(quoteStringAttr, "format", result.attributes))
    return failure();
  int nbargs = 0;

  if (!parser.parseOptionalLParen()) {
    OpAsmParser::UnresolvedOperand *dataop =
        new OpAsmParser::UnresolvedOperand();
    Type datatype;
    while (true) {
      dataop = new OpAsmParser::UnresolvedOperand();
      dataOperands.push_back(*dataop);
      if (parser.parseOperand(dataOperands[nbargs]))
        return mlir::failure();

      if (parser.parseColon())
        return mlir::failure();
      if (parser.parseType(dataType))
        return mlir::failure();
      typeOperands.push_back(dataType);

      //    llvm::errs() << "type " << dataType << "\n";
      //    llvm::errs() << nbargs << "\n";
      nbargs++;
      if (parser.parseOptionalComma())
        break;
    }

    if (parser.parseRParen())
      return mlir::failure();
  }

  if (parser.parseKeyword("from"))
    return failure();
  mlir::OpAsmParser::UnresolvedOperand iostateOperand;
  mlir::Type ioType = parser.getBuilder().getI32Type();
  if (parser.parseOperand(iostateOperand))
    return failure();
  if (parser.resolveOperand(iostateOperand, ioType, result.operands))
    return mlir::failure();

  if (parser.parseKeyword("when"))
    return failure();
  mlir::OpAsmParser::UnresolvedOperand enableOperand;
  mlir::Type enableType = parser.getBuilder().getI1Type();
  if (parser.parseOperand(enableOperand))
    return failure();
  if (parser.resolveOperand(enableOperand, enableType, result.operands))
    return mlir::failure();

  for (int k = 0; k < nbargs; k++) {
    // llvm::outs() << "resolving op " << k << dataOperands[k].name << ":"<<
    // typeOperands[k]<<"\n";
    if (parser.resolveOperand(dataOperands[k], typeOperands[k],
                              result.operands))
      return mlir::failure();
  }
  result.addTypes({ioType});

  NamedAttrList attrs;
  parser.parseOptionalAttrDict(attrs);
  result.addAttributes(attrs);

  return mlir::success();
}

void findAndReplaceAll(std::string &data, const std::string &match,
                       const std::string &replace) {
  // Get the first occurrence
  size_t pos = data.find(match);

  // Repeat till end is reached
  while (pos != std::string::npos) {
    data.replace(pos, match.size(), replace);

    // Get the next occurrence from the current position
    pos = data.find(match, pos + replace.size());
  }
}
void PrintOp::print(mlir::OpAsmPrinter &printer) {
  auto format = this->getFormat().str();
  findAndReplaceAll(format, "\n", "\\n");
  if (this->getNumOperands() >= 3) {
    printer << " \"" << format << "\" ( ";

    for (int k = 2; k < this->getNumOperands(); k++) {
      if (k > 2)
        printer << ",";
      printer << " " << this->getOperand(k) << " : "
              << this->getOperand(k).getType();
    }
    printer << ")";
  }
  printer << " from " << this->getOperands()[0];
  printer << " when " << this->getOperands()[1];
}

/*
 *  DelayOP
 *
 *
 */

mlir::ParseResult DelayOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::UnresolvedOperand firstOperand;
  mlir::OpAsmParser::UnresolvedOperand secondOperand;
  mlir::OpAsmParser::UnresolvedOperand thirdOperand;

  mlir::Type dataType;
  int depth = 0;

  ParseResult nok;
  nok = parser.parseOperand(firstOperand);
  if (nok)
    return mlir::failure();
  ParseResult noInit;
  ParseResult noEnable = parser.parseOptionalArrow();
  if (!noEnable) {
    nok = parser.parseOperand(secondOperand);
    if (nok)
      return mlir::failure();

    nok = parser.parseKeyword(StringRef("by"));
    if (nok)
      return mlir::failure();
    nok = parser.parseInteger(depth);
    if (nok)
      return mlir::failure();

    noInit = parser.parseOptionalLParen();
    if (!noInit) {
      nok = parser.parseOperand(thirdOperand);
      if (nok)
        return mlir::failure();
      nok = parser.parseRParen();
      if (nok)
        return mlir::failure();
    }
  } else {
    nok = parser.parseKeyword(StringRef("by"));
    if (nok)
      return mlir::failure();
    nok = parser.parseInteger(depth);
    if (nok)
      return mlir::failure();
  }
  nok = parser.parseColon();
  if (nok)
    return mlir::failure();

  nok = parser.parseType(dataType);
  if (nok)
    return mlir::failure();

  result.addAttribute("depth", parser.getBuilder().getI32IntegerAttr(depth));

  SmallVector<int, 3> content = {1, !noEnable, !noInit};
  auto attr = parser.getBuilder().getDenseI32ArrayAttr(content);

  result.addAttribute("operandSegmentSizes", attr);
  NamedAttrList attrs;
  nok = parser.parseOptionalAttrDict(attrs);

  // Resolve the input operand to the type we parsed in.
  if (!noEnable) {
    if (parser.resolveOperand(secondOperand, dataType, result.operands))
      return mlir::failure();
    if (parser.resolveOperand(firstOperand,
                              parser.getBuilder().getIntegerType(1),
                              result.operands))
      return mlir::failure();
    //    llvm::errs() << "With data  " << secondOperand.name << ":" << dataType
    //                 << "\n";
    //    llvm::errs() << "With enable  " << firstOperand.name << ":" <<
    //    dataType
    //                 << "\n";
    if (!noInit) {
      if (parser.resolveOperand(thirdOperand, dataType, result.operands))
        return mlir::failure();
      //      llvm::errs() << "With init  " << thirdOperand.name << ":" <<
      //      dataType
      //                   << "\n";
    }
  } else {
    if (parser.resolveOperand(firstOperand, dataType, result.operands))
      return mlir::failure();
    //    llvm::errs() << "With data  " << firstOperand.name << ":" << dataType
    //                 << "\n";
  }

  result.addTypes({dataType});

  result.addAttributes(attrs);
  return mlir::success();
}

/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void DelayOp::print(mlir::OpAsmPrinter &printer) {
  //         %res = SpecHLS.delay [i32 -> i32] %a ? %b:%c:%d
  switch (this->getNumOperands()) {
  case 1: {
    printer << " " << this->getNext() << " by " << this->getDepth() << ": "
            << this->getType();
    break;
  }
  case 2: {
    printer << " " << this->getEnable() << " -> " << this->getNext() << " by "
            << this->getDepth() << " : " << this->getType();
    break;
  }
  case 3: {
    printer << " " << this->getEnable() << " -> " << this->getNext() << " by "
            << this->getDepth() << "(" << this->getInit()
            << ") : " << this->getType();
    break;
  }
  default: {
    printer << " Invalid ";
    break;
  }
  }
}

/*
 *  ExitOP
 *
 *
 */

mlir::ParseResult ExitOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::UnresolvedOperand firstOperand;
  mlir::OpAsmParser::UnresolvedOperand secondOperand;
  mlir::OpAsmParser::UnresolvedOperand thirdOperand;
  bool guarded = false;
  ParseResult nok;
  bool expectLiveOut = false;
  nok = parser.parseOptionalKeyword("live");

  if (nok) {
    nok = parser.parseOperand(firstOperand);
    if (nok)
      return mlir::failure();
    guarded = true;
    if (parser.resolveOperand(firstOperand, parser.getBuilder().getIntegerType(1),result.operands))
      return mlir::failure();
    expectLiveOut = !parser.parseOptionalKeyword("live");

  } else {
    expectLiveOut =true;
  }
  if (expectLiveOut) {

    if (parseOperandList(parser, result)) {
      return mlir::failure();
    };
  }

  NamedAttrList attrs;
  nok = parser.parseOptionalAttrDict(attrs);

  mlir::BoolAttr trueAttr = mlir::BoolAttr::get(parser.getContext(), guarded);
  result.addAttribute("guarded", trueAttr);

  //result.addTypes(parser.getBuilder().getIntegerType(1));
  result.addAttributes(attrs);
  return mlir::success();
}

/*
 *  ExitOP
 *
 *
 */

mlir::ParseResult CommitOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::UnresolvedOperand enableOperand;

  ParseResult nok;
  mlir::Type type;

  SmallVector<OpAsmParser::UnresolvedOperand, 16> operands;
  SmallVector<mlir::Type, 16> types;

  nok = parser.parseOptionalLParen();
  if (!nok) {

    while (1) {
      auto dataop = OpAsmParser::UnresolvedOperand();
      nok = parser.parseOperand(dataop);
      if (nok)
        return mlir::failure();

      nok = parser.parseColon();
      if (nok)
        return mlir::failure();

      nok = parser.parseType(type);
      if (nok)
        return mlir::failure();


      types.push_back(type);
      operands.push_back(dataop);

      if (parser.parseOptionalComma()) {
        break;
      }

    }

    nok = parser.parseRParen();
    if (nok) {
      return mlir::failure();
    }
  }

  nok = parser.parseKeyword("when");
  if (nok) {
    return mlir::failure();
  }
  nok = parser.parseOperand(enableOperand);
  if (nok) {
    return mlir::failure();
  }
  operands.insert(operands.begin(),enableOperand);
  types.insert(types.begin(),parser.getBuilder().getIntegerType(1));


  for (int i = 0; i<operands.size(); i++) {
    assert(operands.size()==types.size());
    assert(i<operands.size());
    assert(i<types.size());
    if (parser.resolveOperand(operands[i], types[i], result.operands)) {
      return mlir::failure();
    }
  }

  NamedAttrList attrs;
  nok = parser.parseOptionalAttrDict(attrs);
  result.addAttributes(attrs);
  return mlir::success();
}

/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void CommitOp::print(mlir::OpAsmPrinter &printer) {
  //         %res = SpecHLS.delay [i32 -> i32] %a ? %b:%c:%d
  int nbOperands = this->getOperands().size();
  if (nbOperands > 1) {
    printer << "(";
    for (int i = 1; i < nbOperands; i++) {
      if (i > 1)
        printer << ",";
      auto operand = this->getOperands()[i];
      printer << operand << ":" << operand.getType();
    }
    printer << ")";
  }
  printer << " when ";

  printer << this->getOperand(0);
}

/*
 *  ExitOP
 *
 *
 */

mlir::ParseResult GecosOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::UnresolvedOperand enableOperand;

  ParseResult nok;
  mlir::Type optype;
  mlir::Type type;

  SmallVector<OpAsmParser::UnresolvedOperand, 16> operands;
  SmallVector<mlir::Type, 16> types;
  std::string name;
  if (parser.parseString(&name)) {
    return mlir::failure();
  }

  if (parser.parseColon()) {
    return mlir::failure();
  }

  if (parser.parseType(optype)) {
    return mlir::failure();
  }

  while (1) {
    auto dataop = OpAsmParser::UnresolvedOperand();
    nok = parser.parseOperand(dataop);
    if (nok)
      return mlir::failure();

    nok = parser.parseColon();
    if (nok)
      return mlir::failure();

    nok = parser.parseType(type);
    if (nok)
      return mlir::failure();

    types.push_back(type);
    operands.push_back(dataop);

    // llvm::errs() << dataop.name <<":" << type << "\n";
    if (parser.parseOptionalComma()) {
      break;
    }
  }
  for (auto i = 0; i < operands.size(); i++) {
    if (i < operands.size() && i < types.size()) {
      auto op = operands[i];
      auto t = types[i];
      if (parser.resolveOperand(op, t, result.operands)) {
        return mlir::failure();
      }
    }
  }

  NamedAttrList attrs;
  nok = parser.parseOptionalAttrDict(attrs);
  result.addAttribute("name", parser.getBuilder().getStringAttr(name));
  result.addTypes(optype);
  result.addAttributes(attrs);
  return mlir::success();
}

/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void GecosOp::print(mlir::OpAsmPrinter &printer) {
  printer << " \"" << this->getName() << "\" :" << this->getResult().getType();
  for (auto i = 0; i < this->getOperands().size(); i++) {
    printer << " ";
    if (i > 0)
      printer << ",";
    auto operand = this->getOperands()[i];
    printer << operand << ":" << operand.getType();
  }

  printer.printOptionalAttrDict(this->getOperation()->getAttrs(), {"name"});
}

/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void ExitOp::print(mlir::OpAsmPrinter &printer) {
  //         %res = SpecHLS.delay [i32 -> i32] %a ? %b:%c:%d
  auto hasGuard = this->getGuarded();
  auto liveOutStart = 0;
  if (hasGuard) {
    printer << " " << this->getOperand(0);
    liveOutStart=1;
  }
  auto nbLiveOut = this->getNumOperands() - liveOutStart;
  if (nbLiveOut>0) {
    printer << " live ";
    auto operand = this->getOperand(liveOutStart);
    printer << " " <<  operand << ":" << operand.getType() << " ";
    for (uint32_t i = 0; i < nbLiveOut; i++) {
      operand =  this->getOperand(liveOutStart+i);
      printer << "," << operand << ":" << operand.getType() << " ";
    }
  }
}

/// From comb dialect
inline bool hasOperandsOutsideOfBlock(Operation *op) {
  Block *thisBlock = op->getBlock();
  return llvm::any_of(op->getOperands(), [&](Value operand) {
    return operand.getParentBlock() != thisBlock;
  });
}
inline static TypedAttr getIntAttr(const APInt &value, MLIRContext *context) {
  return IntegerAttr::get(IntegerType::get(context, value.getBitWidth()),
                          value);
}
OpFoldResult LookUpTableOp::fold(FoldAdaptor adaptor) {
  if (hasOperandsOutsideOfBlock(getOperation()))
    return {};

  auto content = adaptor.getContent();
  auto first = content[0];
  auto different = false;
  for (u_int32_t k = 1; k < content.size(); k++) {
    if (content[k] != first) {
      different = true;
      break;
    }
  }
  auto type = getResult().getType();
  unsigned int bw = type.getWidth();
  if (!different) {
    //llvm::errs() << *getOperation() << " is constant\n";
    if (auto intAttribute = dyn_cast<IntegerAttr>(first)) {
      auto constant = getIntAttr(
          APInt(bw, intAttribute.getInt(), type.isSigned()), getContext());
//      llvm::errs() << "Folding " << *getOperation() << " into constant "
//                   << constant << ":" << constant.getType() << "\n";
      return {constant};
    }
  }
  // Constant fold.
  auto input = adaptor.getInput().dyn_cast_or_null<IntegerAttr>();
  if (input != NULL) {
    auto index = input.getValue().getZExtValue();
    if (index >= adaptor.getContent().size()) {
//      llvm::errs() << "index " << index << " out of range [0" << ":"
//                   << adaptor.getContent().size()
//                   << "] in LookUpTableOp::fold(FoldAdaptor adaptor) \n";
      return {};
    }
    auto cellValue = adaptor.getContent()[index];
    auto arrayCellAttr = dyn_cast<mlir::IntegerAttr>(cellValue);

    if (arrayCellAttr != NULL) {
      if (bw > 32)
        getOperation()->emitError("Unsupported bitwidth in fold]n");
      int64_t res = 0;
      res = arrayCellAttr.getValue().getZExtValue();
      auto constant = getIntAttr(APInt(bw, res, type.isSigned()), getContext());
//      llvm::errs() << "Folding " << *getOperation() << " into constant "
//                   << constant << ":" << constant.getType() << "\n";
      return {constant};
    } else {
      llvm::errs() << "error in LookUpTableOp::fold(FoldAdaptor adaptor) \n";
    }
  }
  return {};
}

OpFoldResult GammaOp::fold(FoldAdaptor adaptor) {
  if (hasOperandsOutsideOfBlock(getOperation()))
    return {};

  // mux(0, a, b) -> b
  // mux(1, a, b) -> a
  if (auto pred = adaptor.getSelect().dyn_cast_or_null<IntegerAttr>()) {
    auto index = pred.getValue().getZExtValue();
    if (index >= getInputs().size()) {
      emitWarning("Out of range gamma folding at offset " +
                  std::to_string(index) + " for " +
                  std::to_string(getInputs().size()) +
                  " input gamma (undefined behavior ?)");
      return getInputs()[getInputs().size() - 1];
    }

    return getInputs()[index];
  }

  return {};
}

LogicalResult GammaOp::verify() {

  auto selectType = dyn_cast<IntegerType>(getSelect().getType());
  auto dataType = dyn_cast<IntegerType>(getResult().getType());

  if (!selectType) {
    return emitOpError("Inconsistent type for select input");
  }

  if (selectType.isUnsigned()) {
    return emitOpError("Gamma nodes do not support unsigned int types");
  }

  if (!dataType) {

    for (auto input : getInputs()) {
      if (input.getType() != getResult().getType()) {
        return emitOpError(
            "Inconsistent Gamma operation input data type width");
      }
    }
  }
  // auto bw = getResult().getType().getIntOrFloatBitWidth();
  auto inbw = getSelect().getType().getIntOrFloatBitWidth();
  auto nbInputs = getInputs().size();
  //  if ((nbInputs > (1 << inbw)) || (nbInputs < (1 << (inbw-1)))) {
  //    //llvm::errs() << "LookupTable : " <<this << "\n";
  //    //llvm::errs() << "select : "<< getSelect() << "\n";
  //    //llvm::errs() << "#inputs : "<< getInputs().size() << "\n";
  //    return emitOpError("Inconsistent Gamma select type width
  //    ("+std::to_string(getSelect().getType().getWidth())+") for
  //    "+std::to_string(nbInputs)+" inputs gamma");
  //  }

  return success();
}

struct ConstantControlGammaNode : public OpRewritePattern<GammaOp> {
  ConstantControlGammaNode(mlir::MLIRContext *context)
      : OpRewritePattern<GammaOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(GammaOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() > 0) {
      Value control = op.getOperand(0);
      auto controlOp = control.getDefiningOp();
      if (controlOp) {
        if (auto constantOp = dyn_cast<circt::hw::ConstantOp>(controlOp)) {
          uint32_t selected = constantOp.getValue().getZExtValue();
          if (selected >= 0 && selected < (op.getNumOperands() - 1)) {
            Value control = op.getOperand(selected + 1);
            rewriter.replaceOp(op, {control});
            return success();
          }
        }
      }
    }
    return failure();
  }
};

void GammaOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                          mlir::MLIRContext *ctxt) {
  results.add<ConstantControlGammaNode>(ctxt);
}

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  if (hasOperandsOutsideOfBlock(getOperation()))
    return {};

  auto op = getOperation();
  auto cast = dyn_cast<CastOp>(op);
  auto inType = cast.getOperand().getType();
  auto outType = cast.getResult().getType();

  if (inType == outType) {
    return getInput();
  }
  return {};
}

/*
 *  ExitOP
 *
 *
 */

mlir::ParseResult ExtractFieldOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  ParseResult nok;
  mlir::Type optype;
  mlir::Type restype;
  std::string name;

  if (parser.parseString(&name))
    return mlir::failure();
  if (parser.parseColon())
    return mlir::failure();
  if (parser.parseType(restype))
    return mlir::failure();

  if (parser.parseKeyword("from"))
    return mlir::failure();

  auto dataop = OpAsmParser::UnresolvedOperand();

  if (parser.parseOperand(dataop))
    return mlir::failure();
  if (parser.parseColon())
    return mlir::failure();
  if (parser.parseType(optype))
    return mlir::failure();
  if (parser.resolveOperand(dataop, optype, result.operands))
    return mlir::failure();

  NamedAttrList attrs;
  nok = parser.parseOptionalAttrDict(attrs);
  result.addTypes(restype);
  result.addAttributes(attrs);
  return mlir::success();
}

/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void ExtractFieldOp::print(mlir::OpAsmPrinter &printer) {
  printer << " \"" << this->getName() << "\":" << this->getResult().getType()
          << " from " << this->getInput() << ":" << this->getInput().getType();
  printer.printOptionalAttrDict(this->getOperation()->getAttrs(), {});
}

mlir::ParseResult PackStructOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  parser.emitError(parser.getCurrentLocation(),"Unsupported parsing for PackStructOp");
  return mlir::failure();
}

void PackStructOp::print(mlir::OpAsmPrinter &printer) {
  llvm::errs() << "Unsupported print for PackStructOp\n";
}

#define CUSTOM_ASM_HTHREAD
#ifdef CUSTOM_ASM_HTHREAD


mlir::ParseResult HTaskOp::parse(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  ParseResult nok;
  mlir::Type type;
  StringAttr name;
  SmallVector<Type> outTypes;

  if (parser.parseSymbolName(name))
    return mlir::failure();

  FlatSymbolRefAttr symbolAttr = FlatSymbolRefAttr::get(name);
  result.addAttribute("name", symbolAttr);

  if (parser.parseLParen())
    return mlir::failure();

  llvm::errs() << "name = " << name <<"\n";

  if (parser.parseOptionalRParen()) {
    if (parseOperandList(parser,result)) {
      return mlir::failure();
    }

    if (parser.parseRParen())
      return mlir::failure();
  }



 if (parser.parseArrowTypeList(outTypes))
   return mlir::failure();


 llvm::errs() <<" type list found  " << name << "\n";

  SmallVector<OpAsmParser::Argument, 4> regionArgs;

  // Parse the region with the parsed arguments.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  llvm::errs() <<" parse block regions success" << name << "\n";

  // Ensure the region has a block.
  if (body->empty())
    body->emplaceBlock();

  result.addTypes(outTypes);

  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs))  return mlir::failure();

  result.addAttributes(attrs);

  llvm::errs() <<" success for " << name << "\n";
  return mlir::success();
}
//
/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void HTaskOp::print(mlir::OpAsmPrinter &printer) {
  printer << "@"<< this->getName() <<" " ;

  printer << " (" ;
  for (auto k=0;k<this->getNumOperands();k++) {
    if (k>0) printer << ", " ;
    auto operand =this->getOperand(k);
    printer.printOperand(operand);
    printer << " : " ;
    printer.printType(operand.getType());
  }
  printer <<  ")  " ;

  printer.printArrowTypeList(this->getResultTypes());


  printer.printRegion(getOperation()->getRegion(0));

  printer.printOptionalAttrDict(this->getOperation()->getAttrs(), {"name"});

}



mlir::ParseResult HKernelOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  ParseResult nok;
  mlir::Type type;
  StringAttr name;
  SmallVector<Type> outTypes;

  if (parser.parseSymbolName(name))
    return mlir::failure();

  FlatSymbolRefAttr symbolAttr = FlatSymbolRefAttr::get(name);
  result.addAttribute("name", symbolAttr);


  if (parser.parseArrow()) {
    return mlir::failure();
  }

  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  // Parse the region with the parsed arguments.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();
  // Ensure the region has a block.
  if (body->empty())
    body->emplaceBlock();


  NamedAttrList attrs;
  parser.parseOptionalAttrDict(attrs);
  result.addAttributes(attrs);

  llvm::errs() <<" success for " << name << "\n";
  return mlir::success();
}
//
/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void HKernelOp::print(mlir::OpAsmPrinter &printer) {
  printer << " @"<< this->getName() << " -> " ;


  // Print the region body.
  printer.printRegion(getOperation()->getRegion(0), /*printEntryBlockArgs=*/false);

  printer.printOptionalAttrDict(this->getOperation()->getAttrs(), {"name"});
}

/*
void SpecHLS::ExitOp::build(mlir::OpBuilder &op, mlir::OperationState &state) {
  // Create an empty list of operands and result types.
  SmallVector<Value> operands = {};
  SmallVector<Type> types = {};

  // Convert the result types and operands to `TypeRange` and `ValueRange`.
  TypeRange resultTypes(types);
  ValueRange operandValues(operands);

  auto res=  op.create<SpecHLS::ExitOp>(state.location,operandValues,false);
//  llvm::errs() << res;
//  mlir::verify(res);
}
*/
#endif
} // namespace SpecHLS
