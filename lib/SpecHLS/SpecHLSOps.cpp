//===- SpecHLSOps.cpp - SpecHLS dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
#include "SpecHLS/SpecHLSUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "SpecHLS/SpecHLSOps.cpp.inc"
using namespace mlir;

namespace SpecHLS {

/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void GammaOp::print(mlir::OpAsmPrinter &printer) {
  //         %res = SpecHLS.gamma [i32 -> i32] %a ? %b:%c:%d
  printer << " @" << this->getName();
  printer << " " << this->getSelect() << " ? ";
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
  printer << " [" << this->getInput() << " ] :" << this->getResult().getType()
          << "= {";

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

  // auto bw = getResult().getType().getIntOrFloatBitWidth();
  auto inbw = getInput().getType().getIntOrFloatBitWidth();
  if (getContent().size() < (1 << inbw)) {
    return emitOpError("Inconsistent LUT content with " +
                       std::to_string(content.size()) +
                       " entry, with address on  " + std::to_string(inbw));
  }
  auto outbw = getResult().getType().getIntOrFloatBitWidth();
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
        // llvm::errs() << *getOperation() ""
        return emitOpError("LUT entry " + s0 + " cannot fit target bitwidth " +
                           s1);
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
  mlir::Type addrType = parser.getBuilder().getIntegerType(depth);

  if (parser.resolveOperand(selectOperand, addrType, result.operands))
    return mlir::failure();
  result.addTypes({dataType});

  NamedAttrList attrs;
  parser.parseOptionalAttrDict(attrs);
  result.addAttributes(attrs);

  return mlir::success();
}

#define SPEC_GAMMAOP_MAXOPERANDS 64

mlir::ParseResult parseOperandList(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  int nbargs = 0;
  ParseResult nok;
  mlir::Type type;

  while (nbargs < SPEC_GAMMAOP_MAXOPERANDS) {
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

    nbargs++;
    if (parser.parseOptionalComma()) {
      break;
    }
    if (parser.resolveOperand(dataop, type, result.operands)) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::ParseResult GammaOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::UnresolvedOperand selectOperand;
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, SPEC_GAMMAOP_MAXOPERANDS>
      dataOperands;

  mlir::Type dataType;

  StringAttr id = parser.getBuilder().getStringAttr("\"undef\"");

  ParseResult nok = parser.parseOptionalSymbolName(id);

  FlatSymbolRefAttr symbolAttr = FlatSymbolRefAttr::get(id);

  result.addAttribute("name", symbolAttr);

  nok = parser.parseOperand(selectOperand);
  if (nok)
    return mlir::failure();
  if (parser.parseQuestion())
    return mlir::failure();
  // llvm::errs() << "Parsed select operand \n" ;
  OpAsmParser::UnresolvedOperand *dataop = new OpAsmParser::UnresolvedOperand();
  int nbargs = 0;
  while (nbargs < SPEC_GAMMAOP_MAXOPERANDS) {
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

  mlir::Type selType =
      parser.getBuilder().getIntegerType(int(ceil(log(nbargs) / log(2))));

  // llvm::errs() << "parsed " << nbargs << " operands\n" ;

  // llvm::errs() << "sel type is " << selType ;

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

  // llvm::errs() << "parsed data operands\n" ;

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

    if (nbargs==0) {
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

  for (int k=0;k<this->getNumOperands();k++) {
    if (k>0) printer << ",";
    printer <<" "<< this->getOperand(k) << " : " << this->getOperand(k).getType() ;
  }
}

mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
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

  if (parser.parseLParen()) return mlir::failure();

  OpAsmParser::UnresolvedOperand *dataop = new OpAsmParser::UnresolvedOperand();
  Type datatype ;
  int nbargs = 0;
  while (true) {
    dataop = new OpAsmParser::UnresolvedOperand();
    dataOperands.push_back(*dataop);
    if (parser.parseOperand(dataOperands[nbargs])) return mlir::failure();

    if (parser.parseColon()) return mlir::failure();
    if (parser.parseType(dataType)) return mlir::failure();
    typeOperands.push_back(dataType);

    llvm::outs() << "type "<<dataType <<"\n";
    llvm::outs() << nbargs <<"\n";
    nbargs++;
    if (parser.parseOptionalComma()) break;

  }

  llvm::outs() << "done1\n";

  if (parser.parseRParen()) return mlir::failure();

  if (parser.parseKeyword("from")) return failure();
  mlir::OpAsmParser::UnresolvedOperand iostateOperand;
  mlir::Type ioType = parser.getBuilder().getI32Type();
  if (parser.parseOperand(iostateOperand)) return failure();
  if (parser.resolveOperand(iostateOperand, ioType, result.operands))
    return mlir::failure();
  llvm::outs() << "done2\n";

  if (parser.parseKeyword("when")) return failure();
  mlir::OpAsmParser::UnresolvedOperand enableOperand;
  mlir::Type enableType = parser.getBuilder().getI1Type();
  if (parser.parseOperand(enableOperand)) return failure();
  if (parser.resolveOperand(enableOperand, enableType, result.operands))
    return mlir::failure();

  llvm::outs() << "done3\n";
  for (int k = 0; k < nbargs; k++) {
    llvm::outs() << "resolving op " << k << dataOperands[k].name << ":"<< typeOperands[k]<<"\n";
    if (parser.resolveOperand(dataOperands[k], typeOperands[k], result.operands))
      return mlir::failure();
  }
  llvm::outs() << "done4\n";

  for (int k = 0; k < result.operands.size(); k++) {
    llvm::outs() << "arg:"<<result.operands[k]  <<"\n";
  }

  llvm::outs() << "done5\n";
  result.addTypes({ioType});

  NamedAttrList attrs;
  parser.parseOptionalAttrDict(attrs);
  result.addAttributes(attrs);

  return mlir::success();
}

void PrintOp::print(mlir::OpAsmPrinter &printer) {
  printer <<" "<< this->getFormat() << " ( ";

  for (int k=2;k<this->getNumOperands();k++) {
    if (k>2) printer << ",";
    printer <<" "<< this->getOperand(k) << " : " << this->getOperand(k).getType() ;
  }
  printer <<")";
  printer <<" from "<< this->getOperands()[0] ;
  printer <<" when "<< this->getOperands()[1];

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

  llvm::errs() << "Data type " << dataType << "\n";
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
    llvm::errs() << "With data  " << secondOperand.name << ":" << dataType
                 << "\n";
    llvm::errs() << "With enable  " << firstOperand.name << ":" << dataType
                 << "\n";
    if (!noInit) {
      if (parser.resolveOperand(thirdOperand, dataType, result.operands))
        return mlir::failure();
      llvm::errs() << "With init  " << thirdOperand.name << ":" << dataType
                   << "\n";
    }
  } else {
    if (parser.resolveOperand(firstOperand, dataType, result.operands))
      return mlir::failure();
    llvm::errs() << "With data  " << firstOperand.name << ":" << dataType
                 << "\n";
  }

  result.addTypes({dataType});

  result.addAttributes(attrs);
  llvm::errs() << "Done !\n";
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

  ParseResult nok;
  nok = parser.parseOperand(firstOperand);
  if (nok)
    return mlir::failure();
  if (parser.resolveOperand(firstOperand, parser.getBuilder().getIntegerType(1),
                            result.operands))
    return mlir::failure();

  nok = parser.parseOptionalKeyword("live");
  if (!nok) {
    if (parseOperandList(parser, result)) {
      return mlir::failure();
    };
  }
  NamedAttrList attrs;
  nok = parser.parseOptionalAttrDict(attrs);

  result.addTypes(parser.getBuilder().getIntegerType(1));

  result.addAttributes(attrs);

  return mlir::success();
}

/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void ExitOp::print(mlir::OpAsmPrinter &printer) {
  //         %res = SpecHLS.delay [i32 -> i32] %a ? %b:%c:%d
  printer << " " << this->getFinished();
  if (!this->getLiveout().empty()) {
    printer << " live ";
    printer << " " << this->getLiveout()[0] << ":"
            << this->getLiveout()[0].getType() << " ";
    for (uint32_t i = 1; i < this->getLiveout().size(); i++) {
      printer << "," << this->getLiveout()[i] << ":"
              << this->getLiveout()[i].getType() << " ";
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
  for (int k = 1; k < content.size(); k++) {
    if (content[k] != first) {
      different = true;
      break;
    }
  }
  if (!different) {
    llvm::errs() << *getOperation() << " is constant\n";
    return {first};
  }
  // Constant fold.
  auto input = adaptor.getInput().dyn_cast_or_null<IntegerAttr>();
  if (input != NULL) {
    auto index = input.getValue().getZExtValue();
    auto cellValue = adaptor.getContent()[index];
    auto arrayCellAttr = dyn_cast<mlir::IntegerAttr>(cellValue);

    if (arrayCellAttr != NULL) {
      auto type = getResult().getType();
      unsigned int bw = type.getWidth();
      if (bw > 32)
        getOperation()->emitError("Unsupported bitwidth in fold]n");
      int64_t res = 0;
      res = arrayCellAttr.getValue().getZExtValue();
      return {getIntAttr(APInt(bw, res, type.isSigned()), getContext())};
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
    return getInputs()[index];
  }

  return {};
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
  if (op->getOperand(0).getType()==op->getResult(0).getType()) {
    return adaptor.getInput();
  }
  //  if // mux(0, a, b) -> b
  //  // mux(1, a, b) -> a
  //  if (auto pred = adaptor.getSelect().dyn_cast_or_null<IntegerAttr>()) {
  //    auto index = pred.getValue().getZExtValue();
  //    return getInputs()[index];
  //  }

  return {};
}






} // namespace SpecHLS
