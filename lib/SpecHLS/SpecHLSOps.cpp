//===- SpecHLSOps.cpp - SpecHLS dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SpecHLS/SpecHLSOps.h"
#include "SpecHLS/SpecHLSDialect.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "SpecHLS/SpecHLSOps.cpp.inc"
using namespace mlir;

namespace SpecHLS {

/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void GammaOp::print(mlir::OpAsmPrinter &printer) {
  //         %res = SpecHLS.gamma [i32 -> i32] %a ? %b:%c:%d
  printer  << " "<< this->getSelect() << " ? ";
  printer.printOptionalAttrDict(this->getOperation()->getAttrs());
  int size = this->getInputs().size();
  for (int i=0;i<(size-1);i++) {
    printer << this->getInputs()[i] << ",";
  }
  printer << this->getInputs()[size-1] ;

  printer  << " :" << this->getResult().getType() ;


}

/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void LookUpTableOp::print(mlir::OpAsmPrinter &printer) {
  //         %res = SpecHLS.gamma [i32 -> i32] %a ? %b:%c:%d
  printer  << " ["<< this->getInput() << " ] :" << this->getResult().getType() << "= {" ;


  ArrayAttr content = this->getContent();

  //printer.printAttribute();

  for (int i=0;i<content.size();i++) {
    if (i>0) printer << ",";
    printer << content[i].cast<IntegerAttr>().getInt();
  }

  printer  << " }";

  printer.printOptionalAttrDict(this->getOperation()->getAttrs(),{this->getContentAttrName()});


}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.

mlir::ParseResult LookUpTableOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    SmallVector<int,1024> content;
    int offset =0 ;
    ParseResult res ;
    mlir::Type dataType;
    mlir::OpAsmParser::UnresolvedOperand selectOperand;

    // [ op ]
    ParseResult  nok = parser.parseLSquare();
    if (nok) return mlir::failure();
    nok = parser.parseOperand(selectOperand);
    if (nok) return mlir::failure();
    nok = parser.parseRSquare();
    if (nok) return mlir::failure();

    // llvm::errs() << "LUT index " << selectOperand.name <<"\n" ;
    // [ op ] : type
    nok = parser.parseColon();
    if (nok) return mlir::failure();
    nok = parser.parseType(dataType);
    if (nok) return mlir::failure();
    // llvm::errs() << "LUT type " << dataType <<"\n" ;

    // [ op ] : type
    if (parser.parseEqual()) return mlir::failure();
    if (parser.parseLBrace()) return mlir::failure();
    int nbelt = 0;

    //
    // Parse comma separated integers
    //
    while (true) {
      int value;
      if (parser.parseInteger(value)) return mlir::failure();
      content.push_back(value);
      // llvm::errs() << " LUT["<< nbelt <<"] = " <<value <<"\n" ;
      nbelt++;
      if (parser.parseOptionalComma()) break;
    }
    if (parser.parseRBrace()) return mlir::failure();

    result.addAttribute("content", parser.getBuilder().getI32ArrayAttr(content));
    int depth = int(ceil(log(nbelt)/log(2)));
    if ((1<<depth)!=nbelt) {
      llvm::errs() << " Inconsistent number of values in LookUpTable (should be a power of two), but currently is  "<< nbelt;
      return mlir::failure();
    }
    mlir::Type addrType = parser.getBuilder().getIntegerType(depth);

    if (parser.resolveOperand(selectOperand, addrType, result.operands))
      return mlir::failure();
    result.addTypes({dataType});

    NamedAttrList attrs;
    parser.parseOptionalAttrDict(attrs);
    result.addAttributes(attrs);

    llvm::errs() << "OK\n";
    return mlir::success();
}


#define SPEC_GAMMAOP_MAXOPERANDS 64

mlir::ParseResult GammaOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::UnresolvedOperand selectOperand;
  SmallVector<mlir::OpAsmParser::UnresolvedOperand,SPEC_GAMMAOP_MAXOPERANDS> dataOperands;

  mlir::Type dataType;
  //llvm::errs() << "hello\n" ;
  MLIRContext *ctx = result.getContext();



  ParseResult  nok = parser.parseOperand(selectOperand);
  if (nok) return mlir::failure();
  if (parser.parseQuestion()) return mlir::failure();
  //llvm::errs() << "Parsed select operand \n" ;
  OpAsmParser::UnresolvedOperand *dataop = new OpAsmParser::UnresolvedOperand();
  int nbargs =0;
  while(nbargs < SPEC_GAMMAOP_MAXOPERANDS) {
    dataop = new OpAsmParser::UnresolvedOperand();
    dataOperands.push_back(*dataop);
    nok = parser.parseOperand(dataOperands[nbargs]);
    nbargs++;
    //llvm::errs() << "parsed operand "<< nbargs <<"\n" ;
    if (nok) return mlir::failure();
    if (parser.parseOptionalComma()) {
      //llvm::errs() << "no comma "<< nbargs <<"\n" ;
      break;
    }
  }
  if (nbargs <2) {
    //llvm::errs() << "invalid umber of operands \n" ;
    return mlir::failure();
  }
  nok = parser.parseColon();
  if (nok) return mlir::failure();

  nok = parser.parseType(dataType);
  if (nok) return mlir::failure();

  mlir::Type selType = parser.getBuilder().getIntegerType(int(ceil(log(nbargs)/log(2))));

  //llvm::errs() << "parsed " << nbargs << " operands\n" ;

  //llvm::errs() << "sel type is " << selType ;

  // Resolve the input operand to the type we parsed in.
  if (parser.resolveOperand(selectOperand, selType, result.operands))
    return mlir::failure();
  //llvm::errs() << "parsed select operands\n" ;

  for (int k=0;k< nbargs;k++) {
    if (parser.resolveOperand(dataOperands[k], dataType, result.operands))
      return mlir::failure();
  }
  result.addTypes({dataType});

  NamedAttrList attrs;
  parser.parseOptionalAttrDict(attrs);
  result.addAttributes(attrs);

  //llvm::errs() << "parsed data operands\n" ;

  return mlir::success();
}
}