

#include "mlir-c/Conversion.h"
#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/MemRef.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/Dialect/Transform.h>
#include <mlir-c/IR.h>

#include <circt-c/Dialect/Comb.h>
#include <circt-c/Dialect/FSM.h>
#include <circt-c/Dialect/HW.h>
#include <circt-c/Dialect/HWArith.h>
#include <circt-c/Dialect/SV.h>
#include <circt-c/Dialect/Seq.h>

#include <CAPI/SpecHLS.h>

#include <CAPI/SSP.h>

#include <mlir-c/IntegerSet.h>
#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Support.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"

extern "C" {

bool isStructType(MlirType ctx) {
  return (llvm::dyn_cast<SpecHLS::StructType>(unwrap(ctx)) != NULL);
}

int getNumberOfFields(MlirType ctx) {
  auto stype = llvm::dyn_cast<SpecHLS::StructType>(unwrap(ctx));
  if (stype != NULL)
    return (stype.getFieldNames().size());
  else
    return -1;
}

MlirType getFieldType(MlirType ctx, int k) {
  auto stype = llvm::dyn_cast<SpecHLS::StructType>(unwrap(ctx));
  mlir::Type type = NULL;
  if (stype != NULL && stype.getFieldTypes().size() > k) {
    type = stype.getFieldTypes()[k];
  }
  return wrap(type);
}

const char *getFieldName(MlirType ctx, int k) {
  auto stype = llvm::dyn_cast<SpecHLS::StructType>(unwrap(ctx));
  mlir::StringRef name;
  if (stype != NULL && stype.getFieldNames().size() > k) {
    name = stype.getFieldNames()[k];
    return name.data();
  }
  return NULL;
}
}