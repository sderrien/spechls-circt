
#include <mlir-c/IR.h>
#include "mlir-c/Conversion.h"
#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/Transform.h>
#include <mlir-c/Dialect/SCF.h>

#include <circt-c/Dialect/SV.h>
#include <circt-c/Dialect/HW.h>
#include <circt-c/Dialect/Seq.h>
#include <circt-c/Dialect/Comb.h>
#include <circt-c/Dialect/HWArith.h>
#include <circt-c/Dialect/FSM.h>

#include "CAPI/SpecHLS.h"

#include <mlir-c/IntegerSet.h>
#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Support.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>




int main(int argc, char **argv) {

  //registerAllUpstreamDialects();
  //fwrite(mlir,strlen(mlir),1,stdout);
  MlirModule m = parseMLIR(mlir);
  traverseMLIRModule(m);

}
