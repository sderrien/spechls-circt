
#include "mlir-c/Conversion.h"
#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Func.h>
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

#include <mlir-c/IntegerSet.h>
#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Support.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlir-c/Conversion.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/Transforms.h"
#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/Dialect/Transform.h>
#include <mlir-c/IR.h>

#include "mlir/Transforms/Transforms.capi.h.inc"
#include <circt-c/Dialect/Comb.h>
#include <circt-c/Dialect/FSM.h>
#include <circt-c/Dialect/HW.h>
#include <circt-c/Dialect/HWArith.h>
#include <circt-c/Dialect/SV.h>
#include <circt-c/Dialect/Seq.h>

#include <CAPI/SpecHLS.h>

#include <mlir-c/IntegerSet.h>
#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Support.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// FIXME move into include file
MlirPass mlirCreateSchedulePass(void);
MlirPass mlirCreateMobilityPass(void);
MlirPass mlirCreateLocalMobilityPass(void);
MlirPass mlirCreateExportVitisHLS(void);

#define DEFINE_GECOS_API_PASS(name, pass)                                      \
  MlirModule name(MlirModule module) {                                         \
    MlirContext ctx = mlirModuleGetContext(module);                            \
    MlirOperation op = mlirModuleGetOperation(module);                         \
    MlirPassManager pm = mlirPassManagerCreate(ctx);                           \
    MlirPass p = mlirCreate##pass();                                           \
    mlirPassManagerAddOwnedPass(pm, p);                                        \
    MlirLogicalResult success = mlirPassManagerRunOnOp(pm, op);                \
    if (mlirLogicalResultIsFailure(success)) {                                 \
      fprintf(stderr, "Unexpected failure running pass manager.\n");           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
    mlirPassManagerDestroy(pm);                                                \
    return module;                                                             \
  }

DEFINE_GECOS_API_PASS(scheduleMLIR, SchedulePass);
DEFINE_GECOS_API_PASS(canonicalizeMLIR, TransformsCanonicalizer);
DEFINE_GECOS_API_PASS(mobilityMLIR, MobilityPass);
DEFINE_GECOS_API_PASS(localMobilityMLIR, LocalMobilityPass);

DEFINE_GECOS_API_PASS(exportVitisHLS, ExportVitisHLS);
// DEFINE_GECOS_API_PASS(extractControl,ExportVitis);