#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/CAPI/Registration.h"


#ifdef __cplusplus
extern "C" {
#endif
  void initSpecHLS() ;
  MlirModule parseMLIR(const char* mlir) ;
  void traverseRegion( MlirRegion region)
  void traverseMLIR(MlirModule module);
  void pass(const char* mlir);
#ifdef __cplusplus
}
#endif


