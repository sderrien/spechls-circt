
#include "circt/Dialect/SSP/SSPPasses.h"

#include "Transforms/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Support.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/Passes.h"

using namespace circt;



#define DEFINE_CAPI_IMPL(pass)                                                 \
  void mlirRegister##pass(void) { SpecHLS::register##pass(); }                 \
  MlirPass mlirCreate##pass(void) {                                            \
    return wrap(SpecHLS::create##pass().release());                            \
  }

#define DEFINE_CAPI_DECL(pass)                                                 \
  void mlirRegister##pass(void);                                               \
  MlirPass mlirCreate##pass(void);

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

#ifdef __cplusplus
extern "C" {
#endif

DEFINE_CAPI_DECL(MergeLookUpTablesPass)
DEFINE_CAPI_DECL(SchedulePass)
DEFINE_CAPI_DECL(MobilityPass)
DEFINE_CAPI_DECL(YosysOptimizerPass)
DEFINE_CAPI_DECL(ExportVitisHLS)
DEFINE_CAPI_DECL(GroupControlNodePass)
DEFINE_CAPI_DECL(GroupGammaNodesPass)
DEFINE_CAPI_DECL(FactorGammaInputsPass)
DEFINE_CAPI_DECL(MergeGammasPass)
DEFINE_CAPI_DECL(EliminateRedundantGammaInputsPass)
DEFINE_CAPI_DECL(InlineModulesPass)
DEFINE_CAPI_DECL(LowerGecosOpsToCombPass)
DEFINE_CAPI_DECL(ConfigurationExcluderPass)
DEFINE_CAPI_DECL(WordLengthPropagationPass)
DEFINE_CAPI_DECL(TopoSortPass)

MlirPass mlirCreateConfigurationExcluderPass(void);
void mlirRegisterConfigurationExcluderPass(void);

MlirPass mlirCreateCSEPass(void);
void mlirRegisterCSEPass(void);

#ifdef __cplusplus
}
#endif
DEFINE_CAPI_IMPL(ConfigurationExcluderPass)
DEFINE_CAPI_IMPL(MergeLookUpTablesPass)
DEFINE_CAPI_IMPL(SchedulePass)
DEFINE_CAPI_IMPL(MobilityPass)
DEFINE_CAPI_IMPL(YosysOptimizerPass)
DEFINE_CAPI_IMPL(ExportVitisHLS)
DEFINE_CAPI_IMPL(GroupControlNodePass)
DEFINE_CAPI_IMPL(GroupGammaNodesPass)
DEFINE_CAPI_IMPL(FactorGammaInputsPass)
DEFINE_CAPI_IMPL(MergeGammasPass)
DEFINE_CAPI_IMPL(EliminateRedundantGammaInputsPass)
DEFINE_CAPI_IMPL(InlineModulesPass)
DEFINE_CAPI_IMPL(LowerGecosOpsToCombPass)
DEFINE_CAPI_IMPL(WordLengthPropagationPass)
DEFINE_CAPI_IMPL(TopoSortPass)

#ifdef __cplusplus
extern "C" {
#endif

DEFINE_C_API_STRUCT(MlirPortInfo, void);

#ifdef __cplusplus
}
#endif

MlirPass mlirCreateCSEPass(void) {
  return wrap(mlir::createCSEPass().release());
}

void mlirRegisterCSEPass(void) {
  mlir::registerCSEPass();
}

