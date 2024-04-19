
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
DEFINE_CAPI_DECL(FactorGammaInputsPass)
DEFINE_CAPI_DECL(MergeGammasPass)
DEFINE_CAPI_DECL(EliminateRedundantGammaInputsPass)
DEFINE_CAPI_DECL(InlineModulesPass)

MlirPass mlirCreateConfigurationExcluderPass(void);
void mlirRegisterConfigurationExcluderPass(void);

MlirPass mlirCreateExportVitisHLS(void);
void mlirRegisterExportVitisHLS(void);

#ifdef __cplusplus
}
#endif

DEFINE_CAPI_IMPL(MergeLookUpTablesPass)
DEFINE_CAPI_IMPL(SchedulePass)
DEFINE_CAPI_IMPL(MobilityPass)
DEFINE_CAPI_IMPL(YosysOptimizerPass)
DEFINE_CAPI_IMPL(ExportVitisHLS)
DEFINE_CAPI_IMPL(GroupControlNodePass)
DEFINE_CAPI_IMPL(FactorGammaInputsPass)
DEFINE_CAPI_IMPL(MergeGammasPass)
DEFINE_CAPI_IMPL(EliminateRedundantGammaInputsPass)
DEFINE_CAPI_IMPL(InlineModulesPass)

#ifdef __cplusplus
extern "C" {
#endif

DEFINE_C_API_STRUCT(MlirPortInfo, void);

#ifdef __cplusplus
}
#endif

//
// bool mlirAttributeIsAArrayAttr(MlirAttribute attr) {
//  return llvm::isa<ArrayAttr>(unwrap(attr));
//}
//
// DEFINE_C_API_STRUCT(MlirTypeID, const void);

MlirPass mlirCreateConfigurationExcluderPass(void) {
  return wrap(SpecHLS::createConfigurationExcluderPass().release());
}

void mlirRegisterConfigurationExcluderPass(void) {
  SpecHLS::registerConfigurationExcluderPass();
}

// MlirPass mlirCreateControlOptimizer(void) {
//   return wrap(SpecHLS::createControlOptimizer().release());
// }

// DEFINE_C_API_PTR_METHODS(MlirPortInfo, circt::hw::PortInfo)
// DEFINE_C_API_METHODS(MlirPortInfo, circt::hw::PortInfo)

// size_t hwModuleOp_getNumPorts(MlirOperation op) {
//   auto hwOp = dyn_cast<circt::hw::HWModuleOp>(unwrap(op));
//   if (hwOp) {
//     return hwOp.getNumPorts();
//   } else {
//     return -1;
//   }
// }
//
// MlirType hwModuleOp_getPortsTypeAt(MlirOperation op, int i) {
//   auto hwOp = dyn_cast<circt::hw::HWModuleOp>(unwrap(op));
//   if (hwOp) {
//     return hwOp.getBodyBlock()->getArgument(i).getType();
//   } else {
//     return -1;
//   }
// }