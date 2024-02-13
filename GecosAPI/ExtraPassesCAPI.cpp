
#include "circt/Dialect/SSP/SSPPasses.h"

#include "Transforms/Passes.h"
#include "mlir/CAPI/Pass.h"

using namespace circt;
#ifdef __cplusplus
extern "C" {
#endif

MlirPass mlirCreateSchedulePass(void);
void mlirRegisterSchedulePass(void);

MlirPass mlirCreateMobilityPass(void);
void mlirRegisterMobilityPass(void);

MlirPass mlirCreateExportVitisHLS(void);
void mlirRegisterExportVitisHLS(void);

#ifdef __cplusplus
}
#endif

MlirPass mlirCreateSchedulePass(void) {
  return wrap(circt::ssp::createSchedulePass().release());
}

void mlirRegisterSchedulePass(void) { circt::ssp::registerSchedulePass(); }

MlirPass mlirCreateMobilityPass(void) {
  return wrap(SpecHLS::createMobilityPass().release());
}

void mlirRegisterMobilityPass(void) { SpecHLS::registerMobilityPass(); }

// MlirPass mlirCreateControlOptimizer(void) {
//   return wrap(SpecHLS::createControlOptimizer().release());
// }

// MlirPass mlirCreateGroupControlNode(void) {
//   return wrap(SpecHLS::createGroupControlNodePass().release());
// }
//
MlirPass mlirCreateExportVitisHLS(void) {
  return wrap(SpecHLS::createExportVitisHLS().release());
}

void mlirExportVitisHLS(void) { SpecHLS::registerExportVitisHLS(); }
// void mlirGroupControlNode(void) { SpecHLS::registerGroupControlNodePass(); }
// void mlirControlOptimizer(void) { SpecHLS::registerControlOptimizer(); }

// mlir::Operation testpass(mlir::Operation op) {
//   {
//     auto ctx = op.getContext();
//     mlir::PassManager pm(ctx);
//     pm.addPass(std::move(SpecHLS::createMergeGammasPass()));
//     //    pm.addPass(std::move(createYYYPass()));
//     //    pm.addPass(std::move(createZZZPass()));
//     pm.run(&op); //
//   }
// }