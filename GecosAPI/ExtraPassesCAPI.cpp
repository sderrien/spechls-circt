
#include <circt/Dialect/SSP/SSPPasses.h>

#include "Transforms/Passes.h"
#include "mlir/CAPI/Pass.h"

using namespace circt;
#ifdef __cplusplus
extern "C" {
#endif

MlirPass mlirCreateSchedulePass(void);
void mlirRegisterSchedulePass(void);

#ifdef __cplusplus
}
#endif

MlirPass mlirCreateSchedulePass(void) {
  return wrap(circt::ssp::createSchedulePass().release());
}

void mlirRegisterSchedulePass(void) { circt::ssp::registerSchedulePass(); }

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