
#include <circt/Dialect/SSP/SSPPasses.h>

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

void mlirRegisterSchedulePass(void) {
  circt::ssp::registerSchedulePass();
}
