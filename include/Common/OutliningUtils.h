#include "mlir/Pass/Pass.h"

#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SetVector.h"

#include <set>

using namespace mlir;
using namespace circt;

void getSliceInputs(mlir::SetVector<Operation *> &slice, SetVector<Value> &inputs);


void getBackwardSlice(Operation &rootOp,
                            SetVector<Operation *> &slice,
                       llvm::function_ref<bool(Operation *)> filter);

hw::HWModuleOp outlineSliceAsHwModule(hw::HWModuleOp hwmodule, Operation &root,
                                      SetVector<Operation *> &slice,
                                      SetVector<Value> &inputs,
                                      SetVector<Value> &outputs,
                                      Twine newName);