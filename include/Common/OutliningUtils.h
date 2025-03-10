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


void getBackwardSlice(Operation &rootOp, SetVector<Operation *> &backwardSlice, SetVector<Value *> &inputs, function_ref<bool(Operation *)> filter) ;

hw::HWModuleOp  outlineSliceAsHwModule(Operation* hwmodule,
                                      SetVector<Operation *> &slice,
                                      SetVector<Value*> &inputs,
                                      SetVector<Value*> &outputs,
                                      Twine newName);

SpecHLS::HTaskOp outlineSliceAsHTask(Operation* op,
                                          SetVector<Operation *> &slice,
                                          SetVector<Value*> &inputs,
                                          SetVector<Value*> &outputs,
                                          Twine newName);
