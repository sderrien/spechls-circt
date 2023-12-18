#ifndef HEIR_LIB_TRANSFORMS_YOSYSOPTIMIZER_RTLILIMPORTER_H_
#define HEIR_LIB_TRANSFORMS_YOSYSOPTIMIZER_RTLILIMPORTER_H_

#include <kernel/rtlil.h> // from @at_clifford_yosys

#include "mlir/Dialect/Func/IR/FuncOps.h" // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h" // from @llvm-project
#include "mlir/IR/MLIRContext.h"          // from @llvm-project
#include "mlir/IR/Operation.h"            // from @llvm-project
#include "mlir/IR/Value.h"                // from @llvm-project
#include "mlir/Support/LLVM.h"            // from @llvm-project
#include "llvm/ADT/MapVector.h"           // from @llvm-project
#include "llvm/ADT/StringMap.h"           // from @llvm-project

namespace mlir {
// namespace heir {

// Returns a list of cell names that are topologically ordered using the Yosys
// toder output. This is extracted from the lines containing cells in the
// output:
// -- Running command `torder -stop * P*;' --

// 14. Executing TORDER pass (print cells in topological order).
// module test_add
//   cell $abc$167$auto$blifparse.cc:525:parse_blif$168
//   cell $abc$167$auto$blifparse.cc:525:parse_blif$170
//   cell $abc$167$auto$blifparse.cc:525:parse_blif$169
//   cell $abc$167$auto$blifparse.cc:525:parse_blif$171
llvm::SmallVector<std::string, 10>
getTopologicalOrder(std::stringstream &torderOutput);

class RTLILImporter {
public:
  RTLILImporter(mlir::MLIRContext *context) : context(context) {}

  // importModule imports an RTLIL module to an MLIR function using the provided
  // config. cellOrdering is a topologically sorted list of cells that can be
  // used to sequentially create the MLIR representation.
  mlir::func::FuncOp
  importModule(Yosys::RTLIL::Module *module,
               const llvm::SmallVector<std::string, 10> &cellOrdering);

protected:
  // cellToOp converts an RTLIL cell to an MLIR operation.
  virtual mlir::Operation *createOp(Yosys::RTLIL::Cell *cell,
                                    llvm::SmallVector<mlir::Value, 4> &inputs,
                                    mlir::ImplicitLocOpBuilder &b) const = 0;

  // Returns a list of RTLIL cell inputs.
  virtual llvm::SmallVector<Yosys::RTLIL::SigSpec, 4>
  getInputs(Yosys::RTLIL::Cell *cell) const = 0;

  // Returns an RTLIL cell output.
  virtual Yosys::RTLIL::SigSpec getOutput(Yosys::RTLIL::Cell *cell) const = 0;

private:
  mlir::MLIRContext *context;

  llvm::StringMap<mlir::Value> wireNameToValue;
  mlir::Value getWireValue(Yosys::RTLIL::Wire *wire);
  void addWireValue(Yosys::RTLIL::Wire *wire, mlir::Value value);

  // getBit gets the MLIR Value corresponding to the given connection. This
  // assumes that the connection is a single bit.
  mlir::Value
  getBit(const Yosys::RTLIL::SigSpec &conn, mlir::ImplicitLocOpBuilder &b,
         llvm::MapVector<Yosys::RTLIL::Wire *, llvm::SmallVector<mlir::Value>>
             &retBitValues);

  // addResultBit assigns an mlir result to the result connection.
  void addResultBit(
      const Yosys::RTLIL::SigSpec &conn, mlir::Value result,
      llvm::MapVector<Yosys::RTLIL::Wire *, llvm::SmallVector<mlir::Value>>
          &retBitValues);
};

//}  // namespace heir
} // namespace mlir

#endif // HEIR_LIB_TRANSFORMS_YOSYSOPTIMIZER_RTLILIMPORTER_H_
