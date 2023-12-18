
#include "LUTImporter.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "SpecHLS/SpecHLSOps.h"

#include <cassert>
#include <kernel/rtlil.h>                    // from @at_clifford_yosys
#include "llvm/ADT/ArrayRef.h"  // from @llvm-project
#include "llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/Support/LLVM.h"             // from @llvm-project

//namespace mlir {
//namespace heir {



mlir::Operation *LUTImporter::createOp(Yosys::RTLIL::Cell *cell,
                                       llvm::SmallVector<mlir::Value, 4> &inputs,
                                       circt::ImplicitLocOpBuilder &b) const {
  assert(cell->type.begins_with("\\lut"));

  // Create truth table from cell attributes.
  int lutBits;
  llvm::StringRef(cell->type.substr(4, 1)).getAsInteger(10, lutBits);

  uint64_t lutValue = 0;
  int lutSize = 1 << lutBits;
  for (int i = 0; i < lutSize; i++) {
    auto lutStr =
        cell->getPort(Yosys::RTLIL::IdString(llvm::formatv("\\P{0}", i)));
    lutValue |= (lutStr.as_bool() ? 1 : 0) << i;
  }

  auto lookupTable = b.getIntegerAttr(b.getIntegerType(lutSize, /*isSigned=*/false), lutValue);

  llvm::SmallVector<int, 1024> newcontent;
  circt::ArrayAttr attr = b.getI32ArrayAttr(newcontent);

  return b.create<SpecHLS::LookUpTableOp>(b.getIntegerType(1),inputs.front(), attr);
}

llvm::SmallVector<Yosys::RTLIL::SigSpec, 4> LUTImporter::getInputs(
    Yosys::RTLIL::Cell *cell) const {
  assert(cell->type.begins_with("\\lut") && "expected lut cells");

  // Return all non-P, non-Y named attributes.
  llvm::SmallVector<Yosys::RTLIL::SigSpec, 4> inputs;
  for (auto &conn : cell->connections()) {
    if (conn.first.contains("P") || conn.first.contains("Y")) {
      continue;
    }
    inputs.push_back(conn.second);
  }
  return inputs;
}

Yosys::RTLIL::SigSpec LUTImporter::getOutput(Yosys::RTLIL::Cell *cell) const {
  assert(cell->type.begins_with("\\lut"));
  return cell->getPort(Yosys::RTLIL::IdString("\\Y"));
}

//}  // namespace heir
//}  // namespace mlir