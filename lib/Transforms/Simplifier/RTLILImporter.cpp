#include "RTLILImporter.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Namespace.h"
#include "kernel/rtlil.h"                  // from @at_clifford_yosys
#include "mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h" // from @llvm-project
#include "mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/Transforms/FoldUtils.h"     // from @llvm-project
#include "llvm/ADT/StringMap.h"            // from @llvm-project
#include "llvm/ADT/MapVector.h"            // from @llvm-project
#include "llvm/Support/FormatVariadic.h"   // from @llvm-project
namespace mlir {

using ::Yosys::RTLIL::Module;
using ::Yosys::RTLIL::SigSpec;
using ::Yosys::RTLIL::Wire;
using ::Yosys::RTLIL::Cell;

#define VERBOSE false
namespace {

// getTypeForWire gets the MLIR type corresponding to the RTLIL wire. If the
// wire is an integer with multiple bits, then the MLIR type is a tensor of
// bits.
Type getTypeForWire(OpBuilder &b, Wire *wire) {
  return b.getIntegerType(wire->width);
}

} // namespace

llvm::SmallVector<Yosys::RTLIL::IdString, 10> getTopologicalOrder(std::stringstream &torderOutput) {
  llvm::SmallVector<Yosys::RTLIL::IdString, 10> cells;
  std::string line;
  while (std::getline(torderOutput, line)) {
    if (VERBOSE) llvm::errs() << "Analyzing line " << line << "\n";
    auto lineCell = line.find("cell ");
    if (lineCell != std::string::npos) {
      auto substr = line.substr(lineCell + 5, std::string::npos);
      auto id = Yosys::RTLIL::IdString(substr);
      if (VERBOSE) llvm::errs() << "Extracted cell " << substr << "\n";
      cells.push_back(id);
    }
  }
  if (VERBOSE) llvm::errs() << "Extracted  " << cells.size() << "cells \n";
  return cells;
}

mlir::Operation *
RTLILImporter::createOp(Cell *cell,
                        llvm::SmallVector<mlir::Value, 4> &inputs,
                        circt::ImplicitLocOpBuilder &b) {

  if (VERBOSE)
    llvm::outs() << "Cell " << cell << ":" << cell->type.str() << "\n";
  // assert(cell->type.begins_with("$lut"));

  // Create truth table from cell attributes.

  if (cell->type.begins_with("$lut")) {
    int lutBits;
    llvm::StringRef(cell->type.substr(4, 1)).getAsInteger(10, lutBits);
    uint64_t lutValue = 0;
    int lutSize = 1 << lutBits;
    for (int i = 0; i < lutSize; i++) {
      auto lutStr =
          cell->getPort(Yosys::RTLIL::IdString(llvm::formatv("\\P{0}", i)));
      lutValue |= (lutStr.as_bool() ? 1 : 0) << i;
    }
    if (VERBOSE)
      llvm::outs() << "Extracting LUT value  ";
    auto lookupTable = b.getIntegerAttr(
        b.getIntegerType(lutSize, /*isSigned=*/false), lutValue);

    llvm::SmallVector<int, 1024> newcontent;
    circt::ArrayAttr attr = b.getI32ArrayAttr(newcontent);
    if (VERBOSE)
      llvm::outs() << "LUT  " << lookupTable << "   " << attr << "\n";

    return b.create<SpecHLS::LookUpTableOp>(b.getIntegerType(1), inputs.front(),
                                            attr);
  } else {
    auto type = cell->type.str();
    if (type == "$_AND_") {
      return b.create<circt::comb::AndOp>(b.getIntegerType(1), inputs);
    } else if (type == "$_OR_") {
      return b.create<circt::comb::OrOp>(b.getIntegerType(1), inputs);
    } else if (type == "$_XOR_") {
      return b.create<circt::comb::XorOp>(b.getIntegerType(1), inputs);
    } else if (type == "$_NOT_") {
      auto allOnes = b.create<circt::hw::ConstantOp>(b.getIntegerType(1), -1);
      llvm::SmallVector<circt::Value> args = {inputs[0], allOnes};
      return b.create<circt::comb::XorOp>(b.getIntegerType(1), args);
    } else if (type == "$and") {
      return b.create<circt::comb::AndOp>(b.getIntegerType(1), inputs);
    } else if (type == "$or") {
      return b.create<circt::comb::OrOp>(b.getIntegerType(1), inputs);
    } else if (type == "$xor") {
      return b.create<circt::comb::XorOp>(b.getIntegerType(1), inputs);
    } else if (type == "$not") {
      llvm::SmallVector<circt::Value> args = {inputs[0], inputs[1]};
      return b.create<circt::comb::XorOp>(b.getIntegerType(1), args);
    } else {
      llvm::errs() << "Error : unsupported cell type " << cell->type.str()
                   << "\n";
      throw "Error : unsupported cell type ";
    }
  }
}

llvm::SmallVector<Yosys::RTLIL::SigSpec, 4>

RTLILImporter::getInputs(Yosys::RTLIL::Cell *cell) {

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


Yosys::RTLIL::SigSpec RTLILImporter::getOutput(Yosys::RTLIL::Cell *cell) {
  return cell->getPort(Yosys::RTLIL::IdString("\\Y"));
}

void RTLILImporter::addWireValue(Wire *wire, Value value) {
  wireNameToValue[wire->name.str()] = value;
}

Value RTLILImporter::getWireValue(Wire *wire) {
  auto wireName = wire->name.str();
  if(wireNameToValue.contains(wireName)) {
    return wireNameToValue[wireName];
  } else {
    llvm::errs() << " No value attached to " << wireName << "\n";
    return NULL;
  }
}

Value RTLILImporter::getBit(
    const SigSpec &conn, ImplicitLocOpBuilder &b,
    llvm::MapVector<Wire *, SmallVector<Value>> &retBitValues) {
  // Because the cells are in topological order, and Yosys should have
  // removed redundant wire-wire mappings, the cell's inputs must be a bit
  // of an input wire, in the map of already defined wires (which are
  // bits), or a constant bit.
  if (!(conn.is_wire() || conn.is_fully_const() || conn.is_bit())) {
    if (VERBOSE)
      llvm::errs() << " connection " << conn.as_string() << "\n";
  }
  assert(conn.is_wire() || conn.is_fully_const() || conn.is_bit());
  if (conn.is_wire()) {
    auto name = conn.as_wire()->name.str();
    assert(wireNameToValue.contains(name));
    return wireNameToValue[name];
  }
  if (conn.is_fully_const()) {
    auto bit = conn.as_const();
    auto constantOp = b.createOrFold<circt::hw::ConstantOp>(
        b.getIntegerAttr(b.getIntegerType(1), bit.as_int()));
    return constantOp;
  }
  // Extract the bit of the multi-bit input or output wire.
  assert(conn.as_bit().is_wire());
  auto bit = conn.as_bit();
  if (retBitValues.contains(bit.wire)) {
    auto offset = retBitValues[bit.wire].size() - bit.offset - 1;
    return retBitValues[bit.wire][offset];
  }
  auto argA = getWireValue(bit.wire);
  auto extractOp = b.create<circt::comb::ExtractOp>(argA, bit.offset, 1);
  return extractOp;
}

void RTLILImporter::addResultBit(
    const SigSpec &conn, Value result,
    llvm::MapVector<Wire *, SmallVector<Value>> &retBitValues) {
  assert(conn.is_wire() || conn.is_bit());
  if (conn.is_wire()) {
    addWireValue(conn.as_wire(), result);
    return;
  }
  // This must be a bit of the multi-bit output wire.
  auto bit = conn.as_bit();
  assert(bit.is_wire() && retBitValues.contains(bit.wire));
  auto offset = retBitValues[bit.wire].size() - bit.offset - 1;
  retBitValues[bit.wire][offset] = result;
}

circt::hw::HWModuleOp
RTLILImporter::importModule(circt::hw::HWModuleOp op, Yosys::RTLIL::Design *design,
                            const SmallVector<Yosys::RTLIL::IdString, 10> &cellOrdering) {
  // Gather input and output wires of the module to match up with the block
  // arguments.
  SmallVector<Type, 4> argTypes;
  SmallVector<std::string, 4> argNames;
  llvm::StringMap<Type> argMap;
  llvm::StringMap<Type> resMap;
  SmallVector<Wire *, 4> wireArgs;
  SmallVector<Type, 4> retTypes;
  SmallVector<Wire *, 4> wireRet;

  Module *module = design->top_module();

  if (VERBOSE)
    llvm::errs() << "Extracting cell Library info from : \n"
                 << module->name.str();
  // Iterate over all modules in the design (each module corresponds to a cell type in the library)
  for (auto &mod_it : design->modules_) {
    Module *module = mod_it.second;

    // Output the name of the cell (module)
    if (VERBOSE)
      llvm::errs() << "Loaded cell type: \n" << module->name.str();

    // Iterate over the ports of the cell
    if (VERBOSE)
      llvm::errs() << "Module " << module->name.str() << "\n";
    for (auto wire : module->wires()) {

      std::string direction;
      if (wire->port_input) {
        direction = "input";
      } else if (wire->port_output) {
        direction = "output";
      } else {
        direction = "unknown";
      }

      if (VERBOSE)
        llvm::errs() << "  Port: " << wire->name.str() << "  Direction "
                     << direction << "\n";
    }
  }
  OpBuilder builder(context);
  // Maintain a map from RTLIL output wires to the Values that comprise it
  // in order to reconstruct the multi-bit output.

  // Convert cells to Operations according to topological order.
  if (VERBOSE) {
    llvm::errs() << "Listing cells :\n";
    for (const auto &cell : module->cells()) {
        llvm::errs() << " - cell \"" << cell->name.str() << "\"\n";
    }
  }
  for (const auto &cellName : cellOrdering) {
    if (VERBOSE) llvm::errs() << " Search for cell \"" << cellName.str() << "\" in " << module->name.str() << "\n";
    auto count= module->cells_.count(cellName);
    if (VERBOSE)llvm::errs() << count << "  instances found \n";
  }
  if (VERBOSE) llvm::errs() << "  done ! \n";
  auto optName = module->name.str().replace(0, 1, "") + std::string("_opt");
  auto nameAttr = builder.getStringAttr(optName);
  SmallVector<circt::hw::PortInfo> ports;
  for (auto port : op.getPortList()) {
    circt::hw::PortInfo newPort = {builder.getStringAttr(port.getName()), port.type, port.dir};
    ports.push_back(newPort);
  }
  auto submodule = builder.create<circt::hw::HWModuleOp>(builder.getUnknownLoc(), nameAttr, ports);
  if (VERBOSE)llvm::errs() << "build empty module " << submodule << "\n";

  llvm::MapVector<Wire *, SmallVector<Value>> retBitValues;


  if (VERBOSE) llvm::errs() << "Filling wireMap.\n";

  llvm::StringMap<Wire*> wireMap;
  for (auto *wire: module->wires()) {
    auto key = wire->name.str();
    if (VERBOSE)llvm::errs() << "Adding "<< key << "-> "<< wire << "\n";
    if (!wireMap.contains(key)) {
      wireMap[key]=wire;
    } else {
      if (VERBOSE)llvm::errs() << "Key "<<key << " already there "<< wireMap[key] << "\n";

    }
  }

  if (VERBOSE)llvm::errs() << "Filling arg/res map\n";
  for (auto *wire : module->wires()) {
    // The RTLIL module may also have intermediate wires that are neither inputs
    // nor outputs.
    auto name= wire->name.str();
    auto type= getTypeForWire(builder, wire);
    if (VERBOSE) llvm::errs() << " wire  " << name << ":" << type << "\n";

    if (wire->port_input) {

      argTypes.push_back(type);
      argNames.push_back(name);
      argMap[name]=type;
      wireArgs.push_back(wire);
      if (VERBOSE)  llvm::errs() << "Input wire  " << name << ":" << type << "\n";

    } else if (wire->port_output) {
      retTypes.push_back(getTypeForWire(builder, wire));
      wireRet.push_back(wire);
      resMap[name]=type;
      if (VERBOSE) llvm::errs() << "Output wire  " << name << ":" << type << "\n";
      retBitValues[wire].resize(wire->width);
    }
  }



  mlir::Block *block = submodule.getBodyBlock();

  // Bind  RTLIL wires to the block arguments' Values.
  if (VERBOSE)llvm::errs() << "Bind  RTLIL wires to the block arguments' Values.\n";

  for (auto i = 0; i < op.getNumInputPorts(); i++) {
    std::string name = ("\\"+ op.getInputName(i)).str();
    if (wireMap.contains(name)) {
      auto wire = wireMap[name];
      if (wire!=NULL) {
        if (VERBOSE) llvm::errs() << " bind wire "<< wire->name.str() << " with blk arg/value  "<< block->getArgument(i) <<"\n";

        addWireValue(wire, block->getArgument(i));
      } else {
        if (VERBOSE) llvm::errs() << " Inconsistent I/O mapping for "<< name << " in "<< op.getName() <<"\n";
        return NULL;
      }
    } else {
      llvm::errs() << "Unknwon wire " << name << "\n";
    }
  }

  auto b = ImplicitLocOpBuilder::atBlockBegin(submodule.getLoc(), block);

//  for (const auto &celldef  :  module->cells_) {
//    auto cellId = celldef.first;
//    auto cell = celldef.second;
//    llvm::errs() << "Cell " << cell->name.str() << "  :  " << cell->type.str() << "\n";
//  }

// llvm::errs() << " ## creating cell library\n";

// llvm::StringMap<circt::hw::HWModuleExternOp*> extModuleMap;
// for (const auto &celldef  :  module->cells_) {
//    auto cellId = celldef.first;
//    auto cell = celldef.second;
//    const auto cellTypeName = cell->type.str();
//    SmallVector<circt::hw::PortInfo> portInfos;
//    llvm::errs() << "Cell  : " << cell->type.str() << "\n";
//    if (!extModuleMap.contains(cellTypeName)) {
//
//      // should me moved into separate utility function
//      llvm::errs() << "  Registering External module :" << cellTypeName <<"\n";
//
//      llvm::errs() << "  Extracting Input ports \n";
//      auto inputs = getInputs(cell);
//
//      for (const auto &wire : inputs) {
//        // 0 = input, 1 = output, 2 = inout (if supported)
//        auto portName = wire->name.str();
//        auto portType =  b.getIntegerType(wire->width);
//        llvm::errs() << "  - input port :" << portName << ":" <<  portType <<"\n";
//        circt::hw::PortInfo portInfo = {builder.getStringAttr(portName), portType, circt::hw::ModulePort::Direction::Input,{}};
//        portInfos.push_back(portInfo);
//      }
//      llvm::errs() << "  Extracting output ports \n";
//      for (const auto &wire : getOutputs(cell)) {
//        auto portName = wire->name.str();
//        auto portType =  b.getIntegerType(wire->width);
//        llvm::errs() << "  - output port :" << wire->name.str() << ":" <<  b.getIntegerType(wire->width) <<"\n";
//        circt::hw::PortInfo portInfo = {builder.getStringAttr(portName), portType, circt::hw::ModulePort::Direction::Output,{}};
//        portInfos.push_back(portInfo);
//      }

//      llvm::errs() << " Attempting to create HWModuleExternOp :" << cellTypeName <<"\n";
//      auto extModuleOp = b.create<circt::hw::HWModuleExternOp>(builder.getStringAttr(cellTypeName),portInfos);
//      llvm::errs() << " Created External op :" << extModuleOp <<"\n";
 //     extModuleMap[cellTypeName]=NULL;
//      llvm::errs() << " Registered " << cellTypeName <<"\n";
//    }
//  }

//  llvm::errs() << " ## creating op instances\n";
//  for (const auto &celldef  :  module->cells_) {
//    auto cellId = celldef.first;
//    auto cell = celldef.second;
//    const auto cellTypeName = cell->type.str();
//
//
//    if (extModuleMap.contains(cellTypeName)) {
//      llvm::errs() << "Found match for :" << cellTypeName <<"\n";
//
//      for (const auto &connection   :   cell->connections() ) {
//        llvm::errs() << "  - connection :" << connection.first.str() << ":" <<  connection.second.as_string() <<"\n";
//      }
//
//      auto externalHWmodule = extModuleMap[cellTypeName];
//      llvm::errs() << " External op :" << *externalHWmodule <<"\n";
//
//      SmallVector<Value, 4> inputValues;
//      llvm::errs() << "  Binding input wire/values  \n";
//      for (const auto sig : getInputs(cell)) {
//        auto value = getBit(sig, b, retBitValues);
//        llvm::errs() << "  - input sig :" << sig.get_hash() << "->" <<  value <<":"<< value.getType() <<"\n";
//        inputValues.push_back(value);
//      }
//
//      // Operation*module, StringAttr name, ArrayRef<Value> inputs, ArrayAttr parameters = {}, InnerSymAttr innerSym = {});
//      auto instance = b.create<circt::hw::InstanceOp>(externalHWmodule->getOperation(),b.getStringAttr(cellId.str()),inputValues);
//      if (VERBOSE) llvm::errs() << "instance op created " << *instance << "\n";
//      auto *op = createOp(cell, inputValues, b);
//      if (VERBOSE) llvm::errs() << "op created " << *op << "\n";
//
//      // multi-output nodes not supported
//      auto value = op->getResult(0);
//      addResultBit(getOutput(cell), value, retBitValues);
//
//      for (const auto &param   :   cell->parameters ) {
//        auto value = param.second;
//        llvm::errs() << " - param : " << param.first.str() << "->" << value.as_string() <<"\n";
//        op->setAttr(param.first.str(),builder.getStringAttr(value.as_string()));
//      }
//    }
//
//  }

  // Convert cells to Operations according to topological order.
  for (const auto &cellName : cellOrdering) {
    if (VERBOSE)  llvm::errs() << " Searching for instances of " << cellName.str() << " in "<< module->name.str()<<"\n";

    auto cells = module->cells_;
//    int count =0;
//    for (auto e : cells) {
//      if (e.first==cellName) {
//        count++;
//      }
//    }
    auto count= cells.count(cellName);
    if (VERBOSE) llvm::errs() << count << "  instances found \n";

    if (module->cells_.count(cellName)) {

    }
    if (count != 0) {
      auto *cell = module->cells_[cellName];

      SmallVector<Value, 4> inputValues;

      for (const auto &conn : getInputs(cell)) {
          auto bit = getBit(conn, b, retBitValues);
          inputValues.push_back(bit);
      }

      auto *op = createOp(cell, inputValues, b);
      if (VERBOSE)
        llvm::errs() << "op created " << *op << "\n";

        addResultBit(getOutput(cell), op->getResult(0), retBitValues);

    } else {
      if (VERBOSE) llvm::errs() << "No cell in optimized design !\n";
    }
  }

  // Wire up remaining connections.
  for (const auto &conn : module->connections()) {
    auto output = conn.first;
    // These must be output wire connections (either an output bit or a bit of a
    // multi-bit output wire).
    if (!(output.is_wire() || output.as_chunk().is_wire() ||
           output.as_bit().is_wire())) {
      throw "connection error";
    }
    if ((output.is_chunk() && !output.is_wire()) ||
        ((conn.second.is_chunk() && !conn.second.is_wire()) ||
         conn.second.chunks().size() > 1)) {
      // If one of the RHS or LHS is a chunk of a wire (and not a whole wire) OR
      // contains multiple chunks, then iterate bit by bit to assign the result
      // bits.
      for (auto i = 0; i < output.size(); i++) {
        Value connValue = getBit(conn.second.bits().at(i), b, retBitValues);
        addResultBit(output.bits().at(i), connValue, retBitValues);
      }
    } else {
      // This may be a single bit, a chunk of a wire, or a whole wire.
      Value connValue = getBit(conn.second, b, retBitValues);
      addResultBit(output, connValue, retBitValues);
    }
  }

  // Concatenate result bits if needed, and return result.
  SmallVector<Value, 4> returnValues;
  for (const auto &[resultWire, retBits] : retBitValues) {
    // If we are returning a whole wire as is (e.g. the input wire) or a single
    // bit, we do not need to concat any return bits.
    if (wireNameToValue.contains(resultWire->name.str())) {
      returnValues.push_back(getWireValue(resultWire));
    } else {
      // We are in a multi-bit scenario.
      assert(retBits.size() > 1);
      auto concatOp = b.create<circt::comb::ConcatOp>(retBits);
      returnValues.push_back(concatOp.getResult());
    }
  }

  circt::hw::OutputOp outOp = cast<circt::hw::OutputOp>(block->getTerminator());

  for (auto retVal : llvm::enumerate(returnValues)) {
    // if (VERBOSE) llvm::outs() << "out_" + std::to_string(retVal.index()) << "
    // " << retVal.value() << "\n";
    outOp->insertOperands(retVal.index(), retVal.value());
  }
  return submodule;
}

} // namespace mlir
