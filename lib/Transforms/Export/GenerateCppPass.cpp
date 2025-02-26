//===- GenerateCppPass.cpp --------------------------------------*- C++ -*-===//
//
// This pass generates C++ code from CIRCT 'comb', 'seq', 'spechls', and 'fsm' dialects,
// incorporating requested improvements.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWDialect.h"         // Include HWDialect for PortInfo
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/FSM/FSMOps.h"           // Include FSM dialect definitions
#include "circt/Dialect/FSM/FSMTypes.h"


#include "llvm/ADT/TypeSwitch.h"                // For TypeSwitch
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <algorithm>
#include <set>

using namespace mlir;
using namespace circt;
using namespace SpecHLS;  // Assuming the namespace for SpecHLS is spechls
using namespace fsm;      // Assuming the namespace for FSM dialect is fsm

namespace {

struct GenerateCppPass : public PassWrapper<GenerateCppPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;

  // Helper functions
  void generateCppForModule(hw::HWModuleOp hwModule, const std::string &moduleName);
  void processFSMMachineOp(fsm::MachineOp machineOp, const std::string &fsmFileName);
  std::string getCppType(Type type);
  std::string getValueName(Value val);
  std::string getRegisterName(Operation *op);
  void translateOperation(Operation *op, std::ostringstream &sourceContent);

  // Member variables
  int tempVarCount = 0;
  std::map<Value, std::string> valueNameMap;
  std::map<std::string, Value> registerValueMap;
  std::map<std::string, Type> inputsMap;
  std::map<std::string, Type> outputsMap;
  std::map<std::string, Type> registersMap;

  // FSM related members
  std::set<std::string> fsmStates;
  std::string currentStateVar;
  std::string nextStateVar;
};

} // namespace

void GenerateCppPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Iterate over hw.module operations
  for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
    // Extract the module name
    std::string moduleName = hwModule.getName().str();

    // Generate code for the module
    generateCppForModule(hwModule, moduleName);
  }
}

void GenerateCppPass::generateCppForModule(hw::HWModuleOp hwModule, const std::string &moduleName) {
  // Create the output files
  std::ofstream headerFile(moduleName + ".h");
  std::ofstream sourceFile(moduleName + ".cpp");

  // Generate the header file content
  std::ostringstream headerContent;
  // Generate the source file content
  std::ostringstream sourceContent;

  // Header guard
  std::string guard = "_" + moduleName + "_H_";
  std::transform(guard.begin(), guard.end(), guard.begin(), ::toupper);
  headerContent << "#ifndef " << guard << "\n";
  headerContent << "#define " << guard << "\n\n";

  // Include necessary headers
  headerContent << "#include <cstdint>\n";
  headerContent << "#include <ap_int.h>\n"; // Include ap_int.h
  headerContent << "\n";

  // Collect module inputs and outputs using PortInfo
  hw::ModulePortInfo portInfo = hwModule.getPorts();

  // Inputs
  for (const auto &port : portInfo.inputs) {
    std::string portName = port.getName().str();
    Type portType = port.type;
    inputsMap[portName] = portType;
    // Map argument to its name
    valueNameMap[hwModule.getArgument(port.argNum)] = portName;
  }

  // Outputs
  for (const auto &port : portInfo.outputs) {
    std::string portName = port.getName().str();
    Type portType = port.type;
    outputsMap[portName] = portType;
  }

  // Collect registers (including from seq and SpecHLS dialects)
  hwModule.walk([&](Operation *op) {
    if (isa<seq::CompRegOp, spechls::MuOp, spechls::DelayOp>(op)) {
      std::string regName = getRegisterName(op);
      registersMap[regName] = op->getResult(0).getType();
      registerValueMap[regName] = op->getResult(0);
      valueNameMap[op->getResult(0)] = regName; // Map result to register name
    }
    // Handle FSM MachineOp separately
    else if (auto machineOp = dyn_cast<fsm::MachineOp>(op)) {
      // Process the FSM MachineOp
      std::string fsmFileName = machineOp.getName().str() + "_fsm.cpp";
      processFSMMachineOp(machineOp, fsmFileName);
    }
  });

  // Generate variable declarations in the header file
  // Inputs
  for (const auto &input : inputsMap) {
    headerContent << "extern " << getCppType(input.second) << " " << input.first << ";\n";
  }
  headerContent << "\n";
  // Outputs
  for (const auto &output : outputsMap) {
    headerContent << "extern " << getCppType(output.second) << " " << output.first << ";\n";
  }
  headerContent << "\n";
  // Registers (current and next states)
  for (const auto &reg : registersMap) {
    headerContent << "extern " << getCppType(reg.second) << " " << reg.first << "_curr;\n";
    headerContent << "extern " << getCppType(reg.second) << " " << reg.first << "_next;\n";
  }
  headerContent << "\n";

  // Declare functions in the header file
  headerContent << "void " << moduleName << "_next_comb();\n";
  headerContent << "void " << moduleName << "_clock_tick();\n";
  headerContent << "void " << moduleName << "_run();\n\n";

  // Close header guard
  headerContent << "#endif // " << guard << "\n";

  // Write the header file
  headerFile << headerContent.str();

  // Generate the source file content
  sourceContent << "#include \"" << moduleName << ".h\"\n";
  sourceContent << "#include <iostream>\n\n";

  // Define variables in the source file
  // Inputs
  for (const auto &input : inputsMap) {
    sourceContent << getCppType(input.second) << " " << input.first << " = {};\n";
  }
  sourceContent << "\n";
  // Outputs
  for (const auto &output : outputsMap) {
    sourceContent << getCppType(output.second) << " " << output.first << " = {};\n";
  }
  sourceContent << "\n";
  // Registers (current and next states)
  for (const auto &reg : registersMap) {
    sourceContent << getCppType(reg.second) << " " << reg.first << "_curr = {};\n";
    sourceContent << getCppType(reg.second) << " " << reg.first << "_next = {};\n";
  }
  sourceContent << "\n";

  // Implement 'next_comb' function
  sourceContent << "void " << moduleName << "_next_comb() {\n";

  // Reset temporary variable counter and value map
  tempVarCount = 0;
  // valueNameMap is already populated with inputs and registers

  // Implement the combinational logic by traversing the operations
  hwModule.walk([&](Operation *op) {
    if (isa<hw::HWModuleOp>(op))
      return;

    // Skip FSM MachineOp here, as it is handled separately
    if (isa<fsm::MachineOp>(op))
      return;

    translateOperation(op, sourceContent);
  });

  sourceContent << "}\n\n";

  // Implement 'clock_tick' function
  sourceContent << "void " << moduleName << "_clock_tick() {\n";
  for (const auto &reg : registersMap) {
    sourceContent << "  " << reg.first << "_curr = " << reg.first << "_next;\n";
  }
  sourceContent << "}\n\n";

  // Implement 'run' function
  sourceContent << "void " << moduleName << "_run() {\n";
  sourceContent << "  // Initialize inputs and registers as needed\n";
  sourceContent << "  // Simulation loop\n";
  sourceContent << "  for (int cycle = 0; cycle < 10; ++cycle) {\n";
  sourceContent << "    // TODO: Set inputs as needed\n";
  sourceContent << "    " << moduleName << "_next_comb();\n";
  sourceContent << "    " << moduleName << "_clock_tick();\n";
  sourceContent << "    // Display outputs\n";
  sourceContent << "    std::cout << \"Cycle \" << cycle << \": \";\n";
  for (const auto &output : outputsMap) {
    sourceContent << "    std::cout << \"" << output.first << "=\" << " << output.first << " << \" \";\n";
  }
  sourceContent << "    std::cout << std::endl;\n";
  sourceContent << "  }\n";
  sourceContent << "}\n\n";

  // Write the source file
  sourceFile << sourceContent.str();

  // Close files
  headerFile.close();
  sourceFile.close();
}

void GenerateCppPass::processFSMMachineOp(fsm::MachineOp machineOp, const std::string &fsmFileName) {
  // Open the FSM file for writing
  std::ofstream fsmFile(fsmFileName);
  std::ostringstream fsmContent;

  // Include necessary headers
  fsmContent << "#include <iostream>\n";
  fsmContent << "#include <string>\n\n";

  // Get the machine name
  std::string machineName = machineOp.getName().str();

  // Collect FSM states and transitions
  std::set<std::string> states;
  std::map<std::string, fsm::StateOp> stateOps;

  // Traverse the regions inside the machineOp
  for (auto &region : machineOp->getRegions()) {
    for (auto &block : region) {
      for (auto &op : block) {
        if (auto stateOp = dyn_cast<fsm::StateOp>(&op)) {
          std::string stateName = stateOp.getName().str();
          states.insert(stateName);
          stateOps[stateName] = stateOp;
        }
      }
    }
  }

  // Define FSM states as enum
  fsmContent << "enum class State {\n";
  for (const auto &state : states) {
    fsmContent << "  " << state << ",\n";
  }
  fsmContent << "};\n\n";

  // Define current and next state variables
  fsmContent << "State currentState = State::" << machineOp.getInitialState().str() << ";\n";
  fsmContent << "State nextState = State::" << machineOp.getInitialState().str() << ";\n\n";

  // Implement the FSM logic
  fsmContent << "void fsm_step() {\n";
  fsmContent << "  switch (currentState) {\n";
  for (const auto &stateName : states) {
    fsmContent << "    case State::" << stateName << ":\n";
    auto stateOp = stateOps[stateName];

    // Process transitions from this state
    for (auto &transitionOp : stateOp.getTransitions()) {
      std::string condition = "true"; // Default condition

      if (auto condValue = transitionOp.getCondition()) {
        // Translate the condition to C++ code
        // For simplicity, assume the condition is a variable already mapped
        condition = getOperandName(condValue);
      }

      // Get the successor state
      std::string successorState = transitionOp.getSuccessor().getName().str();

      // Generate the transition code
      fsmContent << "      if (" << condition << ") {\n";

      // Actions during transition
      fsmContent << "        // Actions during transition\n";
      std::ostringstream actions;
      for (auto &op : transitionOp.getActions().getOps()) {
        translateOperation(&op, actions);
      }
      fsmContent << actions.str();

      fsmContent << "        nextState = State::" << successorState << ";\n";
      fsmContent << "        break;\n";
      fsmContent << "      }\n";
    }

    // Default case
    fsmContent << "      break;\n";
  }
  fsmContent << "    default:\n";
  fsmContent << "      break;\n";
  fsmContent << "  }\n";
  fsmContent << "}\n\n";

  // Implement state update
  fsmContent << "void fsm_update() {\n";
  fsmContent << "  currentState = nextState;\n";
  fsmContent << "}\n\n";

  // Close the FSM file
  fsmFile << fsmContent.str();
  fsmFile.close();
}

std::string GenerateCppPass::getCppType(Type type) {
  std::string cppType = "int"; // Default type

  TypeSwitch<Type>(type)
      .Case<IntegerType>([&](IntegerType intType) {
        unsigned width = intType.getWidth();
        if (width == 1) {
          cppType = "bool";
        } else if (width == 8 || width == 16 || width == 32 || width == 64) {
          if (intType.isUnsigned())
            cppType = "uint" + std::to_string(width) + "_t";
          else
            cppType = "int" + std::to_string(width) + "_t";
        } else {
          cppType = "ap_int<" + std::to_string(width) + ">"; // Use ap_int for other widths
        }
      })
      .Case<MemRefType>([&](MemRefType memrefType) {
        // For memref types, represent them as arrays
        Type elementType = memrefType.getElementType();
        auto shape = memrefType.getShape();
        std::string elementTypeStr = getCppType(elementType);
        cppType = elementTypeStr;
        for (int64_t dim : shape) {
          cppType += "[" + std::to_string(dim) + "]";
        }
      })
      .Default([&](Type) {
        // Handle other types if needed
        cppType = "int"; // Default type
      });

  return cppType;
}

std::string GenerateCppPass::getRegisterName(Operation *op) {
  if (auto nameAttr = op->getAttrOfType<StringAttr>("name")) {
    return nameAttr.getValue().str();
  }
  return "reg" + std::to_string(tempVarCount++);
}

void GenerateCppPass::translateOperation(Operation *op, std::ostringstream &sourceContent) {
  Value result = op->getNumResults() > 0 ? op->getResult(0) : Value();
  std::string resultName;
  if (result)
    resultName = "tmp_" + std::to_string(tempVarCount++);
  std::string cppType = result ? getCppType(result.getType()) : "";

  auto getOperandName = [&](Value operand) -> std::string {
    if (valueNameMap.count(operand))
      return valueNameMap[operand];
    else {
      std::string name = "tmp_" + std::to_string(tempVarCount++);
      valueNameMap[operand] = name;
      return name;
    }
  };

  TypeSwitch<Operation *>(op)
      // Handle comb operations
      .Case<comb::AddOp, comb::SubOp, comb::MulOp, comb::DivUOp, comb::DivSOp,
            comb::ModUOp, comb::ModSOp, comb::AndOp, comb::OrOp, comb::XorOp,
            comb::ShlOp, comb::ShrUOp, comb::ShrSOp>([&](Operation *binOp) {
        // Handle binary operations using TypeSwitch
        std::string lhsName = getOperandName(binOp->getOperand(0));
        std::string rhsName = getOperandName(binOp->getOperand(1));
        const char *opSymbol = TypeSwitch<Operation *, const char *>(binOp)
                                   .Case<comb::AddOp>([] { return "+"; })
                                   .Case<comb::SubOp>([] { return "-"; })
                                   .Case<comb::MulOp>([] { return "*"; })
                                   .Case<comb::DivUOp, comb::DivSOp>([] { return "/"; })
                                   .Case<comb::ModUOp, comb::ModSOp>([] { return "%"; })
                                   .Case<comb::AndOp>([] { return "&"; })
                                   .Case<comb::OrOp>([] { return "|"; })
                                   .Case<comb::XorOp>([] { return "^"; })
                                   .Case<comb::ShlOp>([] { return "<<"; })
                                   .Case<comb::ShrUOp, comb::ShrSOp>([] { return ">>"; })
                                   .Default([] { return "/* unknown op */"; });
        sourceContent << "  " << cppType << " " << resultName << " = " << lhsName << " " << opSymbol << " " << rhsName << ";\n";
        valueNameMap[result] = resultName;
      })
      // Handle other operations (hw::ConstantOp, comb::MuxOp, etc.)
      // [Insert other operation handling code here]
      .Default([&](Operation *op) {
        // Handle other operations or emit error for unsupported ones
        op->emitError("Unsupported operation in code generation");
      });
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createGenerateCppPass() {
  return std::make_unique<GenerateCppPass>();
}

static PassRegistration<GenerateCppPass> pass("generate-cpp",
                                              "Generate C++ code from CIRCT comb, seq, spechls, and fsm dialects");

