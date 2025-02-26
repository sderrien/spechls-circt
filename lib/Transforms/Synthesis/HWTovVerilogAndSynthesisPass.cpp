#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSTypes.h"
#include "Transforms/Passes.h"
#include "circt/Conversion/ExportVerilog.h"

#include "Dialect/SpecHLS/SpecHLSUtils.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <system_error>

namespace {
struct HWTovVerilogAndSynthesisPass
    : public mlir::PassWrapper<HWTovVerilogAndSynthesisPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<circt::hw::HWDialect, circt::comb::CombDialect, circt::sv::SVDialect, circt::seq::SeqDialect>();
  }

  void runOnOperation() override;

private:
  void generateVivadoTclScript(const std::string &verilogDir, const std::string &topModuleName, const std::string &tclFilePath);
  void callVivado(const std::string &tclFilePath);
};
} // namespace

void HWTovVerilogAndSynthesisPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::PassManager pm(module.getContext());

  // Lower HW to SV
  // Add SpecHLS to Comb and Seq passes
  pm.addPass(SpecHLS::createConvertSpecHLSToCombPass());
  pm.addPass(SpecHLS::createConvertSpecHLSToSeqPass());

  if (mlir::failed(pm.run(module))) {
    module.emitError() << "Failed to lower HW to SV";
    return signalPassFailure();
  }

  // Generate Verilog files using exportSplitVerilog
  std::string verilogDir = "verilog_output";
  if (llvm::sys::fs::create_directory(verilogDir)) {
    module.emitError() << "Failed to create Verilog output directory";
    return signalPassFailure();
  }

  if (mlir::failed(circt::exportSplitVerilog(module, verilogDir))) {
    module.emitError() << "Failed to export Verilog";
    return signalPassFailure();
  }

  // Generate Vivado TCL script
  std::string tclFilePath = "synthesize.tcl";
  std::string topModuleName = module.getName().str();
  generateVivadoTclScript(verilogDir, topModuleName, tclFilePath);

  // Call Vivado
  callVivado(tclFilePath);
}

void HWTovVerilogAndSynthesisPass::generateVivadoTclScript(const std::string &verilogDir, const std::string &topModuleName, const std::string &tclFilePath) {
  std::ofstream tclFile(tclFilePath);

  tclFile << "create_project -force project_name ./project_name -part xc7a35tcsg324-1\n";
  tclFile << "add_files [glob " << verilogDir << "/*.v]\n";
  tclFile << "set_property top " << topModuleName << " [current_fileset]\n";
  tclFile << "launch_runs synth_1 -jobs 4\n";
  tclFile << "wait_on_run synth_1\n";
  tclFile << "open_run synth_1 -name impl_1\n";
  tclFile << "launch_runs impl_1 -to_step write_bitstream\n";
  tclFile << "wait_on_run impl_1\n";
  tclFile << "write_checkpoint -force ./checkpoint.dcp\n";
  tclFile << "write_bitstream -force ./output.bit\n";

  tclFile.close();
}

void HWTovVerilogAndSynthesisPass::callVivado(const std::string &tclFilePath) {
  std::string command = "vivado -mode batch -source " + tclFilePath;
  int result = std::system(command.c_str());

  if (result != 0) {
    llvm::errs() << "Vivado synthesis failed with exit code " << result << "\n";
  } else {
    llvm::outs() << "Vivado synthesis completed successfully\n";
  }
}

std::unique_ptr<mlir::Pass> createHWTovVerilogAndSynthesisPass() {
  return std::make_unique<HWTovVerilogAndSynthesisPass>();
}

