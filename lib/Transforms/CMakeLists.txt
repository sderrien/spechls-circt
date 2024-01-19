
add_compile_definitions(_YOSYS_)
#include_directories("/usr/share/yosys/include/")

#include_directories("./YosysOptimizer")
#
#
#		YosysOptimizer.cpp
#
# RTLILImporter.cpp
add_mlir_library(SpecHLSTransforms
		YosysOptimizer.cpp
		LUTImporter.cpp
		RTLILImporter.cpp
		FactorGammaInputs.cpp
		InlineOptimizedModules.cpp
		MergeGammas.cpp
		MergeLUTs.cpp
		GenerateC.cpp
		GroupControlLogic.cpp
		DEPENDS

		MLIRSpecHLSOpsIncGen
		SpecHLSTransformsPassIncGen

		LINK_COMPONENTS
		Core

		LINK_LIBS PUBLIC
		MLIRSpecHLS
		MLIRTransforms
		)
target_compile_options(SpecHLSTransforms PUBLIC -fexceptions)

target_link_libraries(SpecHLSTransforms

		PUBLIC
		CIRCTExportVerilog
		CIRCTComb
		CIRCTHW
		CIRCTSupport
		)