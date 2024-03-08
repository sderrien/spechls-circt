
add_compile_definitions(_YOSYS_)


#include_directories("./YosysOptimizer")
#
#
#		YosysOptimizer.cpp
#
# RTLILImporter.cpp
add_mlir_library(SpecHLSTransforms
        Simplifier/YosysOptimizer.cpp
        Simplifier/RTLILImporter.cpp
        Simplifier/FactorGammaInputs.cpp
        Simplifier/InlineOptimizedModules.cpp
        Simplifier/MergeGammas.cpp
        Simplifier/MergeLUTs.cpp
        SpecHLSToComb/SpecHLSToComb.cpp
        SpecHLSToComb/SpecHLSLUTToComb.cpp
        SpecHLSToComb/GammaConversion.cpp
        SpecHLSToComb/LUTConversion.cpp
        VitisExport/CFileContent.cpp
        VitisExport/PrintOperations.cpp
        VitisExport/PrintUtils.cpp
        VitisExport/ExportVitisHLS.cpp
        Simplifier/GroupControlLogic.cpp
        Scheduling/AutoPipeline.cpp
        Scheduling/MobilityPass.cpp
        Scheduling/Schedule.cpp
        DEPENDS

        MLIRSpecHLSOpsIncGen
        SpecHLSTransformsPassIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRSpecHLS
        MLIRTransforms
        CIRCTExportVerilog
        CIRCTComb
        CIRCTHW
        CIRCTSSP
        CIRCTSupport
        CIRCTSSPTransforms
        CIRCTScheduling
)
target_compile_options(SpecHLSTransforms PUBLIC -fexceptions)