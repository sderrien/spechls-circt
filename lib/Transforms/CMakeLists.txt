
add_compile_definitions(_YOSYS_)


#include_directories("./YosysOptimizer")
#
#
#		YosysOptimizer.cpp
#
# RTLILImporter.cpp
add_mlir_library(SpecHLSTransforms
        Analysis/TransitiveClosure.cpp
        #        Analysis/GuardLogicAnalyzer.cpp
        Common/OutliningUtils.cpp
        Common/UnrollInstr.cpp
        Common/LongestPath.cpp
        Simplifier/YosysOptimizer.cpp
        Simplifier/RTLILImporter.cpp
        Simplifier/FactorGammaInputs.cpp
        Simplifier/EliminateRedundantGammaInputs.cpp
        Simplifier/InlineOptimizedModules.cpp
        Simplifier/MergeGammas.cpp
        Simplifier/MergeLUTs.cpp
        SpecHLSToComb/SpecHLSToComb.cpp
        SpecHLSToComb/SpecHLSLUTToComb.cpp
        SpecHLSToComb/GammaConversion.cpp
        SpecHLSToComb/RollbackConversion.cpp
        SpecHLSToComb/LUTConversion.cpp
        SpecHLSToSeq/SpecHLSToSeq.cpp
        SpecHLSToSeq/DelayConversion.cpp
        VitisExport/CFileContent.cpp
        VitisExport/PrintOperations.cpp
        VitisExport/PrintUtils.cpp
        VitisExport/ExportVitisHLS.cpp
        Simplifier/GroupControlLogic.cpp
        Scheduling/AutoPipeline.cpp
        Scheduling/MobilityPass.cpp
        Scheduling/ConfigurationExcluder.cpp
        Scheduling/Schedule.cpp
        DEPENDS

        MLIRSpecHLSOpsIncGen
        SpecHLSTransformsPassIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRSpecHLS
        SpecHLSScheduleDialect
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
