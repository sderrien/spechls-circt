
add_compile_definitions(_YOSYS_)


add_mlir_library(SpecHLSTransforms
        Analysis/TransitiveClosure.cpp
        Analysis/DefUSeSCC.cpp

        #        Analysis/GuardLogicAnalyzer.cpp
        Common/OutliningUtils.cpp
        Common/InliningUtils.cpp
        Common/UnrollInstr.cpp
        Common/LongestPath.cpp
        Simplifier/YosysOptimizer.cpp
        Simplifier/RTLILImporter.cpp
        Simplifier/FactorGammaInputs.cpp
        Simplifier/EliminateRedundantGammaInputs.cpp
        Simplifier/InlineOptimizedModules.cpp
        Simplifier/LowerGecosOpsToComb.cpp
        Simplifier/MergeGammas.cpp
        Simplifier/MergeLUTs.cpp
        Simplifier/SimplifyCasts.cpp
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
        Merging/DFGMergePass.cpp
        Merging/MaxWeightClique.cpp
        Simplifier/GroupControlLogic.cpp
        Simplifier/GroupGammaNodes.cpp
        Simplifier/OutlineSCCPass.cpp
        Simplifier/ExtractSpeculativeThreads.cpp
        Simplifier/WordLenghtPropagation.cpp
        Simplifier/TopologicalSort.cpp
        Scheduling/AutoPipeline.cpp
        Scheduling/MobilityPass.cpp
        Scheduling/ConfigurationExcluder.cpp
        Scheduling/Schedule.cpp
        Export/ExportUclid5Pass.cpp
        Export/ExportElkPass.cpp
        Export/ExportCpp.cpp

        DEPENDS

        MLIRSpecHLSOpsIncGen
        SpecHLSTransformsPassIncGen

        LINK_LIBS PUBLIC
        MLIR
        MLIRSpecHLS
)
target_compile_options(SpecHLSTransforms PUBLIC -fexceptions)

