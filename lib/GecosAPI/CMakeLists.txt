get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRSpecHLS
        MLIRTargetCustomCEmit
        MLIRTransforms
        MLIRCAPIIR
        MLIRSpecHLS
        SpecHLSToComb
        SpecHLSTransforms
        SpecHLSCAPI
        CIRCTCAPIFSM
        CIRCTCAPIHW
        CIRCTCAPIHWArith
        CIRCTCAPIComb
        CIRCTDC
        CIRCTFSM
        CIRCTHW
        CIRCTSupport
        CIRCTTransforms
        CIRCTHWArithToHW
        CIRCTSV
        CIRCTSVTransforms
        CIRCTScheduling
        CIRCTSchedulingAnalysis
        CIRCTSeqTransforms
        CIRCTSeq
        CIRCTHWArith
        MLIRCAPIArith
        MLIRCAPIAsync
        MLIRCAPIControlFlow
        MLIRCAPIConversion
        MLIRCAPIDebug
        MLIRCAPIExecutionEngine
        MLIRCAPIFunc
        MLIRCAPIIR
        MLIRCAPIInterfaces
        MLIRCAPILLVM
        MLIRCAPIMLProgram
        MLIRCAPIPDL
        MLIRCAPIQuant

        MLIRCAPIAMDGPU
        MLIRCAPIArith
        MLIRCAPIAsync
        MLIRCAPIControlFlow
        MLIRCAPIConversion
        MLIRCAPIDebug
        MLIRCAPIExecutionEngine
        MLIRCAPIFunc
        MLIRCAPIGPU
        MLIRCAPIIR
        MLIRCAPIInterfaces
        MLIRCAPILLVM
        MLIRCAPILinalg
        MLIRCAPIMLProgram
        MLIRCAPIMath
        MLIRCAPIMemRef
        MLIRCAPINVGPU
        MLIRCAPINVVM
        MLIRCAPIOpenMP
        MLIRCAPIPDL
        MLIRCAPIQuant
        MLIRCAPIROCDL
        MLIRCAPIRegisterEverything
        MLIRCAPISCF
        MLIRCAPIShape
        MLIRCAPISparseTensor
        MLIRCAPITensor
        MLIRCAPITransformDialect
        MLIRCAPITransforms
        MLIRCAPIVector

        MLIRCAPISCF
        CIRCTComb
        CIRCTHW
        CIRCTDC
        CIRCTFIRRTL
        CIRCTHWArith
        CIRCTComb
        CIRCTSeq
        CIRCTCombTransforms
        CIRCTFIRRTL
        CIRCTFSM
        CIRCTHW
        CIRCTHWArith
        CIRCTScheduling
        CIRCTHWArith
        CIRCTSystemC
        CIRCTTransforms
        CIRCTSupport

        )


add_library(gecosapi SHARED
        SpecHLSGecosAPI.c
        )

llvm_update_compile_flags(gecosapi)
target_link_libraries(gecosapi PRIVATE ${LIBS})
target_link_directories(gecosapi PRIVATE "/opt/circt/build/clion/lib")


set_target_properties(gecosapi PROPERTIES
        VERSION 1
        SOVERSION 1
        PUBLIC_HEADER include/CAPI/gecosapi.h)


