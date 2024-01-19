get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(circt_dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(circt_conversion_libs GLOBAL PROPERTY CIRCT_CONVERSION_LIBS)

set(LIBSp
        MLIRSpecHLS.a
        MLIRTargetCustomCEmit.a
        libMLIRTransforms.a
        libMLIRCAPIIR.a
        libSpecHLSToComb.a
        libSpecHLSTransforms.a
        libSpecHLSCAPI.a
        libCIRCTCAPIFSM.a
        libCIRCTCAPIHW.a
        libMLIROptLib.a
)

set(LIBS
        ${mlir_dialect_libs}
        ${mlir_conversion_libs}
        MLIROptLib
        MLIRSpecHLS
        MLIRTargetCustomCEmit
        MLIRTransforms
        MLIRCAPIIR
        SpecHLSToComb
        SpecHLSTransforms
        SpecHLSCAPI
        CIRCTCAPIFSM
        CIRCTCAPIHW

        CIRCTCAPIHWArith

        CIRCTSeqTransforms
        CIRCTCAPISeq

        #        CIRCTComb
        CIRCTCAPIComb

        CIRCTSV

        CIRCTCAPISV

        MLIRCAPIFunc
        MLIRCAPIIR

        MLIRCAPISCF

        #        CIRCTDC
        CIRCTFSM #
        CIRCTCAPIFSM #
        CIRCTSupport
        CIRCTTransforms
        CIRCTTransforms
        CIRCTHWArithToHW

        CIRCTCAPISV
        CIRCTFSMToSV
        CIRCTHWToSV
        CIRCTSV
        CIRCTSVTransforms
        CIRCTSeqToSV
        CIRCTVerifToSV

        #        CIRCTSV
        #        CIRCTSVTransforms
        CIRCTSSP
        CIRCTSSPTransforms
        CIRCTScheduling
        CIRCTSchedulingAnalysis
        #        CIRCTSeqTransforms
        #
        #        CIRCTHWArith
        MLIRCAPIArith
        #        MLIRCAPIAsync
        #        MLIRCAPIControlFlow
        #        MLIRCAPIConversion
        #        MLIRCAPIDebug
        #        MLIRCAPIExecutionEngine
        #        MLIRCAPIInterfaces
        #        MLIRCAPILLVM
        #        MLIRCAPIMLProgram
        #        MLIRCAPIPDL
        #        MLIRCAPIQuant
        #        MLIRCAPIAMDGPU
        #        MLIRCAPIArith
        #        MLIRCAPIAsync
        MLIRCAPIControlFlow
        MLIRCAPIConversion
        MLIRCAPIDebug
        MLIRCAPIExecutionEngine
        MLIRCAPIFunc
        #        MLIRCAPIGPU
        MLIRCAPIIR
        MLIRCAPIInterfaces
        #        MLIRCAPILLVM
        #        MLIRCAPILinalg
        #        MLIRCAPIMLProgram
        #        MLIRCAPIMath
        MLIRCAPIMemRef
        #        MLIRCAPINVGPU
        #        MLIRCAPINVVM
        #        MLIRCAPIOpenMP
        #        MLIRCAPIPDL
        #        MLIRCAPIQuant
        #        MLIRCAPIROCDL
        #        MLIRCAPIRegisterEverything
        #        MLIRCAPIShape
        #        MLIRCAPISparseTensor
        #        MLIRCAPITensor
        MLIRCAPITransformDialect
        MLIRCAPITransforms
        #        MLIRCAPIVector
        MLIRCAPISCF
)


add_library(gecosapi SHARED
        ExtraPassesCAPI.cpp
        SpecHLSGecosAPI.c
        SpecHLSGecosAPIPasses.c
        BuiltinAttributes.cpp
        BuiltinTypes.cpp

)
#../lib/Analysis/RecMIIAnalysis.cpp) )

#llvm_update_compile_flags(gecosapi)

target_link_libraries(gecosapi PUBLIC ${LIBS})
#target_link_libraries(gecosapi PUBLIC "-Wl,--whole-archive ${LIBSp} -Wl,--no-whole-archive"  ${LIBS})

install(TARGETS gecosapi DESTINATION GecosAPI)

set_target_properties(gecosapi PROPERTIES
        VERSION 1
        SOVERSION 1
        PUBLIC_HEADER include/CAPI/gecosapi.h)