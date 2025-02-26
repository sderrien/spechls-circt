get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(circt_dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(circt_conversion_libs GLOBAL PROPERTY CIRCT_CONVERSION_LIBS)


set(MLIR_LIBS
        ${mlir_dialect_libs}
        ${mlir_conversion_libs}
        MLIROptLib
        MLIRTransforms
        MLIRCAPIIR
        MLIRCAPIFunc
        MLIRCAPISCF
        MLIRCAPIArith
        MLIRCAPIControlFlow
        MLIRCAPIConversion
        MLIRCAPIDebug
        MLIRCAPIExecutionEngine
        MLIRCAPIFunc
        MLIRCAPIIR
        MLIRCAPIInterfaces
        MLIRCAPIMemRef
        MLIRCAPITransformDialect
        MLIRCAPITransforms

)
set(CIRCT_LIBS
        ${Z3_LIBRARIES}
        CIRCTLogicalEquivalence
        CIRCTExportVerilog

        CIRCTCAPIHW

        CIRCTCAPIHWArith

        CIRCTSeqTransforms
        CIRCTCAPISeq
        CIRCTComb
        CIRCTCAPIComb
        CIRCTCombTransforms
        CIRCTSV
        CIRCTCAPISV
        CIRCTFSM #
        CIRCTCAPIFSM #
        CIRCTSupport
        CIRCTTransforms
        CIRCTHWArithToHW

        CIRCTFSMToSV
        CIRCTHWToSV
        CIRCTSV
        CIRCTSVTransforms
        CIRCTSeqToSV
        CIRCTSeqTransforms
        CIRCTVerifToSV

        CIRCTSSP
        CIRCTSSPTransforms
        CIRCTScheduling
        CIRCTSchedulingAnalysis

)
set(SPECHLS_LIBS
        SpecHLSScheduleDialect
        SpecHLSTransforms
        SpecHLSCAPI
        MLIRSpecHLS
        SpecHLSScheduleDialect
        SpecHLSTransforms

)
set(YOSYSLIBS
        yosys
)



add_library(gecosapi SHARED
        ExtraPassesCAPI.cpp
        SpecHLSGecosCAPI.cpp
        SpecHLSGecosPassesCAPI.cpp
        SpecHLSTypeCAPI.cpp
        BuiltinAttributes.cpp
        BuiltinTypes.cpp
)

add_library(spechlsbase SHARED dummy.cpp)
add_library(mlirbase SHARED dummy.cpp)
add_library(circtbase SHARED dummy.cpp)



target_link_libraries(mlirbase PUBLIC -Wl,-force_load ${MLIR_LIBS})
target_link_libraries(circtbase PUBLIC -Wl,-force_load ${CIRCT_LIBS} mlirbase yosys)
target_link_libraries(spechlsbase PUBLIC -Wl,-force_load ${SPECHLS_LIBS} circtbase )

target_link_libraries(gecosapi PUBLIC spechlsbase)

install(TARGETS gecosapi DESTINATION lib)
install(TARGETS circtbase DESTINATION lib)
install(TARGETS mlirbase DESTINATION lib)
install(TARGETS spechlsbase DESTINATION lib)

set_target_properties(gecosapi PROPERTIES
        VERSION 1
        SOVERSION 1
        PUBLIC_HEADER include/CAPI/gecosapi.h)

set_target_properties(circtbase PROPERTIES
VERSION 1
SOVERSION 1
)

set_target_properties(mlirbase PROPERTIES
VERSION 1
SOVERSION 1
)

set_target_properties(spechlsbase PROPERTIES
VERSION 1
SOVERSION 1
)
