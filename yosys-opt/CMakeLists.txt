add_compile_definitions(_YOSYS_)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(circt_dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(circt_conversion_libs GLOBAL PROPERTY CIRCT_CONVERSION_LIBS)



set(LIBS
        gecosapi
#        ${mlir_dialect_libs}
#        ${mlir_conversion_libs}
#        CIRCTExportVerilog
#
#        MLIROptLib
#        MLIRTransforms
#        MLIRCAPIIR
#        SpecHLSTransforms
#        SpecHLSCAPI
#        CIRCTCAPIHW
#
#        CIRCTSeq
#        CIRCTComb
#        CIRCTCAPIComb
#        CIRCTCombTransforms
#        CIRCTSV
#
#        MLIRCAPIIR
#
#        #        CIRCTDC
#        CIRCTSupport
#        CIRCTTransforms
#        CIRCTHWArithToHW
#
#        CIRCTCAPISV
#        CIRCTFSMToSV
#        CIRCTHWToSV
#        CIRCTSV
#        CIRCTSVTransforms
#
#        MLIRCAPIControlFlow
#        MLIRCAPIConversion
#        MLIRCAPIDebug
#        MLIRCAPIExecutionEngine
#        MLIRCAPIFunc
#        MLIRCAPIIR
#        MLIRCAPIInterfaces
#        MLIRCAPIMemRef
#        MLIRCAPITransformDialect
#        MLIRCAPITransforms
        yosys
)

add_llvm_executable(yosys-opt yosys-opt.cpp)




llvm_update_compile_flags(yosys-opt)
set(CMAKE_BUILD_RPATH "/opt/spechls-opt/cmake-build-debug")
target_link_libraries(yosys-opt PRIVATE ${LIBS})
target_compile_options(yosys-opt PUBLIC -fexceptions)
install(TARGETS yosys-opt DESTINATION bin)