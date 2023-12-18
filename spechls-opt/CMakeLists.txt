get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
link_directories(/usr/local/lib/yosys/ /opt/yosys/)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        CIRCTExportVerilog
        MLIROptLib
        MLIRSpecHLS
        SpecHLSToComb
        SpecHLSTransforms
        SpecHLSScheduling
        CIRCTSeq
        CIRCTComb
        CIRCTHW
        CIRCTDC
        CIRCTFIRRTL
        CIRCTHWArith
        CIRCTComb
        CIRCTSeq
        CIRCTSSP
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
        MLIRTargetCustomCEmit
        yosys
        )
add_llvm_executable(spechls-opt spechls-opt.cpp)
llvm_update_compile_flags(spechls-opt)

target_link_libraries(spechls-opt PRIVATE ${LIBS})

