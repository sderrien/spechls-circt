get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        CIRCTExportVerilog
        CIRCTHWTransforms
        CIRCTOMTransforms
        CIRCTSeq
        CIRCTSVTransforms
        CIRCTTransforms
        CIRCTFirtool
        CIRCTTargetDebugInfo
        MLIROptLib
        MLIRSpecHLS
        SpecHLSToComb
        SpecHLSTransforms
        CIRCTSeq
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
        MLIRTargetCustomCEmit
        #       yosys
)
add_llvm_executable(circt2verilog circt2verilog.cpp)

llvm_update_compile_flags(circt2verilog)
target_link_libraries(circt2verilog PRIVATE ${LIBS})
target_compile_options(circt2verilog PUBLIC -fexceptions)


export_executable_symbols_for_plugins(circt2verilog)