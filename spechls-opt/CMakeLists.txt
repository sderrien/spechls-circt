get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRSpecHLS
        SpecHLSToComb
        MLIRTargetCustomCEmit
        MLIRTargetCustomCEmit
        )
add_llvm_executable(spechls-opt spechls-opt.cpp)

llvm_update_compile_flags(spechls-opt)
target_link_libraries(spechls-opt PRIVATE ${LIBS})
