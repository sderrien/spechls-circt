get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${mlir_dialect_libs}
        ${mlir_conversion_libs}
        gecosapi
)
add_llvm_executable(test-gecos test-gecos.c)

llvm_update_compile_flags(test-gecos)
target_link_libraries(test-gecos PRIVATE ${LIBS})
target_compile_options(test-gecos PUBLIC -fexceptions)