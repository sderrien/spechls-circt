
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(circt_dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(circt_conversion_libs GLOBAL PROPERTY CIRCT_CONVERSION_LIBS)



set(LIBS
        gecosapi
)

add_llvm_executable(dfg-merge dfg-merge.cpp)

llvm_update_compile_flags(dfg-merge)
set(CMAKE_BUILD_RPATH "/opt/spechls-opt/cmake-build-debug")
target_link_libraries(dfg-merge PRIVATE ${LIBS})
target_compile_options(dfg-merge PUBLIC -fexceptions)
install(TARGETS dfg-merge DESTINATION bin)