add_llvm_executable(spechls-opt spechls-opt.cpp)

llvm_update_compile_flags(spechls-opt)

target_link_libraries(spechls-opt PRIVATE
        SpecHLSScheduleDialect
        SpecHLSTransforms
        MLIR
        CIRCT
        ${Z3_LIBRARIES}
        Yosys::Yosys
)
target_compile_options(spechls-opt PUBLIC -fexceptions)
install(TARGETS spechls-opt DESTINATION bin)
