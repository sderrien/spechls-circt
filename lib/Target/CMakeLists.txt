

add_llvm_library(MLIRTargetCustomCEmit
        ../Transforms/VitisExport/PrintOperations.cpp

        ADDITIONAL_HEADER_DIRS
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/Verilog
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/CustomC
        ${PROJECT_SOURCE_DIR}/include/Target/CustomC
        ${PROJECT_SOURCE_DIR}/include/Target/Verilog
        DEPENDS
        intrinsics_gen
        )
target_link_libraries(MLIRTargetCustomCEmit
        MLIRIR
        )