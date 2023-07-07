

add_llvm_library(MLIRTargetCustomCEmit
        CustomC/TestPrintNestingPass.cpp
        #        CustomC/CustomCEmitter.cpp

        ADDITIONAL_HEADER_DIRS
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/CustomC
        ${PROJECT_SOURCE_DIR}/include/Target/CustomC
        DEPENDS
        intrinsics_gen
        )
target_link_libraries(MLIRTargetCustomCEmit
        MLIRIR
        )