add_mlir_library(SpecHLSScheduling
        Problems.cpp
        ASAPScheduler.cpp
        SchedulePass.cpp
        DEPENDS

        MLIRSpecHLSOpsIncGen
        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRSpecHLS
        MLIRTransforms
)
target_link_libraries(SpecHLSScheduling

        PUBLIC
        CIRCTComb
        CIRCTHW
        CIRCTSupport
)