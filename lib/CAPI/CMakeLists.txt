# TODO: Make the check source feature optional as an argument on *_add_library.
set(LLVM_OPTIONAL_SOURCES
        SpecHLSCAPI.cpp
        SSP.cpp
)

add_mlir_public_c_api_library(SpecHLSCAPI
        SpecHLSCAPI.cpp
        SSP.cpp
        LINK_LIBS PUBLIC
        MLIRSpecHLS
        SpecHLSScheduleDialect
        MLIR
)
