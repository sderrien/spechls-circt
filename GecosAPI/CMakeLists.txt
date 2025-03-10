add_library(gecosapi SHARED
        ExtraPassesCAPI.cpp
        SpecHLSGecosCAPI.cpp
        SpecHLSGecosPassesCAPI.cpp
        BuiltinAttributes.cpp
        BuiltinTypes.cpp
        ../lib/Transforms/Analysis/DataflowDominatorAnalysis.cpp
        ../lib/Transforms/Analysis/PostDataflowDominatorAnalysis.cpp
        ../lib/Transforms/SpecHLSToSeq/DelayConversion.cpp

)

target_link_libraries(gecosapi PUBLIC
        SpecHLSScheduleDialect
        SpecHLSTransforms
        SpecHLSCAPI
        CIRCT
        Yosys::Yosys
)

install(TARGETS gecosapi)
set_property(TARGET gecosapi PROPERTY INSTALL_RPATH "$ORIGIN/../lib")

set_target_properties(gecosapi PROPERTIES
        VERSION 1
        SOVERSION 1
        PUBLIC_HEADER include/CAPI/gecosapi.h)
