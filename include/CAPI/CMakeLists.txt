add_mlir_dialect(SpecHLSOps SpecHLS)
add_mlir_doc(SpecHLSDialect SpecHLSDialect CAPI/ -gen-dialect-doc)
add_mlir_doc(SpecHLSOps SpecHLSOps CAPI/ -gen-op-doc)
include_directories(${PROJECT_SOURCE_DIR}/include/CAPI/)