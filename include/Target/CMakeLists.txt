add_mlir_dialect(SpecHLSOps SpecHLS)
add_mlir_doc(SpecHLSDialect SpecHLSDialect SpecHLS/ -gen-dialect-doc)
add_mlir_doc(SpecHLSOps SpecHLSOps SpecHLS/ -gen-op-doc)
include_directories(${PROJECT_SOURCE_DIR}/include/Target/CustomC)