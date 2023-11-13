file(REMOVE_RECURSE
  "../libMLIRSpecHLS.a"
  "../libMLIRSpecHLS.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRSpecHLS.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
