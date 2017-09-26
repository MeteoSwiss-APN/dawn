file(REMOVE_RECURSE
  "libDawn.pdb"
  "libDawn.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/DawnStatic.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
