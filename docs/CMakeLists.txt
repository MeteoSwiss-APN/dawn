##===------------------------------------------------------------------------------*- CMake -*-===##
##                          _ 
##                         | |
##                       __| | __ ___      ___ ___
##                      / _` |/ _` \ \ /\ / / '_  |
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

set(PROJECT_ROOT ${PROJECT_SOURCE_DIR})
set(SPHINX_ROOT ${CMAKE_CURRENT_LIST_DIR})

# Extract CMake documentation
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules-generate.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/cmake-modules-generate.cmake @ONLY)
add_custom_target(DawnDocsCMakeModules
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake-modules-generate.cmake
  COMMENT "Configuring dawn-cmake" VERBATIM
)

#
# HTML documentation
#
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in
               ${CMAKE_CURRENT_SOURCE_DIR}/conf.py @ONLY)
set(rst_files index.rst basics.rst)

# Run sphinx
add_custom_target(DawnDocsSphinxHtml
   COMMAND ${SPHINX_EXECUTABLE} -b html -d ${CMAKE_BINARY_DIR}/sphinx/doctrees 
           ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/html
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
   COMMENT "Generating sphinx HTML documentation" VERBATIM
   DEPENDS conf.py ${rst_files}
)
add_dependencies(DawnDocsSphinxHtml DawnDocsCMakeModules)

# Copy HTML
add_custom_target(docs
 COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/sphinx/html 
                                            ${CMAKE_SOURCE_DIR}/docs/html
 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
 COMMENT "Generating HTML documentation" VERBATIM
 DEPENDS DawnDocsSphinxHtml
)

#
# LaTeX documentation
#
if(DAWN_HAS_LATEX)

  # Run sphinx
  add_custom_target(DawnDocsSphinxLaTeX
     COMMAND ${SPHINX_EXECUTABLE} -b latex -d ${CMAKE_BINARY_DIR}/sphinx/doctrees 
             ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/latex
     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
     COMMENT "Generating sphinx LaTeX documentation" VERBATIM
     DEPENDS conf.py ${rst_files}
  )
  add_dependencies(DawnDocsSphinxLaTeX DawnDocsCMakeModules)

  # Run PdfLaTex
  add_custom_target(DawnDocsPdfLaTeX
   COMMAND ${PDFLATEX_COMPILER} -synctex=1 -interaction=nonstopmode 
           ${CMAKE_BINARY_DIR}/sphinx/latex/dawn.tex
   # Rerun to get cross-references right
   COMMAND ${PDFLATEX_COMPILER} -synctex=1 -interaction=nonstopmode 
           ${CMAKE_BINARY_DIR}/sphinx/latex/dawn.tex
   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/sphinx/latex
   COMMENT "Generating LaTeX (pdf) documentation" VERBATIM
   DEPENDS DawnDocsSphinxLaTeX
  )

  # Copy pdf
  add_custom_target(latex
   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/sphinx/latex/dawn.pdf 
                                                 ${CMAKE_SOURCE_DIR}/docs/latex/dawn.pdf
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
   COMMENT "Generating LaTeX (pdf) documentation" VERBATIM
   DEPENDS  DawnDocsPdfLaTeX
  )
endif()

# Deploy documentation
add_custom_target(deploy-docs
  COMMAND ${BASH_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/gh-pages/deploy.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Deploying documentation" VERBATIM
)

add_dependencies(deploy-docs docs)
if(DAWN_HAS_LATEX)
  add_dependencies(deploy-docs latex)
endif()