if(NOT DEFINED GIT_EXECUTABLE)
  find_program(GIT_EXECUTABLE
    NAMES git
    DOC "Path to git executable"
  )
  mark_as_advanced(GIT_EXECUTABLE)
endif()

function(get_git_version_hash HASH_OUT_VAR_NAME)
    if(GIT_EXECUTABLE)
        execute_process(COMMAND git rev-parse --short HEAD OUTPUT_VARIABLE __hash OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    set(${HASH_OUT_VAR_NAME} ${__hash} PARENT_SCOPE)
endfunction()
