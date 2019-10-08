#!/bin/bash

# no_dependency "<of_module>" "<on_module>"
# checks that <of_module> does not depend on <on_module>
# i.e. <of_module> does not include any file from <on_module>
function no_dependency() {
    last_result=`grep -rE "#include .*$2/.*(hpp|h)" src/dawn/$1 | wc -l`
    if [ "$last_result" -gt 0 ]; then
        echo "ERROR Modularization violated: found dependency of $1 on $2"
        echo "`grep -rE "#include .*$2/.*(hpp|h)" src/dawn/$1`"
    fi
    modularization_result=$(( modularization_result || last_result ))
}

function are_independent() {
    no_dependency "$1" "$2"
    no_dependency "$2" "$1"
}
modularization_result=0

# # list of non-dependencies
no_dependency "Support" "SIR"
no_dependency "Support" "IIR"
no_dependency "Support" "CodeGen"
no_dependency "Support" "Serialization"
no_dependency "Support" "Optimizer"
no_dependency "Support" "Compiler"

no_dependency "SIR" "IIR"
no_dependency "SIR" "Optimizer"
no_dependency "SIR" "Compiler"
no_dependency "SIR" "Serialization"
no_dependency "SIR" "CodeGen"

no_dependency "IIR" "Optimizer"
no_dependency "IIR" "Compiler"
no_dependency "IIR" "Serialization"
no_dependency "IIR" "CodeGen"

no_dependency "Serialization" "Compiler"
no_dependency "Optimizer" "Compiler"
no_dependency "CodeGen" "Compiler"

are_independent "Serialization" "Optimizer"
are_independent "Optimizer" "CodeGen"
are_independent "Optimizer" "CodeGen"

exit $modularization_result
