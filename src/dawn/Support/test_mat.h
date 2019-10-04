#pragma once

#include <stdio.h>
#include <string>
#include <vector>

#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/NonCopyable.h"

void mat_dbg_print_extents(std::shared_ptr<dawn::iir::StencilInstantiation>& target);