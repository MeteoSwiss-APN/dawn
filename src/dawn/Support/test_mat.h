#pragma once

#include <stdio.h>
#include <string>
#include <vector>

#include "dawn/SIR/SIR.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"

void mat_dbg_print_extents(std::shared_ptr<dawn::iir::StencilInstantiation>& target);