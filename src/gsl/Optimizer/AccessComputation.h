//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_OPTIMIZER_ACCESSCOMPUTATION_H
#define GSL_OPTIMIZER_ACCESSCOMPUTATION_H

#include "gsl/Support/ArrayRef.h"
#include <memory>
#include <vector>

namespace gsl {

class StatementAccessesPair;
class StencilInstantiation;
class StencilFunctionInstantiation;

/// @name Access computation routines
/// @ingroup optimizer
/// @{

/// @fn computeAccesses
/// @brief Compute the Accesses of `statementAccessesPairs`
/// @ingroup optimizer
extern void
computeAccesses(StencilInstantiation* instantiation,
                ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs);

/// @fn computeAccesses
/// @brief Compute the caller and callee Accesses of `statementCallerAccessesPairs`
///
/// The caller Accesses will have the initial offset added (e.g if a stencil function is called with
/// `avg(u(i+1))` the initial offset of `u` is `[1, 0, 0]`) while the callee will not.
///
/// @see StencilFunctionInstantiation
/// @ingroup optimizer
extern void
computeAccesses(StencilFunctionInstantiation* stencilFunctionInstantiation,
                ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs);

/// @}

} // namespace gsl

#endif
