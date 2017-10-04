//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _                   
//                        | |     | |                  
//                    __ _| |_ ___| | __ _ _ __   __ _ 
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/ 
//
//
//  This file is distributed under the MIT License (MIT). 
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#pragma once

namespace gridtools {

    namespace clang {

        /**
         * @brief Halo extend (passed to gtclang by "-max-halo")
         * @ingroup gridtools_clang
         */
        struct halo {
#ifdef GRIDTOOLS_CLANG_HALO_EXTEND
            static constexpr int value = GRIDTOOLS_CLANG_HALO_EXTEND;
#else
            static constexpr int value = 3;
#endif
        };
    }
}
