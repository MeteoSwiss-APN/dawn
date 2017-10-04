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

#ifndef GRIDTOOLS_CLANG_BENCHMARKER_HPP
#define GRIDTOOLS_CLANG_BENCHMARKER_HPP

#ifdef GRIDTOOLS_CLANG_GENERATED
#include "gridtools/clang_dsl.hpp"
#include <cstdlib>
#include <iostream>
#endif

namespace gridtools {

    namespace clang {

#ifdef GRIDTOOLS_CLANG_GENERATED
        struct cache_flusher {
            cache_flusher(int n = 8388608) : m_n(n / 8) {
                m_a.resize(m_n);
                m_b.resize(m_n);
                m_c.resize(m_n);
            }

            void flush() {
                for (int i = 0; i < m_n; i++)
                    m_a[i] = m_b[i] * m_c[i];
            };

          private:
            const int m_n;
            std::vector< double > m_a;
            std::vector< double > m_b;
            std::vector< double > m_c;
        };

        class benchmarker {
          public:
            template < class StencilWrapperType >
            static void run(StencilWrapperType &stencil_wrapper, std::size_t tsteps) {
                cache_flusher flusher;
                auto gridtools_stencils = stencil_wrapper.get_stencils();
                bool timing_not_available = false;

                double total_time = 0;
                for (std::size_t i = 0; i < gridtools_stencils.size(); ++i) {
                    auto *stencil = gridtools_stencils[i];

                    stencil->ready();
                    stencil->steady();
                    stencil->run();
                    flusher.flush();

                    stencil->reset_meter();
                    for (std::size_t t = 0; t < tsteps; ++t) {
                        flusher.flush();
                        stencil->run();
                    }

                    double time = stencil->get_meter() / tsteps;
                    total_time += time;

                    timing_not_available |= time <= 0;
                }
                if (timing_not_available) {
                    std::cout << "NO_TIMES_AVAILABLE";
#if defined(GRIDTOOLS_CLANG_HOST) && !defined(_OPENMP)
                    std::cout << " - OpenMP required";
#endif
#ifndef GRIDTOOLS_CLANG_PERFORMANCE_METERS
                    std::cout << " - define GRIDTOOLS_CLANG_PERFORMANCE_METERS";
#endif
                    std::cout << std::endl;
                } else {
                    std::cout << "{\n";
                    std::cout << "  \"Name\" : \"" << stencil_wrapper.get_name() << "\",\n";
                    std::cout << "  \"Time\" : \"" << total_time << "\"\n";
                    std::cout << "}\n";
                }
            }
        };

        struct command_line {
            static std::array< unsigned int, 3 > parse_dimensions(int argc, char *argv[]) {
                auto error = [&]() {
                    std::cerr << "usage " << argv[0] << " dim_x dim_y dim_z" << std::endl;
                    std::exit(1);
                };

                decltype(parse_dimensions(argc, argv)) dims;
                int parsed_dims = 0;
                for (int i = 1; i < argc; ++i) {
                    if (argv[i][0] != '-')
                        dims[parsed_dims++] = std::atoi(argv[i]);
                    if (parsed_dims == 3)
                        break;
                }
                if (parsed_dims != 3)
                    error();

                return dims;
            }
        };

#else
        struct benchmarker {
            template < class StencilWrapperType >
            static void run(StencilWrapperType &&, std::size_t) {}
        };

        struct command_line {
            static std::array< unsigned int, 3 > parse_dimensions(int argc, char *argv[]) {
                return decltype(parse_dimensions(argc, argv)){};
            }
        };

#endif
    }
}

#endif
