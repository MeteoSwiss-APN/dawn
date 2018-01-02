// gtclang (0.0.1-a7a9177-x86_64-linux-gnu-5.4.0)
// based on LLVM/Clang (3.8.0), Dawn (0.0.1)
// Generated on 2018-01-02  00:53:20

#define GRIDTOOLS_CLANG_GENERATED 1
#ifndef BOOST_RESULT_OF_USE_TR1
 #define BOOST_RESULT_OF_USE_TR1 1
#endif
#ifndef BOOST_NO_CXX11_DECLTYPE
 #define BOOST_NO_CXX11_DECLTYPE 1
#endif
#ifndef FUSION_MAX_VECTOR_SIZE
 #define FUSION_MAX_VECTOR_SIZE 20
#endif
#ifndef FUSION_MAX_MAP_SIZE
 #define FUSION_MAX_MAP_SIZE 20
#endif
#ifndef BOOST_MPL_LIMIT_VECTOR_SIZE
 #define BOOST_MPL_LIMIT_VECTOR_SIZE 20
#endif
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

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

namespace gridtools {

struct globals : public gridtools::clang::globals_impl<globals> {
  using base_t = gridtools::clang::globals_impl<globals>;

  struct var_compiletime_adapter : public base_t::variable_adapter_impl<bool> {
    var_compiletime_adapter() : base_t::variable_adapter_impl<bool>(bool(true)) {}

    template <class ValueType>
    var_compiletime_adapter& operator=(ValueType&& value) {
      throw std::runtime_error("invalid assignment to constant variable 'var_compiletime'");
      return *this;
    }
  } var_compiletime;

  struct var_default_adapter : public base_t::variable_adapter_impl<int> {
    var_default_adapter() : base_t::variable_adapter_impl<int>(int(2)) {}

    template <class ValueType>
    var_default_adapter& operator=(ValueType&& value) {
      get_value() = value;
      return *this;
    }
  } var_default;

  struct var_runtime_adapter : public base_t::variable_adapter_impl<int> {
    var_runtime_adapter() : base_t::variable_adapter_impl<int>(int()) {}

    template <class ValueType>
    var_runtime_adapter& operator=(ValueType&& value) {
      get_value() = value;
      return *this;
    }
  } var_runtime;
};
template <>
globals* gridtools::clang::globals_impl<globals>::s_instance = nullptr;
}  // namespace gridtool
;

namespace gridtools {

class globals_stencil {
 public:
  struct stencil_0 {
    // Intervals
    using interval_start_0_end_0 = gridtools::interval<gridtools::level<0, -1>, gridtools::level<1, -1>>;
    using axis_stencil_0 = gridtools::interval<gridtools::level<0, -2>, gridtools::level<1, 1>>;
    using grid_stencil_0 = gridtools::grid<axis_stencil_0>;

    // Members
    std::shared_ptr<gridtools::stencil> m_stencil;
    std::unique_ptr<grid_stencil_0> m_grid;

    struct stage_0_0 {
      using out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using var_runtime = gridtools::global_accessor<2, gridtools::enumtype::inout>;
      using var_default = gridtools::global_accessor<3, gridtools::enumtype::inout>;
      using arg_list = boost::mpl::vector<out, in, var_runtime, var_default>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        if ((bool)true) {
          eval(out(0, 0, 0)) = ((eval(in(0, 0, 0)) + eval(var_runtime()).value) + eval(var_default()).value);
        }
      }
    };

    template <class S1, class S2>
    stencil_0(const gridtools::clang::domain& dom, S1& in, S2& out) {
      // Domain
      using p_in = gridtools::arg<0, S1>;
      using p_out = gridtools::arg<1, S2>;
      using p_var_runtime =
          gridtools::arg<2, typename std::decay<decltype(globals::get().var_runtime.as_global_parameter())>::type>;
      using p_var_default =
          gridtools::arg<3, typename std::decay<decltype(globals::get().var_default.as_global_parameter())>::type>;
      using domain_arg_list = boost::mpl::vector<p_in, p_out, p_var_runtime, p_var_default>;
      auto gt_domain = new gridtools::aggregator_type<domain_arg_list>(
          in, out, globals::get().var_runtime.as_global_parameter(), globals::get().var_default.as_global_parameter());

      // Grid
      unsigned int di[5] = {dom.iminus(), dom.iminus(), dom.iplus(), dom.isize() - 1 - dom.iplus(), dom.isize()};
      unsigned int dj[5] = {dom.jminus(), dom.jminus(), dom.jplus(), dom.jsize() - 1 - dom.jplus(), dom.jsize()};
      m_grid = std::unique_ptr<grid_stencil_0>(new grid_stencil_0(di, dj));
      m_grid->value_list[0] = dom.kminus();
      m_grid->value_list[1] = dom.ksize() == 0 ? 0 : dom.ksize() - dom.kplus() - 1;

      // Computation
      m_stencil = gridtools::make_computation<gridtools::clang::backend_t>(
          *gt_domain, *m_grid,
          gridtools::make_multistage(
              gridtools::enumtype::execute<gridtools::enumtype::forward /*parallel*/>(),
              gridtools::make_stage<stage_0_0>(p_out(), p_in(), p_var_runtime(), p_var_default())));
    }

    ~stencil_0() { m_stencil->finalize(); }

    gridtools::stencil* get_stencil() { return m_stencil.get(); }
  };

  // Members
  stencil_0 m_stencil_0;
  static constexpr const char* s_name = "globals_stencil";

 public:
  globals_stencil(const globals_stencil&) = delete;

  template <class S1, class S2>
  globals_stencil(const gridtools::clang::domain& dom, S1& in, S2& out) : m_stencil_0(dom, in, out) {
    static_assert(gridtools::is_data_store<S1>::value,
                  "argument 'in' is not a 'gridtools::data_store' (2nd argument invalid)");
    static_assert(gridtools::is_data_store<S2>::value,
                  "argument 'out' is not a 'gridtools::data_store' (3rd argument invalid)");
  }

  void make_steady() {
    m_stencil_0.get_stencil()->ready();
    m_stencil_0.get_stencil()->steady();
  }

  void run(bool make_steady = true) {
    if (make_steady) this->make_steady();
    m_stencil_0.get_stencil()->run();
  }

  std::vector<gridtools::stencil*> get_stencils() {
    return std::vector<gridtools::stencil*>({m_stencil_0.get_stencil()});
  }

  const char* get_name() const { return s_name; }
};
}  // namespace gridtool
;
