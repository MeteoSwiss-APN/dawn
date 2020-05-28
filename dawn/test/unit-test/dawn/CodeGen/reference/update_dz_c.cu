#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CUDA
#ifndef BOOST_RESULT_OF_USE_TR1
 #define BOOST_RESULT_OF_USE_TR1 1
#endif
#ifndef BOOST_NO_CXX11_DECLTYPE
 #define BOOST_NO_CXX11_DECLTYPE 1
#endif
#ifndef GRIDTOOLS_DAWN_HALO_EXTENT
 #define GRIDTOOLS_DAWN_HALO_EXTENT 3
#endif
#ifndef BOOST_PP_VARIADICS
 #define BOOST_PP_VARIADICS 1
#endif
#ifndef BOOST_FUSION_DONT_USE_PREPROCESSED_FILES
 #define BOOST_FUSION_DONT_USE_PREPROCESSED_FILES 1
#endif
#ifndef BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
 #define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS 1
#endif
#ifndef GT_VECTOR_LIMIT_SIZE
 #define GT_VECTOR_LIMIT_SIZE 30
#endif
#ifndef BOOST_FUSION_INVOKE_MAX_ARITY
 #define BOOST_FUSION_INVOKE_MAX_ARITY GT_VECTOR_LIMIT_SIZE
#endif
#ifndef FUSION_MAX_VECTOR_SIZE
 #define FUSION_MAX_VECTOR_SIZE GT_VECTOR_LIMIT_SIZE
#endif
#ifndef FUSION_MAX_MAP_SIZE
 #define FUSION_MAX_MAP_SIZE GT_VECTOR_LIMIT_SIZE
#endif
#ifndef BOOST_MPL_LIMIT_VECTOR_SIZE
 #define BOOST_MPL_LIMIT_VECTOR_SIZE GT_VECTOR_LIMIT_SIZE
#endif
#include <driver-includes/gridtools_includes.hpp>
using namespace gridtools::dawn;
namespace dawn_generated{
namespace cuda{

struct globals {
  double dt;

  globals() : dt(0){
  }
};
} // namespace cuda
} // namespace dawn_generated


namespace dawn_generated{
namespace cuda{
template<typename TmpStorage>__global__ void __launch_bounds__(192)  update_dz_c_stencil443_ms653_kernel(globals globals_, const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, const int tmpBeginIIndex, const int tmpBeginJIndex, const int jstride_tmp, const int kstride_tmp, ::dawn::float_type * const dp_ref, ::dawn::float_type * const ut, ::dawn::float_type * const vt, gridtools::data_view<TmpStorage>xfx_dv, gridtools::data_view<TmpStorage>yfx_dv) {

  // Start kernel
  ::dawn::float_type* xfx = &xfx_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0);
  ::dawn::float_type* yfx = &yfx_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0);
  const unsigned int nx = isize;
  const unsigned int ny = jsize;
  const int block_size_i = (blockIdx.x + 1) * 32 < nx ? 32 : nx - blockIdx.x * 32;
  const int block_size_j = (blockIdx.y + 1) * 4 < ny ? 4 : ny - blockIdx.y * 4;

  // computing the global position in the physical domain

  // In a typical cuda block we have the following regions

  // aa bbbbbbbb cc

  // aa bbbbbbbb cc

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // ee ffffffff gg

  // ee ffffffff gg

  // Regions b,d,f have warp (or multiple of warp size)

  // Size of regions a, c, h, i, e, g are determined by max_extent_t

  // Regions b,d,f are easily executed by dedicated warps (one warp for each line)

  // Regions (a,h,e) and (c,i,g) are executed by two specialized warp
  int iblock = 0 - 1;
  int jblock = 0 - 1;
if(threadIdx.y < +5) {
    iblock = threadIdx.x;
    jblock = (int)threadIdx.y + 0;
}else if(threadIdx.y < 6) {
    iblock = threadIdx.x % 1 + 32;
    jblock = (int)threadIdx.x / 1+0;
}
  // initialized iterators
  int idx111 = (blockIdx.x*32+iblock)*1+(blockIdx.y*4+jblock)*stride_111_1;
  int idx_tmp = (iblock+0)*1 + (jblock+0)*jstride_tmp;

  // jump iterators to match the intersection of beginning of next interval and the parallel execution block
  idx111 += max(0, blockIdx.z * 4) * stride_111_2;

  // jump tmp iterators to match the intersection of beginning of next interval and the parallel execution block
  idx_tmp += max(0, blockIdx.z * 4) * kstride_tmp;
  int kleg_lower_bound = max(0,blockIdx.z*4);
  int kleg_upper_bound = min(1,(blockIdx.z+1)*4-1);;
for(int k = kleg_lower_bound+0; k <= kleg_upper_bound+0; ++k) {
  if(iblock >= 0 && iblock <= block_size_i -1 + 1 && jblock >= 0 && jblock <= block_size_j -1 + 1) {
::dawn::float_type __local_ratio__d15_16_429 = (__ldg(&(dp_ref[idx111])) / (__ldg(&(dp_ref[idx111])) + __ldg(&(dp_ref[idx111+stride_111_2*1]))));
xfx[idx_tmp] = (__ldg(&(ut[idx111])) + ((__ldg(&(ut[idx111])) - __ldg(&(ut[idx111+stride_111_2*1]))) * __local_ratio__d15_16_429));
::dawn::float_type __local_ratio__d15_17_431 = (__ldg(&(dp_ref[idx111])) / (__ldg(&(dp_ref[idx111])) + __ldg(&(dp_ref[idx111+stride_111_2*1]))));
yfx[idx_tmp] = (__ldg(&(vt[idx111])) + ((__ldg(&(vt[idx111])) - __ldg(&(vt[idx111+stride_111_2*1]))) * __local_ratio__d15_17_431));
  }
    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
    idx_tmp += kstride_tmp;
}}
template<typename TmpStorage>__global__ void __launch_bounds__(192)  update_dz_c_stencil443_ms656_kernel(globals globals_, const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, const int tmpBeginIIndex, const int tmpBeginJIndex, const int jstride_tmp, const int kstride_tmp, ::dawn::float_type * const dp_ref, ::dawn::float_type * const ut, ::dawn::float_type * const vt, gridtools::data_view<TmpStorage>xfx_dv, gridtools::data_view<TmpStorage>yfx_dv) {

  // Start kernel
  ::dawn::float_type* xfx = &xfx_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0);
  ::dawn::float_type* yfx = &yfx_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0);
  const unsigned int nx = isize;
  const unsigned int ny = jsize;
  const int block_size_i = (blockIdx.x + 1) * 32 < nx ? 32 : nx - blockIdx.x * 32;
  const int block_size_j = (blockIdx.y + 1) * 4 < ny ? 4 : ny - blockIdx.y * 4;

  // computing the global position in the physical domain

  // In a typical cuda block we have the following regions

  // aa bbbbbbbb cc

  // aa bbbbbbbb cc

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // ee ffffffff gg

  // ee ffffffff gg

  // Regions b,d,f have warp (or multiple of warp size)

  // Size of regions a, c, h, i, e, g are determined by max_extent_t

  // Regions b,d,f are easily executed by dedicated warps (one warp for each line)

  // Regions (a,h,e) and (c,i,g) are executed by two specialized warp
  int iblock = 0 - 1;
  int jblock = 0 - 1;
if(threadIdx.y < +5) {
    iblock = threadIdx.x;
    jblock = (int)threadIdx.y + 0;
}else if(threadIdx.y < 6) {
    iblock = threadIdx.x % 1 + 32;
    jblock = (int)threadIdx.x / 1+0;
}
  // initialized iterators
  int idx111 = (blockIdx.x*32+iblock)*1+(blockIdx.y*4+jblock)*stride_111_1;
  int idx_tmp = (iblock+0)*1 + (jblock+0)*jstride_tmp;

  // jump iterators to match the intersection of beginning of next interval and the parallel execution block
  idx111 += max(ksize - 1+-1, blockIdx.z * 4) * stride_111_2;

  // jump tmp iterators to match the intersection of beginning of next interval and the parallel execution block
  idx_tmp += max(ksize - 1+-1, blockIdx.z * 4) * kstride_tmp;
  int kleg_lower_bound = max( ksize - 1 + -1,blockIdx.z*4);
  int kleg_upper_bound = min( ksize - 1 + 0,(blockIdx.z+1)*4-1);;
for(int k = kleg_lower_bound+0; k <= kleg_upper_bound+0; ++k) {
  if(iblock >= 0 && iblock <= block_size_i -1 + 1 && jblock >= 0 && jblock <= block_size_j -1 + 1) {
::dawn::float_type __local_ratio__c74_19_433 = (__ldg(&(dp_ref[idx111+stride_111_2*-1])) / (__ldg(&(dp_ref[idx111+stride_111_2*-2])) + __ldg(&(dp_ref[idx111+stride_111_2*-1]))));
xfx[idx_tmp] = (__ldg(&(ut[idx111+stride_111_2*-1])) + ((__ldg(&(ut[idx111+stride_111_2*-1])) - __ldg(&(ut[idx111+stride_111_2*-2]))) * __local_ratio__c74_19_433));
::dawn::float_type __local_ratio__c74_20_434 = (__ldg(&(dp_ref[idx111+stride_111_2*-1])) / (__ldg(&(dp_ref[idx111+stride_111_2*-2])) + __ldg(&(dp_ref[idx111+stride_111_2*-1]))));
yfx[idx_tmp] = (__ldg(&(vt[idx111+stride_111_2*-1])) + ((__ldg(&(vt[idx111+stride_111_2*-1])) - __ldg(&(vt[idx111+stride_111_2*-2]))) * __local_ratio__c74_20_434));
  }
    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
    idx_tmp += kstride_tmp;
}}
template<typename TmpStorage>__global__ void __launch_bounds__(192)  update_dz_c_stencil443_ms659_kernel(globals globals_, const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, const int tmpBeginIIndex, const int tmpBeginJIndex, const int jstride_tmp, const int kstride_tmp, ::dawn::float_type * const dp_ref, ::dawn::float_type * const ut, ::dawn::float_type * const vt, gridtools::data_view<TmpStorage>xfx_dv, gridtools::data_view<TmpStorage>yfx_dv) {

  // Start kernel
  ::dawn::float_type* xfx = &xfx_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0);
  ::dawn::float_type* yfx = &yfx_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0);
  const unsigned int nx = isize;
  const unsigned int ny = jsize;
  const int block_size_i = (blockIdx.x + 1) * 32 < nx ? 32 : nx - blockIdx.x * 32;
  const int block_size_j = (blockIdx.y + 1) * 4 < ny ? 4 : ny - blockIdx.y * 4;

  // computing the global position in the physical domain

  // In a typical cuda block we have the following regions

  // aa bbbbbbbb cc

  // aa bbbbbbbb cc

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // ee ffffffff gg

  // ee ffffffff gg

  // Regions b,d,f have warp (or multiple of warp size)

  // Size of regions a, c, h, i, e, g are determined by max_extent_t

  // Regions b,d,f are easily executed by dedicated warps (one warp for each line)

  // Regions (a,h,e) and (c,i,g) are executed by two specialized warp
  int iblock = 0 - 1;
  int jblock = 0 - 1;
if(threadIdx.y < +5) {
    iblock = threadIdx.x;
    jblock = (int)threadIdx.y + 0;
}else if(threadIdx.y < 6) {
    iblock = threadIdx.x % 1 + 32;
    jblock = (int)threadIdx.x / 1+0;
}
  // initialized iterators
  int idx111 = (blockIdx.x*32+iblock)*1+(blockIdx.y*4+jblock)*stride_111_1;
  int idx_tmp = (iblock+0)*1 + (jblock+0)*jstride_tmp;

  // jump iterators to match the intersection of beginning of next interval and the parallel execution block
  idx111 += max(1, blockIdx.z * 4) * stride_111_2;

  // jump tmp iterators to match the intersection of beginning of next interval and the parallel execution block
  idx_tmp += max(1, blockIdx.z * 4) * kstride_tmp;
  int kleg_lower_bound = max(1,blockIdx.z*4);
  int kleg_upper_bound = min( ksize - 1 + -1,(blockIdx.z+1)*4-1);;
for(int k = kleg_lower_bound+0; k <= kleg_upper_bound+0; ++k) {
  if(iblock >= 0 && iblock <= block_size_i -1 + 1 && jblock >= 0 && jblock <= block_size_j -1 + 1) {
::dawn::float_type __local_int_ratio__330_22_435 = ((::dawn::float_type) 1.0 / (__ldg(&(dp_ref[idx111+stride_111_2*-1])) + __ldg(&(dp_ref[idx111]))));
xfx[idx_tmp] = (((__ldg(&(dp_ref[idx111])) * __ldg(&(ut[idx111+stride_111_2*-1]))) + (__ldg(&(dp_ref[idx111+stride_111_2*-1])) * __ldg(&(ut[idx111])))) * __local_int_ratio__330_22_435);
::dawn::float_type __local_int_ratio__330_23_436 = ((::dawn::float_type) 1.0 / (__ldg(&(dp_ref[idx111+stride_111_2*-1])) + __ldg(&(dp_ref[idx111]))));
yfx[idx_tmp] = (((__ldg(&(dp_ref[idx111])) * __ldg(&(vt[idx111+stride_111_2*-1]))) + (__ldg(&(dp_ref[idx111+stride_111_2*-1])) * __ldg(&(vt[idx111])))) * __local_int_ratio__330_23_436);
  }
    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
    idx_tmp += kstride_tmp;
}}
template<typename TmpStorage>__global__ void __launch_bounds__(192)  update_dz_c_stencil443_ms664_kernel(globals globals_, const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, const int tmpBeginIIndex, const int tmpBeginJIndex, const int jstride_tmp, const int kstride_tmp, ::dawn::float_type * const area, ::dawn::float_type * const gz_x, ::dawn::float_type * const gz_y, ::dawn::float_type * const gz_0, gridtools::data_view<TmpStorage>xfx_dv, gridtools::data_view<TmpStorage>yfx_dv, gridtools::data_view<TmpStorage>fx_dv, gridtools::data_view<TmpStorage>fy_dv) {

  // Start kernel
  ::dawn::float_type* xfx = &xfx_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0);
  ::dawn::float_type* yfx = &yfx_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0);
  ::dawn::float_type* fx = &fx_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0);
  ::dawn::float_type* fy = &fy_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0);
  const unsigned int nx = isize;
  const unsigned int ny = jsize;
  const int block_size_i = (blockIdx.x + 1) * 32 < nx ? 32 : nx - blockIdx.x * 32;
  const int block_size_j = (blockIdx.y + 1) * 4 < ny ? 4 : ny - blockIdx.y * 4;

  // computing the global position in the physical domain

  // In a typical cuda block we have the following regions

  // aa bbbbbbbb cc

  // aa bbbbbbbb cc

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // ee ffffffff gg

  // ee ffffffff gg

  // Regions b,d,f have warp (or multiple of warp size)

  // Size of regions a, c, h, i, e, g are determined by max_extent_t

  // Regions b,d,f are easily executed by dedicated warps (one warp for each line)

  // Regions (a,h,e) and (c,i,g) are executed by two specialized warp
  int iblock = 0 - 1;
  int jblock = 0 - 1;
if(threadIdx.y < +5) {
    iblock = threadIdx.x;
    jblock = (int)threadIdx.y + 0;
}else if(threadIdx.y < 6) {
    iblock = threadIdx.x % 1 + 32;
    jblock = (int)threadIdx.x / 1+0;
}
  // initialized iterators
  int idx111 = (blockIdx.x*32+iblock)*1+(blockIdx.y*4+jblock)*stride_111_1;
  int idx_tmp = (iblock+0)*1 + (jblock+0)*jstride_tmp;

  // jump iterators to match the intersection of beginning of next interval and the parallel execution block
  idx111 += max(0, blockIdx.z * 4) * stride_111_2;

  // jump tmp iterators to match the intersection of beginning of next interval and the parallel execution block
  idx_tmp += max(0, blockIdx.z * 4) * kstride_tmp;
  int kleg_lower_bound = max(0,blockIdx.z*4);
  int kleg_upper_bound = min( ksize - 1 + 0,(blockIdx.z+1)*4-1);;
for(int k = kleg_lower_bound+0; k <= kleg_upper_bound+0; ++k) {
  if(iblock >= 0 && iblock <= block_size_i -1 + 1 && jblock >= 0 && jblock <= block_size_j -1 + 1) {
::dawn::float_type __local_fx__389_25_437 = (__ldg(&(xfx[idx_tmp])) * ((__ldg(&(xfx[idx_tmp])) > (::dawn::float_type) 0.0) ? __ldg(&(gz_x[idx111+1*-1])) : __ldg(&(gz_x[idx111]))));
::dawn::float_type __local_fy__389_25_438 = (__ldg(&(yfx[idx_tmp])) * ((__ldg(&(yfx[idx_tmp])) > (::dawn::float_type) 0.0) ? __ldg(&(gz_y[idx111+stride_111_1*-1])) : __ldg(&(gz_y[idx111]))));
fx[idx_tmp] = __local_fx__389_25_437;
fy[idx_tmp] = __local_fy__389_25_438;
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
gz_0[idx111] = ((((((__ldg(&(gz_y[idx111])) * __ldg(&(area[idx111]))) + fx[idx_tmp]) - fx[idx_tmp+1*1]) + fy[idx_tmp]) - fy[idx_tmp+jstride_tmp*1]) / ((((__ldg(&(area[idx111])) + __ldg(&(xfx[idx_tmp]))) - __ldg(&(xfx[idx_tmp+1*1]))) + __ldg(&(yfx[idx_tmp]))) - __ldg(&(yfx[idx_tmp+jstride_tmp*1]))));
  }
    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
    idx_tmp += kstride_tmp;
}}
__global__ void __launch_bounds__(128)  update_dz_c_stencil443_ms789_kernel(globals globals_, const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, ::dawn::float_type * const gz, ::dawn::float_type * const gz_0) {

  // Start kernel
  const unsigned int nx = isize;
  const unsigned int ny = jsize;
  const int block_size_i = (blockIdx.x + 1) * 32 < nx ? 32 : nx - blockIdx.x * 32;
  const int block_size_j = (blockIdx.y + 1) * 4 < ny ? 4 : ny - blockIdx.y * 4;

  // computing the global position in the physical domain

  // In a typical cuda block we have the following regions

  // aa bbbbbbbb cc

  // aa bbbbbbbb cc

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // ee ffffffff gg

  // ee ffffffff gg

  // Regions b,d,f have warp (or multiple of warp size)

  // Size of regions a, c, h, i, e, g are determined by max_extent_t

  // Regions b,d,f are easily executed by dedicated warps (one warp for each line)

  // Regions (a,h,e) and (c,i,g) are executed by two specialized warp
  int iblock = 0 - 1;
  int jblock = 0 - 1;
if(threadIdx.y < +4) {
    iblock = threadIdx.x;
    jblock = (int)threadIdx.y + 0;
}
  // initialized iterators
  int idx111 = (blockIdx.x*32+iblock)*1+(blockIdx.y*4+jblock)*stride_111_1;

  // jump iterators to match the intersection of beginning of next interval and the parallel execution block
  idx111 += max(ksize - 1+-1, blockIdx.z * 4) * stride_111_2;
  int kleg_lower_bound = max( ksize - 1 + -1,blockIdx.z*4);
  int kleg_upper_bound = min( ksize - 1 + 0,(blockIdx.z+1)*4-1);;
for(int k = kleg_lower_bound+0; k <= kleg_upper_bound+0; ++k) {
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
gz_0[idx111] = __ldg(&(gz[idx111]));
  }
    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
}}
__global__ void __launch_bounds__(128)  update_dz_c_stencil443_ms669_kernel(globals globals_, const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, ::dawn::float_type * const zs, ::dawn::float_type * const ws3, ::dawn::float_type * const gz_0) {

  // Start kernel
  const unsigned int nx = isize;
  const unsigned int ny = jsize;
  const int block_size_i = (blockIdx.x + 1) * 32 < nx ? 32 : nx - blockIdx.x * 32;
  const int block_size_j = (blockIdx.y + 1) * 4 < ny ? 4 : ny - blockIdx.y * 4;

  // computing the global position in the physical domain

  // In a typical cuda block we have the following regions

  // aa bbbbbbbb cc

  // aa bbbbbbbb cc

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // ee ffffffff gg

  // ee ffffffff gg

  // Regions b,d,f have warp (or multiple of warp size)

  // Size of regions a, c, h, i, e, g are determined by max_extent_t

  // Regions b,d,f are easily executed by dedicated warps (one warp for each line)

  // Regions (a,h,e) and (c,i,g) are executed by two specialized warp
  int iblock = 0 - 1;
  int jblock = 0 - 1;
if(threadIdx.y < +4) {
    iblock = threadIdx.x;
    jblock = (int)threadIdx.y + 0;
}
  // initialized iterators
  int idx111 = (blockIdx.x*32+iblock)*1+(blockIdx.y*4+jblock)*stride_111_1;

  // jump iterators to match the intersection of beginning of next interval and the parallel execution block
  idx111 += max(ksize - 1+-1, blockIdx.z * 4) * stride_111_2;
  int kleg_lower_bound = max( ksize - 1 + -1,blockIdx.z*4);
  int kleg_upper_bound = min( ksize - 1 + 0,(blockIdx.z+1)*4-1);;
for(int k = kleg_lower_bound+0; k <= kleg_upper_bound+0; ++k) {
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
ws3[idx111] = ((__ldg(&(zs[idx111])) - __ldg(&(gz_0[idx111]))) * ((::dawn::float_type) 1.0 / globals_.dt));
  }
    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
}}
__global__ void __launch_bounds__(128)  update_dz_c_stencil443_ms673_kernel(globals globals_, const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, ::dawn::float_type * const gz, ::dawn::float_type * const gz_0) {

  // Start kernel
  const unsigned int nx = isize;
  const unsigned int ny = jsize;
  const int block_size_i = (blockIdx.x + 1) * 32 < nx ? 32 : nx - blockIdx.x * 32;
  const int block_size_j = (blockIdx.y + 1) * 4 < ny ? 4 : ny - blockIdx.y * 4;

  // computing the global position in the physical domain

  // In a typical cuda block we have the following regions

  // aa bbbbbbbb cc

  // aa bbbbbbbb cc

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // ee ffffffff gg

  // ee ffffffff gg

  // Regions b,d,f have warp (or multiple of warp size)

  // Size of regions a, c, h, i, e, g are determined by max_extent_t

  // Regions b,d,f are easily executed by dedicated warps (one warp for each line)

  // Regions (a,h,e) and (c,i,g) are executed by two specialized warp
  int iblock = 0 - 1;
  int jblock = 0 - 1;
if(threadIdx.y < +4) {
    iblock = threadIdx.x;
    jblock = (int)threadIdx.y + 0;
}
  // initialized iterators
  int idx111 = (blockIdx.x*32+iblock)*1+(blockIdx.y*4+jblock)*stride_111_1;

  // jump iterators to match the intersection of beginning of next interval and the parallel execution block
  idx111 += max(0, blockIdx.z * 4) * stride_111_2;
  int kleg_lower_bound = max(0,blockIdx.z*4);
  int kleg_upper_bound = min( ksize - 1 + -1,(blockIdx.z+1)*4-1);;
for(int k = kleg_lower_bound+0; k <= kleg_upper_bound+0; ++k) {
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
::dawn::float_type __local_gz_442 = (__ldg(&(gz_0[idx111+stride_111_2*1])) + (::dawn::float_type) 2.0);
gz[idx111] = ((gz[idx111] > __local_gz_442) ? gz[idx111] : __local_gz_442);
  }
    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
}}

class update_dz_c {
public:

  struct sbase : public timer_cuda {

    sbase(std::string name) : timer_cuda(name){}

    double get_time() {
      return total_time();
    }
  };

  struct stencil_443 : public sbase {

    // Members

    // Temporary storage typedefs
    using tmp_halo_t = gridtools::halo< 0,0, 0, 0, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t< 0, 5, tmp_halo_t >;
    using tmp_storage_t = storage_traits_t::data_store_t< ::dawn::float_type, tmp_meta_data_t>;
    globals& m_globals;
    const gridtools::dawn::domain m_dom;

    // temporary storage declarations
    tmp_meta_data_t m_tmp_meta_data;
    tmp_storage_t m_xfx;
    tmp_storage_t m_yfx;
    tmp_storage_t m_fx;
    tmp_storage_t m_fy;
  public:

    stencil_443(const gridtools::dawn::domain& dom_, globals& globals_, int rank, int xcols, int ycols) : sbase("stencil_443"), m_dom(dom_), m_globals(globals_), m_tmp_meta_data(32+1, 4+1, (dom_.isize()+ 32 - 1) / 32, (dom_.jsize()+ 4 - 1) / 4, dom_.ksize() + 2 * 0), m_xfx(m_tmp_meta_data), m_yfx(m_tmp_meta_data), m_fx(m_tmp_meta_data), m_fy(m_tmp_meta_data){}
    static constexpr dawn::driver::cartesian_extent dp_ref_extent = {0,1, 0,1, -2,1};
    static constexpr dawn::driver::cartesian_extent zs_extent = {0,0, 0,0, 0,0};
    static constexpr dawn::driver::cartesian_extent area_extent = {0,0, 0,0, 0,0};
    static constexpr dawn::driver::cartesian_extent ut_extent = {0,1, 0,1, -2,1};
    static constexpr dawn::driver::cartesian_extent vt_extent = {0,1, 0,1, -2,1};
    static constexpr dawn::driver::cartesian_extent gz_extent = {0,0, 0,0, 0,0};
    static constexpr dawn::driver::cartesian_extent gz_x_extent = {-1,1, 0,1, 0,0};
    static constexpr dawn::driver::cartesian_extent gz_y_extent = {0,1, -1,1, 0,0};
    static constexpr dawn::driver::cartesian_extent ws3_extent = {0,0, 0,0, 0,0};
    static constexpr dawn::driver::cartesian_extent gz_0_extent = {0,0, 0,0, 0,1};

    void run(storage_ijk_t dp_ref_ds, storage_ijk_t zs_ds, storage_ijk_t area_ds, storage_ijk_t ut_ds, storage_ijk_t vt_ds, storage_ijk_t gz_ds, storage_ijk_t gz_x_ds, storage_ijk_t gz_y_ds, storage_ijk_t ws3_ds, storage_ijk_t gz_0_ds) {

      // starting timers
      start();
      {;
      gridtools::data_view<storage_ijk_t> dp_ref= gridtools::make_device_view(dp_ref_ds);
      gridtools::data_view<storage_ijk_t> ut= gridtools::make_device_view(ut_ds);
      gridtools::data_view<storage_ijk_t> vt= gridtools::make_device_view(vt_ds);
      gridtools::data_view<tmp_storage_t> xfx= gridtools::make_device_view( m_xfx);
      gridtools::data_view<tmp_storage_t> yfx= gridtools::make_device_view( m_yfx);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,4+2,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 4 - 1) / 4;
      const unsigned int nbz = (m_dom.ksize()+4-1) / 4;
      dim3 blocks(nbx, nby, nbz);
      update_dz_c_stencil443_ms653_kernel<<<blocks, threads>>>(m_globals,nx,ny,nz,dp_ref_ds.strides()[1],dp_ref_ds.strides()[2],m_xfx.get_storage_info_ptr()->template begin<0>(),m_xfx.get_storage_info_ptr()->template begin<1>(),m_xfx.get_storage_info_ptr()->template stride<1>(),m_xfx.get_storage_info_ptr()->template stride<4>(),(dp_ref.data()+dp_ref_ds.get_storage_info_ptr()->index(dp_ref.begin<0>(), dp_ref.begin<1>(),0 )),(ut.data()+ut_ds.get_storage_info_ptr()->index(ut.begin<0>(), ut.begin<1>(),0 )),(vt.data()+vt_ds.get_storage_info_ptr()->index(vt.begin<0>(), vt.begin<1>(),0 )),xfx,yfx);
      };
      {;
      gridtools::data_view<storage_ijk_t> dp_ref= gridtools::make_device_view(dp_ref_ds);
      gridtools::data_view<storage_ijk_t> ut= gridtools::make_device_view(ut_ds);
      gridtools::data_view<storage_ijk_t> vt= gridtools::make_device_view(vt_ds);
      gridtools::data_view<tmp_storage_t> xfx= gridtools::make_device_view( m_xfx);
      gridtools::data_view<tmp_storage_t> yfx= gridtools::make_device_view( m_yfx);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,4+2,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 4 - 1) / 4;
      const unsigned int nbz = (m_dom.ksize()+4-1) / 4;
      dim3 blocks(nbx, nby, nbz);
      update_dz_c_stencil443_ms656_kernel<<<blocks, threads>>>(m_globals,nx,ny,nz,dp_ref_ds.strides()[1],dp_ref_ds.strides()[2],m_xfx.get_storage_info_ptr()->template begin<0>(),m_xfx.get_storage_info_ptr()->template begin<1>(),m_xfx.get_storage_info_ptr()->template stride<1>(),m_xfx.get_storage_info_ptr()->template stride<4>(),(dp_ref.data()+dp_ref_ds.get_storage_info_ptr()->index(dp_ref.begin<0>(), dp_ref.begin<1>(),0 )),(ut.data()+ut_ds.get_storage_info_ptr()->index(ut.begin<0>(), ut.begin<1>(),0 )),(vt.data()+vt_ds.get_storage_info_ptr()->index(vt.begin<0>(), vt.begin<1>(),0 )),xfx,yfx);
      };
      {;
      gridtools::data_view<storage_ijk_t> dp_ref= gridtools::make_device_view(dp_ref_ds);
      gridtools::data_view<storage_ijk_t> ut= gridtools::make_device_view(ut_ds);
      gridtools::data_view<storage_ijk_t> vt= gridtools::make_device_view(vt_ds);
      gridtools::data_view<tmp_storage_t> xfx= gridtools::make_device_view( m_xfx);
      gridtools::data_view<tmp_storage_t> yfx= gridtools::make_device_view( m_yfx);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,4+2,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 4 - 1) / 4;
      const unsigned int nbz = (m_dom.ksize()+4-1) / 4;
      dim3 blocks(nbx, nby, nbz);
      update_dz_c_stencil443_ms659_kernel<<<blocks, threads>>>(m_globals,nx,ny,nz,dp_ref_ds.strides()[1],dp_ref_ds.strides()[2],m_xfx.get_storage_info_ptr()->template begin<0>(),m_xfx.get_storage_info_ptr()->template begin<1>(),m_xfx.get_storage_info_ptr()->template stride<1>(),m_xfx.get_storage_info_ptr()->template stride<4>(),(dp_ref.data()+dp_ref_ds.get_storage_info_ptr()->index(dp_ref.begin<0>(), dp_ref.begin<1>(),0 )),(ut.data()+ut_ds.get_storage_info_ptr()->index(ut.begin<0>(), ut.begin<1>(),0 )),(vt.data()+vt_ds.get_storage_info_ptr()->index(vt.begin<0>(), vt.begin<1>(),0 )),xfx,yfx);
      };
      {;
      gridtools::data_view<storage_ijk_t> area= gridtools::make_device_view(area_ds);
      gridtools::data_view<storage_ijk_t> gz_x= gridtools::make_device_view(gz_x_ds);
      gridtools::data_view<storage_ijk_t> gz_y= gridtools::make_device_view(gz_y_ds);
      gridtools::data_view<storage_ijk_t> gz_0= gridtools::make_device_view(gz_0_ds);
      gridtools::data_view<tmp_storage_t> xfx= gridtools::make_device_view( m_xfx);
      gridtools::data_view<tmp_storage_t> yfx= gridtools::make_device_view( m_yfx);
      gridtools::data_view<tmp_storage_t> fx= gridtools::make_device_view( m_fx);
      gridtools::data_view<tmp_storage_t> fy= gridtools::make_device_view( m_fy);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,4+2,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 4 - 1) / 4;
      const unsigned int nbz = (m_dom.ksize()+4-1) / 4;
      dim3 blocks(nbx, nby, nbz);
      update_dz_c_stencil443_ms664_kernel<<<blocks, threads>>>(m_globals,nx,ny,nz,area_ds.strides()[1],area_ds.strides()[2],m_xfx.get_storage_info_ptr()->template begin<0>(),m_xfx.get_storage_info_ptr()->template begin<1>(),m_xfx.get_storage_info_ptr()->template stride<1>(),m_xfx.get_storage_info_ptr()->template stride<4>(),(area.data()+area_ds.get_storage_info_ptr()->index(area.begin<0>(), area.begin<1>(),0 )),(gz_x.data()+gz_x_ds.get_storage_info_ptr()->index(gz_x.begin<0>(), gz_x.begin<1>(),0 )),(gz_y.data()+gz_y_ds.get_storage_info_ptr()->index(gz_y.begin<0>(), gz_y.begin<1>(),0 )),(gz_0.data()+gz_0_ds.get_storage_info_ptr()->index(gz_0.begin<0>(), gz_0.begin<1>(),0 )),xfx,yfx,fx,fy);
      };
      {;
      gridtools::data_view<storage_ijk_t> gz= gridtools::make_device_view(gz_ds);
      gridtools::data_view<storage_ijk_t> gz_0= gridtools::make_device_view(gz_0_ds);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,4+0,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 4 - 1) / 4;
      const unsigned int nbz = (m_dom.ksize()+4-1) / 4;
      dim3 blocks(nbx, nby, nbz);
      update_dz_c_stencil443_ms789_kernel<<<blocks, threads>>>(m_globals,nx,ny,nz,gz_ds.strides()[1],gz_ds.strides()[2],(gz.data()+gz_ds.get_storage_info_ptr()->index(gz.begin<0>(), gz.begin<1>(),0 )),(gz_0.data()+gz_0_ds.get_storage_info_ptr()->index(gz_0.begin<0>(), gz_0.begin<1>(),0 )));
      };
      {;
      gridtools::data_view<storage_ijk_t> zs= gridtools::make_device_view(zs_ds);
      gridtools::data_view<storage_ijk_t> ws3= gridtools::make_device_view(ws3_ds);
      gridtools::data_view<storage_ijk_t> gz_0= gridtools::make_device_view(gz_0_ds);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,4+0,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 4 - 1) / 4;
      const unsigned int nbz = (m_dom.ksize()+4-1) / 4;
      dim3 blocks(nbx, nby, nbz);
      update_dz_c_stencil443_ms669_kernel<<<blocks, threads>>>(m_globals,nx,ny,nz,zs_ds.strides()[1],zs_ds.strides()[2],(zs.data()+zs_ds.get_storage_info_ptr()->index(zs.begin<0>(), zs.begin<1>(),0 )),(ws3.data()+ws3_ds.get_storage_info_ptr()->index(ws3.begin<0>(), ws3.begin<1>(),0 )),(gz_0.data()+gz_0_ds.get_storage_info_ptr()->index(gz_0.begin<0>(), gz_0.begin<1>(),0 )));
      };
      {;
      gridtools::data_view<storage_ijk_t> gz= gridtools::make_device_view(gz_ds);
      gridtools::data_view<storage_ijk_t> gz_0= gridtools::make_device_view(gz_0_ds);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,4+0,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 4 - 1) / 4;
      const unsigned int nbz = (m_dom.ksize()+4-1) / 4;
      dim3 blocks(nbx, nby, nbz);
      update_dz_c_stencil443_ms673_kernel<<<blocks, threads>>>(m_globals,nx,ny,nz,gz_ds.strides()[1],gz_ds.strides()[2],(gz.data()+gz_ds.get_storage_info_ptr()->index(gz.begin<0>(), gz.begin<1>(),0 )),(gz_0.data()+gz_0_ds.get_storage_info_ptr()->index(gz_0.begin<0>(), gz_0.begin<1>(),0 )));
      };

      // stopping timers
      pause();
    }
  };
  static constexpr const char* s_name = "update_dz_c";
  stencil_443 m_stencil_443;
public:

  update_dz_c(const update_dz_c&) = delete;

  // Members

  // Stencil-Data
  gridtools::dawn::meta_data_t m_meta_data;
  gridtools::dawn::storage_t m_gz_0;
  globals m_globals;

  update_dz_c(const gridtools::dawn::domain& dom, int rank = 1, int xcols = 1, int ycols = 1) : m_stencil_443(dom,m_globals, rank, xcols, ycols), m_meta_data(dom.isize(), dom.jsize(), dom.ksize() /*+ 2 *0*/ + 1), m_gz_0 (m_meta_data, "gz_0"){}

  // Access-wrapper for globally defined variables

  double get_dt() {
    return m_globals.dt;
  }

  void set_dt(double dt) {
    m_globals.dt=dt;
  }

  template<typename S>
  void sync_storages(S field) {
    field.sync();
  }

  template<typename S0, typename ... S>
  void sync_storages(S0 f0, S... fields) {
    f0.sync();
    sync_storages(fields...);
  }

  void run(storage_ijk_t dp_ref, storage_ijk_t zs, storage_ijk_t area, storage_ijk_t ut, storage_ijk_t vt, storage_ijk_t gz, storage_ijk_t gz_x, storage_ijk_t gz_y, storage_ijk_t ws3) {
    sync_storages(dp_ref,zs,area,ut,vt,gz,gz_x,gz_y,ws3);
    m_stencil_443.run(dp_ref,zs,area,ut,vt,gz,gz_x,gz_y,ws3,m_gz_0);
;
    sync_storages(dp_ref,zs,area,ut,vt,gz,gz_x,gz_y,ws3);
  }

  std::string get_name()  const {
    return std::string(s_name);
  }

  void reset_meters() {
m_stencil_443.reset();  }

  double get_total_time() {
    double res = 0;
    res +=m_stencil_443.get_time();
    return res;
  }
};
} // namespace cuda
} // namespace dawn_generated
