namespace dawn_generated{
namespace cuda{
__global__ void __launch_bounds__(32)  tridiagonal_solve_stencil49_ms102_kernel(const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, gridtools::clang::float_type * const a, gridtools::clang::float_type * const b, gridtools::clang::float_type * const c, gridtools::clang::float_type * const d) {

  // Start kernel
  gridtools::clang::float_type c_kcache[2];
  gridtools::clang::float_type d_kcache[2];
  const unsigned int nx = isize;
  const unsigned int ny = jsize;
  const int block_size_i = (blockIdx.x + 1) * 32 < nx ? 32 : nx - blockIdx.x * 32;
  const int block_size_j = (blockIdx.y + 1) * 1 < ny ? 1 : ny - blockIdx.y * 1;

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
if(threadIdx.y < +1) {
    iblock = threadIdx.x;
    jblock = (int)threadIdx.y + 0;
}
  // initialized iterators
  int idx111 = (blockIdx.x*32+iblock)*1+(blockIdx.y*1+jblock)*stride_111_1;

  // Pre-fill of kcaches
for(int k = 0+0; k <= 0+0; ++k) {

    // Head fill of kcaches
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
      c_kcache[1] =c[idx111];
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
c_kcache[1] = (c_kcache[1] / __ldg(&(b[idx111])));
  }
    // Flush of kcaches

    // Flush of kcaches
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
      c[idx111]= c_kcache[1];
  }
    // Slide kcaches
    c_kcache[0] = c_kcache[1];

    // increment iterators
    idx111+=stride_111_2;
}
  // Final flush of kcaches
if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
}
  // Final flush of kcaches

  // Final flush of kcaches

  // Pre-fill of kcaches
if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
    d_kcache[0] =d[idx111+stride_111_2*-1];
}for(int k = 1+0; k <=  ksize - 1 + 0+0; ++k) {

    // Head fill of kcaches
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
      c_kcache[1] =c[idx111];
      d_kcache[1] =d[idx111];
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
int __local_m_98 = ((gridtools::clang::float_type) 1.0 / (__ldg(&(b[idx111])) - (__ldg(&(a[idx111])) * c_kcache[0])));
c_kcache[1] = (c_kcache[1] * __local_m_98);
d_kcache[1] = ((d_kcache[1] - (__ldg(&(a[idx111])) * d_kcache[0])) * __local_m_98);
  }
    // Flush of kcaches

    // Flush of kcaches
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
      c[idx111+stride_111_2*-1]= c_kcache[0];
    if( k - 1 >= 1) {
        d[idx111+stride_111_2*-1]= d_kcache[0];
    }  }
    // Slide kcaches
    c_kcache[0] = c_kcache[1];
    d_kcache[0] = d_kcache[1];

    // increment iterators
    idx111+=stride_111_2;
}
  // Final flush of kcaches
if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
    c[idx111+stride_111_2*-1]= c_kcache[0];
  if(  ksize - 1 + 1 - 1 >= 1) {
      d[idx111+stride_111_2*-1]= d_kcache[0];
  }}
  // Final flush of kcaches

  // Final flush of kcaches
}
__global__ void __launch_bounds__(32)  tridiagonal_solve_stencil49_ms103_kernel(const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, gridtools::clang::float_type * const c, gridtools::clang::float_type * const d) {

  // Start kernel
  gridtools::clang::float_type d_kcache[2];
  const unsigned int nx = isize;
  const unsigned int ny = jsize;
  const int block_size_i = (blockIdx.x + 1) * 32 < nx ? 32 : nx - blockIdx.x * 32;
  const int block_size_j = (blockIdx.y + 1) * 1 < ny ? 1 : ny - blockIdx.y * 1;

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
if(threadIdx.y < +1) {
    iblock = threadIdx.x;
    jblock = (int)threadIdx.y + 0;
}
  // initialized iterators
  int idx111 = (blockIdx.x*32+iblock)*1+(blockIdx.y*1+jblock)*stride_111_1;

  // jump iterators to match the beginning of next interval
  idx111 += stride_111_2*(ksize - 1+-1);

  // Pre-fill of kcaches
if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
    d_kcache[1] =d[idx111+stride_111_2*1];
}for(int k =  ksize - 1 + -1+0; k >= 0+0; --k) {

    // Head fill of kcaches
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
      d_kcache[0] =d[idx111];
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
d_kcache[0] -= (__ldg(&(c[idx111])) * d_kcache[1]);
  }
    // Flush of kcaches

    // Flush of kcaches
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
    if(  ksize - 1 + -1 - k >= 1) {
        d[idx111+stride_111_2*1]= d_kcache[1];
    }  }
    // Slide kcaches
    d_kcache[1] = d_kcache[0];

    // increment iterators
    idx111-=stride_111_2;
}
  // Final flush of kcaches
if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
  if(  ksize - 1 + -1 - -1 >= 1) {
      d[idx111+stride_111_2*1]= d_kcache[1];
  }}
  // Final flush of kcaches

  // Final flush of kcaches
}

class tridiagonal_solve {
public:

  struct sbase : public timer_cuda {

    sbase(std::string name) : timer_cuda(name){}

    double get_time() {
      return total_time();
    }

    virtual ~sbase() {
    }
  };

  struct stencil_49 : public sbase {

    // Members

    // Temporary storage typedefs
    using tmp_halo_t = gridtools::halo< 0,0, 0, 0, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t< 0, 5, tmp_halo_t >;
    using tmp_storage_t = storage_traits_t::data_store_t< float_type, tmp_meta_data_t>;
    const gridtools::clang::domain& m_dom;
  public:

    stencil_49(const gridtools::clang::domain& dom_, storage_ijk_t& a_, storage_ijk_t& b_, storage_ijk_t& c_, storage_ijk_t& d_) : sbase("stencil_49"), m_dom(dom_){}

    void run(storage_ijk_t a_ds, storage_ijk_t b_ds, storage_ijk_t c_ds, storage_ijk_t d_ds) {

      // starting timers
      start();
      {;
      gridtools::data_view<storage_ijk_t> a= gridtools::make_device_view(a_ds);
      gridtools::data_view<storage_ijk_t> b= gridtools::make_device_view(b_ds);
      gridtools::data_view<storage_ijk_t> c= gridtools::make_device_view(c_ds);
      gridtools::data_view<storage_ijk_t> d= gridtools::make_device_view(d_ds);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,1+0,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 1 - 1) / 1;
      const unsigned int nbz = 1;
      dim3 blocks(nbx, nby, nbz);
      tridiagonal_solve_stencil49_ms102_kernel<<<blocks, threads>>>(nx,ny,nz,a_ds.strides()[1],a_ds.strides()[2],(a.data()+a_ds.get_storage_info_ptr()->index(a.begin<0>(), a.begin<1>(),0 )),(b.data()+b_ds.get_storage_info_ptr()->index(b.begin<0>(), b.begin<1>(),0 )),(c.data()+c_ds.get_storage_info_ptr()->index(c.begin<0>(), c.begin<1>(),0 )),(d.data()+d_ds.get_storage_info_ptr()->index(d.begin<0>(), d.begin<1>(),0 )));
      };
      {;
      gridtools::data_view<storage_ijk_t> c= gridtools::make_device_view(c_ds);
      gridtools::data_view<storage_ijk_t> d= gridtools::make_device_view(d_ds);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,1+0,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 1 - 1) / 1;
      const unsigned int nbz = 1;
      dim3 blocks(nbx, nby, nbz);
      tridiagonal_solve_stencil49_ms103_kernel<<<blocks, threads>>>(nx,ny,nz,c_ds.strides()[1],c_ds.strides()[2],(c.data()+c_ds.get_storage_info_ptr()->index(c.begin<0>(), c.begin<1>(),0 )),(d.data()+d_ds.get_storage_info_ptr()->index(d.begin<0>(), d.begin<1>(),0 )));
      };

      // stopping timers
      pause();
    }
  };
  static constexpr const char* s_name = "tridiagonal_solve";
  stencil_49* m_stencil_49;
public:

  tridiagonal_solve(const tridiagonal_solve&) = delete;

  // Members

  // Stencil-Data

  tridiagonal_solve(const gridtools::clang::domain& dom, storage_ijk_t& a, storage_ijk_t& b, storage_ijk_t& c, storage_ijk_t& d) : m_stencil_49(new stencil_49(dom,a,b,c,d) ){}

  template<typename S>
  void sync_storages(S field) {
    field.sync();
  }

  template<typename S0, typename ... S>
  void sync_storages(S0 f0, S... fields) {
    f0.sync();
    sync_storages(fields...);
  }

  void run(storage_ijk_t a, storage_ijk_t b, storage_ijk_t c, storage_ijk_t d) {
    sync_storages(a,b,c,d);
    m_stencil_49->run(a,b,c,d);
;
    sync_storages(a,b,c,d);
  }

  std::string get_name()  const {
    return std::string(s_name);
  }

  void reset_meters() {
m_stencil_49->reset();  }

  double get_total_time() {
    double res = 0;
    res +=m_stencil_49->get_time();
    return res;
  }
};
} // namespace cuda
} // namespace dawn_generated
