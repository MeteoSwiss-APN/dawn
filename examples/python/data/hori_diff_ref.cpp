namespace dawn_generated{
namespace cuda{
__global__ void __launch_bounds__(256)  hori_diff_stencil77_ms85_kernel(const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, gridtools::clang::float_type * const in, gridtools::clang::float_type * const out, gridtools::clang::float_type * const coeff) {

  // Start kernel
  __shared__ gridtools::clang::float_type lap_ijcache[34*6];
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
  int iblock = -1 - 1;
  int jblock = -1 - 1;
if(threadIdx.y < +6) {
    iblock = threadIdx.x;
    jblock = (int)threadIdx.y + -1;
}else if(threadIdx.y < +7) {
    iblock = -1 + (int)threadIdx.x % 1;
    jblock = (int)threadIdx.x / 1+-1;
}else if(threadIdx.y < 8) {
    iblock = threadIdx.x % 1 + 32;
    jblock = (int)threadIdx.x / 1+-1;
}
  // initialized iterators
  int idx111 = (blockIdx.x*32+iblock)*1+(blockIdx.y*4+jblock)*stride_111_1;
  int ijcacheindex= iblock + 1 + (jblock + 1)*34;

  // jump iterators to match the intersection of beginning of next interval and the parallel execution block 
  idx111 += max(0, blockIdx.z * 4) * stride_111_2;
  int kleg_lower_bound = max(0,blockIdx.z*4);
  int kleg_upper_bound = min( ksize - 1 + 0,(blockIdx.z+1)*4-1);;
for(int k = kleg_lower_bound+0; k <= kleg_upper_bound+0; ++k) {
  if(iblock >= -1 && iblock <= block_size_i -1 + 1 && jblock >= -1 && jblock <= block_size_j -1 + 1) {
lap_ijcache[ijcacheindex] = (((gridtools::clang::float_type) -4.0 * __ldg(&(in[idx111]))) + (__ldg(&(coeff[idx111])) * (__ldg(&(in[idx111+1*1])) + (__ldg(&(in[idx111+1*-1])) + (__ldg(&(in[idx111+stride_111_1*1])) + __ldg(&(in[idx111+stride_111_1*-1])))))));
  }    __syncthreads();
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
out[idx111] = (((gridtools::clang::float_type) -4.0 * lap_ijcache[ijcacheindex]) + (__ldg(&(coeff[idx111])) * (lap_ijcache[ijcacheindex+1] + (lap_ijcache[ijcacheindex+-1] + (lap_ijcache[ijcacheindex+1*34] + lap_ijcache[ijcacheindex+-1*34])))));
  }    __syncthreads();

    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
}}

class hori_diff {
public:

  struct sbase : public timer_cuda {

    sbase(std::string name) : timer_cuda(name){}

    double get_time() {
      return total_time();
    }

    virtual void run() {
    }

    virtual void sync_storages() {
    }

    virtual ~sbase() {
    }
  };

  struct stencil_77 : public sbase {

    // Members

    // Temporary storage typedefs
    using tmp_halo_t = gridtools::halo< 1,1, 0, 0, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t< 0, 5, tmp_halo_t >;
    using tmp_storage_t = storage_traits_t::data_store_t< float_type, tmp_meta_data_t>;
    const gridtools::clang::domain& m_dom;

    // storage declarations
    storage_ijk_t& m_in;
    storage_ijk_t& m_out;
    storage_ijk_t& m_coeff;

    // temporary storage declarations
    tmp_meta_data_t m_tmp_meta_data;
    tmp_storage_t m_lap;
  public:

    stencil_77(const gridtools::clang::domain& dom_, storage_ijk_t& in_, storage_ijk_t& out_, storage_ijk_t& coeff_) : sbase("stencil_77"), m_dom(dom_), m_in(in_), m_out(out_), m_coeff(coeff_), m_tmp_meta_data(32+2, 4+2, (dom_.isize()+ 32 - 1) / 32, (dom_.jsize()+ 4 - 1) / 4, dom_.ksize() + 2 * 0), m_lap(m_tmp_meta_data){}

    ~stencil_77() {
    }

    void sync_storages() {
      m_in.sync();
      m_out.sync();
      m_coeff.sync();
    }

    virtual void run() {

      // starting timers
      start();
      {;
      gridtools::data_view<storage_ijk_t> in= gridtools::make_device_view(m_in);
      gridtools::data_view<storage_ijk_t> out= gridtools::make_device_view(m_out);
      gridtools::data_view<storage_ijk_t> coeff= gridtools::make_device_view(m_coeff);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,4+4,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 4 - 1) / 4;
      const unsigned int nbz = (m_dom.ksize()+4-1) / 4;
      dim3 blocks(nbx, nby, nbz);
      hori_diff_stencil77_ms85_kernel<<<blocks, threads>>>(nx,ny,nz,m_in.strides()[1],m_in.strides()[2],(in.data()+m_in.get_storage_info_ptr()->index(in.begin<0>(), in.begin<1>(),0 )),(out.data()+m_out.get_storage_info_ptr()->index(out.begin<0>(), out.begin<1>(),0 )),(coeff.data()+m_coeff.get_storage_info_ptr()->index(coeff.begin<0>(), coeff.begin<1>(),0 )));
      };

      // stopping timers
      pause();
    }

    sbase* get_stencil() {
      return this;
    }
  };
  static constexpr const char* s_name = "hori_diff";
  sbase* m_stencil_77;
public:

  hori_diff(const hori_diff&) = delete;

  // Members

  // Stencil-Data

  hori_diff(const gridtools::clang::domain& dom, storage_ijk_t& in, storage_ijk_t& out, storage_ijk_t& coeff) : m_stencil_77(new stencil_77(dom,in,out,coeff) ){}

  void run() {
    sync_storages();
    m_stencil_77->run();
;
    sync_storages();
  }

  void sync_storages() {
    m_stencil_77->sync_storages();
  }

  std::string get_name()  const {
    return std::string(s_name);
  }

  std::vector<sbase*> getStencils() {
    return std::vector<sbase*>({m_stencil_77});
  }

  void reset_meters() {
m_stencil_77->reset();  }
};
} // namespace cuda
} // namespace dawn_generated
