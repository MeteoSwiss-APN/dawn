namespace dawn_generated{
namespace cuda{
__global__ void __launch_bounds__(32)  copy_stencil_stencil19_ms23_kernel(const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, gridtools::clang::float_type * const in, gridtools::clang::float_type * const out) {

  // Start kernel
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

  // jump iterators to match the intersection of beginning of next interval and the parallel execution block 
  idx111 += max(0, blockIdx.z * 4) * stride_111_2;
  int kleg_lower_bound = max(0,blockIdx.z*4);
  int kleg_upper_bound = min( ksize - 1 + 0,(blockIdx.z+1)*4-1);;
for(int k = kleg_lower_bound+0; k <= kleg_upper_bound+0; ++k) {
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
out[idx111] = __ldg(&(in[idx111+1*1]));
  }
    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
}}

class copy_stencil {
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

  struct stencil_19 : public sbase {

    // Members

    // Temporary storage typedefs
    using tmp_halo_t = gridtools::halo< 0,0, 0, 0, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t< 0, 5, tmp_halo_t >;
    using tmp_storage_t = storage_traits_t::data_store_t< float_type, tmp_meta_data_t>;
    const gridtools::clang::domain& m_dom;

    // storage declarations
    storage_ijk_t& m_in;
    storage_ijk_t& m_out;

    // temporary storage declarations
  public:

    stencil_19(const gridtools::clang::domain& dom_, storage_ijk_t& in_, storage_ijk_t& out_) : sbase("stencil_19"), m_dom(dom_), m_in(in_), m_out(out_){}

    ~stencil_19() {
    }

    void sync_storages() {
      m_in.sync();
      m_out.sync();
    }

    virtual void run() {

      // starting timers
      start();
      {;
      gridtools::data_view<storage_ijk_t> in= gridtools::make_device_view(m_in);
      gridtools::data_view<storage_ijk_t> out= gridtools::make_device_view(m_out);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,1+0,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 1 - 1) / 1;
      const unsigned int nbz = (m_dom.ksize()+4-1) / 4;
      dim3 blocks(nbx, nby, nbz);
      copy_stencil_stencil19_ms23_kernel<<<blocks, threads>>>(nx,ny,nz,m_in.strides()[1],m_in.strides()[2],(in.data()+m_in.get_storage_info_ptr()->index(in.begin<0>(), in.begin<1>(),0 )),(out.data()+m_out.get_storage_info_ptr()->index(out.begin<0>(), out.begin<1>(),0 )));
      };

      // stopping timers
      pause();
    }

    sbase* get_stencil() {
      return this;
    }
  };
  static constexpr const char* s_name = "copy_stencil";
  sbase* m_stencil_19;
public:

  copy_stencil(const copy_stencil&) = delete;

  // Members

  // Stencil-Data

  copy_stencil(const gridtools::clang::domain& dom, storage_ijk_t& in, storage_ijk_t& out) : m_stencil_19(new stencil_19(dom,in,out) ){}

  void run() {
    sync_storages();
    m_stencil_19->run();
;
    sync_storages();
  }

  void sync_storages() {
    m_stencil_19->sync_storages();
  }

  std::string get_name()  const {
    return std::string(s_name);
  }

  std::vector<sbase*> getStencils() {
    return std::vector<sbase*>({m_stencil_19});
  }

  void reset_meters() {
m_stencil_19->reset();  }
};
} // namespace cuda
} // namespace dawn_generated
