module dawn_utils
use iso_c_binding

implicit none
  interface
    subroutine serialize_dense_cells(start_idx, end_idx, num_k, &
                                  dense_stride, field, stencil_name, &
                                  field_name, iteration) bind(c)
      use iso_c_binding
      integer(c_int), value :: start_idx
      integer(c_int), value :: end_idx
      integer(c_int), value :: num_k
      integer(c_int), value :: dense_stride
      real(c_double), dimension(*) :: field
      character(kind=c_char), dimension(*) :: stencil_name
      character(kind=c_char), dimension(*) :: field_name
      integer(c_int), value :: iteration
    end subroutine
    subroutine serialize_dense_edges(start_idx, end_idx, num_k, &
                                  dense_stride, field, stencil_name, &
                                  field_name, iteration) bind(c)
      use iso_c_binding
      integer(c_int), value :: start_idx
      integer(c_int), value :: end_idx
      integer(c_int), value :: num_k
      integer(c_int), value :: dense_stride
      real(c_double), dimension(*) :: field
      character(kind=c_char), dimension(*) :: stencil_name
      character(kind=c_char), dimension(*) :: field_name
      integer(c_int), value :: iteration
    end subroutine
    subroutine serialize_dense_verts(start_idx, end_idx, num_k, &
                                  dense_stride, field, stencil_name, &
                                  field_name, iteration) bind(c)
      use iso_c_binding
      integer(c_int), value :: start_idx
      integer(c_int), value :: end_idx
      integer(c_int), value :: num_k
      integer(c_int), value :: dense_stride
      real(c_double), dimension(*) :: field
      character(kind=c_char), dimension(*) :: stencil_name
      character(kind=c_char), dimension(*) :: field_name
      integer(c_int), value :: iteration
    end subroutine

    subroutine set_splitter_index_lower(mesh, loc, space, offset, index) bind(c)
      use, intrinsic :: iso_c_binding
      type(c_ptr), value, target :: mesh
      integer(c_int), value :: loc
      integer(c_int), value :: space
      integer(c_int), value :: offset
      integer(c_int), value :: index
    end subroutine

    subroutine set_splitter_index_upper(mesh, loc, space, offset, index) bind(c)
      use, intrinsic :: iso_c_binding
      type(c_ptr), value, target :: mesh
      integer(c_int), value :: loc
      integer(c_int), value :: space
      integer(c_int), value :: offset
      integer(c_int), value :: index
    end subroutine

  end interface
  contains
    TYPE, PRIVATE :: LocationType_Values 
        INTEGER :: Cell ! = 0
        INTEGER :: Edge ! = 1
        INTEGER :: Vertex ! = 2
      END TYPE LocationType_Values
      !
    TYPE (LocationType_Values), PUBLIC, PARAMETER :: LocationType = LocationType_Values (0,1,2)
    TYPE, PRIVATE :: SubdomainMarker_Values 
      INTEGER :: LB ! = 0
      INTEGER :: Nudging ! = 1000
      INTEGER :: Interior ! = 2000
      INTEGER :: Halo ! = 3000
      INTEGER :: End ! = 4000
    END TYPE SubdomainMarker_Values
    !
    TYPE (SubdomainMarker_Values), PUBLIC, PARAMETER :: SubdomainMarker = SubdomainMarker_Values (0,1000,2000,3000,4000)
end module