module vegas_mod
#ifdef USE_IFORT
   use ifport
#endif
!$ use omp_lib
! Vegas module
! Public: vegas(f_integrand, number of dimensions, number of iterations, number of events per iterations, final_result, sigma)
   implicit none
   private

   integer, parameter :: dp = kind(1.d0)
   ! Damping parameter, alpha = 0 -> no adaptation
   real(dp), parameter :: ALPHA = 1.5d0
   ! Subdivisions of the Vegas grid per dimension
   integer, parameter :: NDMX = 50

   public :: vegas

   contains
      real(dp) function internal_rand()
         internal_rand = rand()
      end function internal_rand

      subroutine generate_random_array(n_dim, x, wgt)
         integer, intent(in) :: n_dim
         real(dp), dimension(n_dim), intent(out) :: x
         real(dp), intent(out) :: wgt
         integer :: i

         do i = 1, n_dim
            x(i) = internal_rand()
         enddo
         wgt = 1
      end subroutine generate_random_array

      subroutine vegas(f_integrand, n_dim, n_iter, n_events, final_result, sigma)
         real(dp), external :: f_integrand
         integer, intent(in) :: n_dim, n_iter, n_events
         real(dp), intent(out) :: final_result, sigma

         integer :: i, k
         integer :: ev_counter

         real(dp), dimension(n_dim) :: x
         real(dp) :: wgt, xjac, xwgt
         real(dp) :: res, res2
         ! Temporary variables
         real(dp) :: tmp, tmp2, err_tmp2

         print*, "Starting Fortran Vegas"

         !$ print *, " $ OMP active "
         !$ print *, " $ Maximum number of threads: ", OMP_get_num_procs()
         !$ print *, " $ Number of threads selected: ", OMP_get_max_threads()

         xjac = 1d0/n_events
         do k = 1, n_iter
            ev_counter = 0
            res2 = 0
            res = 0

            !$omp parallel private(tmp, tmp2, xwgt, wgt, x)
            !$omp do schedule (dynamic)
            do i = 1, n_events
               ! Generate a random vector of dimensions n_dim
               !$omp critical
               call generate_random_array(n_dim, x, xwgt)
               !$omp end critical
               wgt = xjac*xwgt

               ! Call the integrand
               tmp = wgt*f_integrand(x, n_dim)
               tmp2 = tmp**2

               res = res + tmp
               res2 = res2 + tmp2
            enddo
            !$omp enddo
            !$omp end parallel

            ! Compute the error
            err_tmp2 = max((n_events*res2 - res**2)/(n_events - 1d0), 1d-30)
            sigma = dsqrt(err_tmp2)

            print *, "Result for iteration", k, ": ", res, " +/- ", sigma
            ! TODO: refine the grid
         enddo

         final_result = res

      end subroutine
end module vegas_mod






