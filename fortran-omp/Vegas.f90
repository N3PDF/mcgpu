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
   integer, parameter :: NDMX = 10

   ! Divisions of the Vegas grid
   real(dp), dimension(:, :), allocatable :: divisions

   public :: vegas

   contains
      real(dp) function internal_rand()
         internal_rand = rand()
      end function internal_rand

      subroutine generate_random_array(n_dim, x, wgt, div_index)
         integer, intent(in) :: n_dim
         real(dp), dimension(n_dim), intent(out) :: x
         real(dp), intent(out) :: wgt
         integer, dimension(n_dim), intent(out) :: div_index
         integer :: i, int_xn
         real(dp) :: rn, x_n, aux_rand, x_ini, xdelta, rand_x
         ! NOTE: I assume here the region of integration to be 0 to 1 in all dimensions
         real(dp), parameter :: reg_i = 0d0, reg_f = 1d0
         ! Subrotuine to generate random numbers
         ! Arguments:
         ! n_dim: number of dimensions
         ! x: array of dimension n_dim with a random number per dimension
         ! wgt: weight of the point
         ! div_index; array of dimension n_dim with the index of the grid-division
         !            corresponding each of the random numbers generated
         wgt = 1d0
         do i = 1, n_dim
            rn = internal_rand()
            ! Ger a random number randomly assigned to one of the subdivisions
            x_n = 1d0 + NDMX*(1d0 - rn)
            int_xn = max(1, min( int(x_n), NDMX ))
            ! in practice int_xn = int(xn) unless x_n < 1
            aux_rand = x_n - int_xn 
            if (int_xn == 1) then
               x_ini = 0d0
            else
               x_ini = divisions(int_xn - 1, i)
            endif
            xdelta = divisions(int_xn, i) - x_ini
            rand_x = x_ini + xdelta*aux_rand
            x(i) = reg_i + rand_x*(reg_f - reg_i)
            wgt = wgt*xdelta*NDMX
            div_index(i) = int_xn
         enddo
      end subroutine generate_random_array

      subroutine rebin(rc, rw, subdivisions)
         real(dp), intent(in) :: rc
         real(dp), intent(in), dimension(NDMX) :: rw
         real(dp), intent(inout), dimension(NDMX) :: subdivisions
         integer :: i, k
         real(dp) :: dr, old_xf, old_xi
         real(dp), dimension(NDMX) :: aux
         ! Reweight the integration subdivisions according to
         ! the vector rw.
         ! This functon should be called by every dimension at the end
         ! of each warmup iteraton
         k = 0
         dr = 0
         do i = 1, NDMX-1
            do while (rc > dr) 
               k = k + 1
               dr = dr + rw(k)
            enddo
            if (k > 1) then
               old_xi = subdivisions(k-1)
            else
               old_xi = 0d0
            endif
            old_xf = subdivisions(k)
            dr = dr - rc
            aux(i) = old_xf - (old_xf - old_xi)*(dr / rw(k))
         enddo
         subdivisions(1:NDMX-1) = aux(1:NDMX-1)
         subdivisions(NDMX) = 1d0
      end subroutine rebin

      subroutine refine_grid(res_sq)
         real(dp), dimension(NDMX), intent(in) :: res_sq
         real(dp), dimension(NDMX) :: aux, rw
         integer :: i
         real(dp) :: rc, aux_sum
         rc = 0d0
         ! First we smear out the array div_sq, where we have store
         ! the value of f^2 for each sub_division for each dimension
         aux(1) = (res_sq(1) + res_sq(2))/2d0
         aux_sum = aux(1)
         do i = 2, NDMX - 1
            aux(i) = (res_sq(i-1) + res_sq(i) + res_sq(i+1))/3d0
            aux_sum = aux_sum + aux(i)
         enddo
         aux(NDMX) = (res_sq(NDMX-1) + res_sq(NDMX))/2d0
         aux_sum = aux_sum + aux(NDMX)
         ! Now we refine the grid according to 
         ! journal of comp phys, 27, 192-203 (1978) G.P. Lepage
         do i = 1, NDMX
            if (aux(i) < 1d-30) then
               aux(i) = 1d-30
            endif
            rw(i) = ( (1d0 - aux(i)/aux_sum)/(dlog(aux_sum) - dlog(aux(i))) )**ALPHA
            rc = rc + rw(i)
         enddo
         call rebin(rc/NDMX, rw, divisions)
      end subroutine

      subroutine vegas(f_integrand, warmup, n_dim, n_iter, n_events, final_result, sigma)
         real(dp), external :: f_integrand
         logical, intent(in) :: warmup
         integer, intent(in) :: n_dim, n_iter, n_events
         real(dp), intent(out) :: final_result, sigma

         ! Counters
         integer :: i, j, k
         integer :: ev_counter, ind
         integer, dimension(n_dim) :: div_index

         ! Results and weights
         real(dp), dimension(n_dim) :: x
         real(dp) :: wgt, xjac, xwgt
         real(dp) :: res, res2
         real(dp), dimension(n_iter, 2) :: all_results

         ! Grid storage
         real(dp), dimension(NDMX, n_dim) :: arr_res2

         ! Temporary variables
         real(dp) :: tmp, tmp2, err_tmp2
         real(dp) :: weight_sum, aux_result, weight_tmp
         real(dp), dimension(NDMX) :: rw

         print*, "Starting Fortran Vegas"

         !$ print *, " $ OMP active "
         !$ print *, " $ Maximum number of threads: ", OMP_get_num_procs()
         !$ print *, " $ Number of threads selected: ", OMP_get_max_threads()

         ! Initialize variables
         xjac = 1d0/n_events
         allocate( divisions(NDMX, n_dim) )
         divisions(1, :) = 1d0
         divisions(2:, :) = 0d0
         rw(:) = 1d0
         do j = 1, n_dim
            call rebin(1d0/NDMX, rw, divisions(:, j))
         enddo

         !  Iterations loop
         do k = 1, n_iter
            ev_counter = 0
            res2 = 0d0
            res = 0d0
            arr_res2(:,:) = 0d0

            !$omp parallel private(tmp, tmp2, xwgt, wgt, x)
            !$omp do schedule (dynamic)
            do i = 1, n_events
               ! Generate a random vector of dimensions n_dim
               !$omp critical
               call generate_random_array(n_dim, x, xwgt, div_index)
               !$omp end critical
               wgt = xjac*xwgt

               ! Call the integrand
               tmp = wgt*f_integrand(x, n_dim)
               tmp2 = tmp**2

               res = res + tmp
               res2 = res2 + tmp2

               if (warmup) then
                  do j = 1, n_dim
                     ind = div_index(j)
                     arr_res2(ind, j) = arr_res2(ind, j) + tmp2
                  enddo
               endif
                     
            enddo
            !$omp enddo
            !$omp end parallel

            ! Compute the error
            err_tmp2 = max((n_events*res2 - res**2)/(n_events - 1d0), 1d-30)
            sigma = dsqrt(err_tmp2)

            print *, "Result for iteration", k, ": ", res, " +/- ", sigma
            all_results(k, 1) = res
            all_results(k, 2) = sigma
            if (warmup) then
               do j = 1, n_dim
                  call refine_grid(arr_res2(:, j))
               enddo
            endif
         enddo

         ! Compute the final results
         weight_sum = 0d0
         aux_result = 0d0
         do k = 1, n_iter
            res = all_results(k,1)
            sigma = all_results(k,2)
            weight_tmp = 1d0/sigma**2
            aux_result = aux_result + res*weight_tmp
            weight_sum = weight_sum + weight_tmp
         enddo
         final_result = aux_result / weight_sum
         sigma = dsqrt(1d0/weight_sum)

         deallocate(divisions)

      end subroutine
end module vegas_mod
