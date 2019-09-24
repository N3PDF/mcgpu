function lepage_test(x, n)
   implicit none
   integer, parameter :: dp = kind(1d0)
   real(dp), parameter :: pi = 3.141592653589793238d0
   real(dp) :: lepage_test
   integer, intent(in) :: n
   real(dp), dimension(n), intent(in) :: x

   real(dp) :: a, pref, coef
   integer :: i

   ! Result after integration = 1.0

   a = 0.1d0
   pref = (1d0/a/dsqrt(pi))**n
   coef = 0d0
   do i = 1, 100*n
      coef = coef + float(i)
   enddo
   do i = 1, n
      coef = coef + (x(i) - 1d0/2d0)**2/a**2
   enddo
   coef = coef - float(100*n)*float(100*n+1)/2d0

   lepage_test = pref*exp(-coef)

end function lepage_test
