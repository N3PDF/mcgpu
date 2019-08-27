program vegas_main
   use Vegas_mod, only: vegas
   implicit none
   integer, parameter :: dp = kind(1.d0)
   real(dp), external :: lepage_test
   integer :: n_dim, n_iter, n_events
   real(dp) :: res, sigma

   n_dim = 3
   n_iter = 10
   n_events = 1d5

   call vegas(lepage_test, n_dim, n_iter, n_events, res, sigma)

   print *, "integral total is: ", res, "+/-", sigma

end program
