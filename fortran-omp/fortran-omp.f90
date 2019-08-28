program vegas_main
   use Vegas_mod, only: vegas
   implicit none
   integer, parameter :: dp = kind(1.d0)
   real(dp), external :: lepage_test
   integer :: n_dim, n_iter, n_events
   real(dp) :: res, sigma

   n_dim = 6
   n_iter = 10
   n_events = 1d6

   call vegas(lepage_test, .true., n_dim, n_iter, n_events, res, sigma)

   print *, "integral total is: ", res, "+/-", sigma

end program
