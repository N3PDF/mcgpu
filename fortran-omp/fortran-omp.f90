program vegas_main
   use Vegas_mod, only: vegas
   implicit none
   integer, parameter :: dp = kind(1.d0)
   real(dp), external :: lepage_test
   integer :: n_dim, n_iter, n_events
   real(dp) :: res, sigma, time
   integer :: time_start, time_rate, time_end

   n_dim = 4
   n_iter = 5
   n_events = 1d6

   call system_clock(time_start, time_rate)
   call vegas(lepage_test, .true., n_dim, n_iter, n_events, res, sigma)
   call system_clock(time_end)

   print *, "integral total is: ", res, "+/-", sigma

   time = real(time_end-time_start)/real(time_rate)

   print *, "Time: ", time, "s"

end program
