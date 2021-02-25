/* CUDA tutorial: Basic Linear Algebra (BLA) Library

!Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle)

!This file is part of CUDA BLA tutorial. */

#include <chrono>

#include "timer.hpp"

namespace bla{

double time_sys_sec()
{
 auto stamp = std::chrono::system_clock::now(); //current time point
 auto durat = std::chrono::duration<double>(stamp.time_since_epoch()); //duration (sec) since the begining of the clock
 return durat.count(); //number of seconds
}

double time_high_sec()
{
 auto stamp = std::chrono::high_resolution_clock::now(); //current time point
 auto durat = std::chrono::duration<double>(stamp.time_since_epoch()); //duration (sec) since the begining of the clock
 return durat.count(); //number of seconds
}

} //namespace bla
