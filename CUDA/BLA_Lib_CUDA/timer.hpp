/* CUDA tutorial: Basic Linear Algebra (BLA) Library

!Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle)

!This file is part of CUDA BLA tutorial. */

#ifndef _TIMER_HPP
#define _TIMER_HPP

namespace bla{

double time_sys_sec();  //system time stamp in seconds (thread-global)
double time_high_sec(); //high-resolution time stamp in seconds

} //namespace bla

#endif //_TIMER_HPP
