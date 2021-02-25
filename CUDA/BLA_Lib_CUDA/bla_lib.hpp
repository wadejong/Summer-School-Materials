/* CUDA tutorial: Basic Linear Algebra (BLA) Library

!Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle)

!This file is part of CUDA BLA tutorial. */

#ifndef _BLA_LIB_HPP
#define _BLA_LIB_HPP

//#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "timer.hpp"
#include "memory.hpp"
#include "matrix.hpp"

namespace bla{

//Initialization:
void init();
//Shutdown:
void shutdown();

//Tests:
void test_bla();

} //namespace bla

#endif //_BLA_LIB_HPP
