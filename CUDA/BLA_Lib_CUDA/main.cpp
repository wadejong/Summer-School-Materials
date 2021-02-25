/* CUDA tutorial: Basic Linear Algebra (BLA) Library

!Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle)

!This file is part of CUDA BLA tutorial. */


#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>

#include "bla_lib.hpp"

void use_bla()
{
 std::cout << "Let's try to use BLA library ..." << std::endl;
 //Create matrix A:
 bla::Matrix<float> A(1000,2000);
 //Allocate matrix A body on Host:
 A.allocateBody(-1,bla::MemKind::Pinned);
 //Set matrix A body to some value:
 A.setBodyHost();

 //Create matrix B:
 bla::Matrix<float> B(2000,3000);
 //Allocate matrix B body on Host:
 B.allocateBody(-1,bla::MemKind::Pinned);
 //Set matrix B body to some value:
 B.setBodyHost();

 //Create matrix C:
 bla::Matrix<float> C(1000,3000);
 //Allocate matrix C body on GPU#0:
 C.allocateBody(0,bla::MemKind::Pinned);
 //Set matrix C body to zero:
 C.zeroBody(0);

 std::cout << "Seems like it works!" << std::endl;
 return;
}


int main(int argc, char ** argv)
{

//Init the BLA library:
 bla::init();

//Test the BLA library:
 bla::test_bla();

//Use the BLA library:
 use_bla();

//Shutdown the BLA library:
 bla::shutdown();

 return 0;
}
