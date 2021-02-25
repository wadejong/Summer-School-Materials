/* CUDA tutorial: Basic Linear Algebra (BLA) Library

!Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle)

!This file is part of CUDA BLA tutorial. */

#ifndef _MEMORY_HPP
#define _MEMORY_HPP

namespace bla{

//Memory kinds:
enum class MemKind{
 Regular, //regular memory
 Pinned,  //pinned memory (only matters for Host)
 Mapped,  //mapped pinned memory (only matters for Host)
 Unified  //unified memory
};

//Allocates memory on any device (Host:-1; Device:>=0):
void * allocate(int device, size_t size, MemKind mem_kind);
//Deallocates memory on any device (Host:-1; Device:>=0):
void deallocate(int device, void * ptr, MemKind mem_kind);

} //namespace bla

#endif //_MEMORY_HPP
