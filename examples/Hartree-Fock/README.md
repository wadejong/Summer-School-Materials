Implments an integral-direct, self-consistent field (SCF) algorithm for Hartree-Fock.  [LIBINT](https://github.com/evaleev/libint) is used to calculate the integrals.

The command (`scf`) takes a single argument which should be the name of the file holding the molecular geometry in XYZ format.

By far the most intense computation is the two-electron contribution to the Fock matrix

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\sum&#95;{kl}{D&#95;{kl}\left(2(ij|kl)-(ik|jl)\right)}" title="Amdahl" />

Two algorithms are provided for this.  The simple one is easiest to understand but it does not use the 8-fold permutational symmetry of the integrals.  The more complex algorithm is much more efficient but is a bit harder to understand.  Fortunately, it is possible to parallelize at a coarse scale that in a manner that is oblivious to these details.





