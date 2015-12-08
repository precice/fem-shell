FEM-shell
========

Features
-----------

- Structure solver build with libMesh FEM framework
- Parallelized with MPI
- Stand-alone and coupled version. Coupling achieved with preCICE (https://github.com/precice)
- Implements different types of flat shell element
- Efficient solving with PETSc support
- Applicable in multi-physics simulations, like fluid-structure interactions (FSI)
- Additional tool to generate meshes (meshGen)

Installation
-----------

First, compile `PETSc`. A description for the compilation of PETSc can be found here: http://www.mcs.anl.gov/petsc/documentation/installation.html

``` bash
git clone -b maint https://bitbucket.org/petsc/petsc petsc
cd petsc
./configure
make all test
```

Second, compile `libMesh`. A description for the compilation of libMesh can be found here: http://libmesh.github.io/installation.html

``` bash
git clone git://github.com/libMesh/libmesh.git
cd libmesh
./configure
make
```

Third, compile the two FEM-shell versions (and meshGen) by executing the corresponding SConstructs

``` bash
cd src/meshgen
scons
cd ../fem-shell
scons
cd preCICE
scons
```

Requirements
-----------

gcc 4.8 or higher due to C++11 features.

Example runs
-----------

To test the program, several example problems are provided with the code. These tests can be run automatically be executing the two bash-scripts as follows:

``` bash
cd src/fem-shell
./run_examples.sh
cd preCICE
./run_example.sh
```

Credits
-----------

The FEM-shell project has been created by Stephan Herb within the scope of his master thesis