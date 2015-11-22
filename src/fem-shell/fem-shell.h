#ifndef FEMSHELL_H
#define FEMSHELL_H

// C++ include files that we need
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <math.h>

// libMesh includes
//#include "libmesh/perf_log.h"
#include "libmesh/getpot.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/gmsh_io.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe.h"
#include "libmesh/dof_map.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/elem.h"
#include "libmesh/zero_function.h"
#include "libmesh/dirichlet_boundaries.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// global variables
std::string in_filename;
std::string out_filename;
bool debug;
Real nu;
Real em;
Real thickness;
std::vector<DenseVector<Real> > forces;

DenseMatrix<Real> Dp, Dm;

// function prototypes:
void read_parameters(int argc, char **argv);

void initMaterialMatrices();
void initElement(const Elem **elem, DenseMatrix<Real> &transUV, DenseMatrix<Real> &trafo, DenseMatrix<Real> &dphi, Real *area);
void calcPlane(ElemType type, DenseMatrix<Real> &transUV, DenseMatrix<Real> &dphi, Real *area, DenseMatrix<Real> &Ke_m);
void calcPlate(ElemType type, DenseMatrix<Real> &dphi, Real *area, DenseMatrix<Real> &Ke_p);

void  evalBTri(DenseMatrix<Real>& C, Real L1, Real L2, DenseMatrix<Real> &dphi_p, DenseMatrix<Real> &out);
void evalBQuad(DenseMatrix<Real>& Hcoeffs, Real xi, Real eta, DenseMatrix<Real> &Jinv,   DenseMatrix<Real> &out);

void constructStiffnessMatrix(ElemType type, DenseMatrix<Real> &Ke_m, DenseMatrix<Real> &Ke_p, DenseMatrix<Real> &K_out);
void localToGlobalTrafo(ElemType type, DenseMatrix<Real> &trafo, DenseMatrix<Real> &Ke_inout);

void contribRHS(const Elem **elem, DenseVector<Real> &Fe, std::unordered_set<unsigned int> *processedNodes);

// Matrix and right-hand side assemble
void assemble_elasticity(EquationSystems &es, const std::string &system_name);

void writeOutput(Mesh &mesh, EquationSystems &es);

#endif // FEMSHELL_H
