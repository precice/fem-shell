/*************************************************************
 * todo
 *
 *  Code preCICE-ready machen
 *  Gegenstück solver besorgen
 *  Testen!
 *  Ergebnis aufzeichnen
 * ***********************************************************/

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
//#include "libmesh/vtk_io.h"
//#include "libmesh/gmsh_io.h"
//#include "libmesh/gnuplot_io.h"
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
//#include "libmesh/mesh_generation.h"
//#include "libmesh/boundary_info.h"
//#include "libmesh/string_to_enum.h"
//#include "libmesh/quadrature_gauss.h"
//#include "libmesh/dense_submatrix.h"
//#include "libmesh/dense_subvector.h"

// preCICE includes
#include "precice/SolverInterface.hpp"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// Matrix and right-hand side assemble
void assemble_elasticity(EquationSystems& es,
                         const std::string& system_name);

void eval_B(DenseMatrix<Real>& Hcoeffs, Real L1, Real L2, DenseMatrix<Real> &dphi_p, DenseMatrix<Real> &out);
void eval_B_quad(DenseMatrix<Real>& Hcoeffs, Real xi, Real eta, DenseMatrix<Real> &Jinv, DenseMatrix<Real> &out);

bool debug = false;
Real nu = 0.3;
Real em = 1.0e6;
Real thickness = 1.0;
std::vector<DenseVector<Real> > forces;

// Begin the main program.
int main (int argc, char** argv)
{
    // Initialize libMesh and any dependent libaries
    LibMeshInit init (argc, argv);

    // Initialize the cantilever mesh
    const unsigned int dim = 2;

    std::cout << "Start of process " << global_processor_id() << "\n";

    if (argc != 8)
    {
        err << "Usage: " << argv[0] << " -d -nu -e -t -mesh -out -config\n"
            << "-d: Debug-Mode (1=on, 0=off (default))\n"
            << "-nu: Possion-Number (0.3 default)\n"
            << "-e: Elasticity Modulus (1.0e6 default)\n"
            << "-t: Thickness (1.0 default)\n"
            << "-mesh: Input mesh file (*.xda or *.msh)\n"
            << "-out: Output file name (without extension)\n"
            << "-config: preCICE configuration file name\n";

        libmesh_error_msg("Error, must choose valid parameters.");
    }

    // Parse command line
    GetPot command_line (argc, argv);

    debug = false;
    if ( command_line.search(1, "-d") )
        debug = command_line.next(0) == 1? true : false;
    nu = 0.3;
    if ( command_line.search(1, "-nu") )
        nu = command_line.next(nu);
    em = 1.0e6;
    if ( command_line.search(1, "-e") )
        em = command_line.next(em);
    thickness = 1.0;
    if ( command_line.search(1, "-t") )
        thickness = command_line.next(thickness);
    std::string filename;
    if ( command_line.search(1, "-mesh") )
        filename = command_line.next("1_tri.xda");
    std::string outfile;
    if ( command_line.search(1, "-out") )
        outfile = command_line.next("out");
    std::string configFileName;
    if ( command_line.search(1, "-config") )
        configFileName = command_line.next("");

    SolverInterface *interface = NULL;
    if (global_processor_id() == 0) // TODO: schauen ob das so funktioniert. SolverInterface darf nur von rank 0 benutzt werden (im MPI Fall)
    {
        std::string solverName = "STRUCTURE";
        interface = new SolverInterface(solverName, 0, 1);
        interface->configure(configFileName);
    }

    // Create a 2D mesh distributed across the default MPI communicator.
    Mesh mesh(init.comm(), interface.getDimensions()-1);
    mesh.allow_renumbering(false);
    if (mesh.allow_renumbering())
        std::cout << "mesh erlaubt renumbering\n";
    mesh.read(filename);
    // TODO (MPI): schauen, ob und wie man automatisches partioning besser machen kann und vor allen zu welchem Zeitpunkt

    if (filename.find(".msh") != std::string::npos) // if we load a GMSH mesh file, we need to execute a preparation step
    {
        std::cout << "we use gmsh\n";
        //mesh.prepare_for_use(true, false);// skip renumbering, skip find neighbors (depricated)
    }

    // Print information about the mesh to the screen.
    if (debug)
        mesh.print_info();

    /** mit preCICE nicht mehr nötig
    // Load file with forces (only needed for stand-alone version)
    std::filebuf fb;
    if (filename.find(".xda") != std::string::npos ||
        filename.find(".msh") != std::string::npos)
        filename.resize(filename.size()-4);

    filename += "_f";

    // MPI TODO: 0 lädt und broadcastet an alle anderen
    if (fb.open (filename.c_str(),std::ios::in))
    {
        std::istream input(&fb);
        int n_Forces;
        input >> n_Forces;
        double factor = 1.0;
        input >> factor;
        for (int i = 0; i < n_Forces; i++)
        {
            DenseVector<Real> p(3);
            for (int j = 0; j < 3; j++)
                input >> p(j);
            p *= factor;
            forces.push_back(p);
        }
    }
    **/

    // stattdessen kommt hier die preCICE Initialisierung hin:
    int n_elements = mesh.n_nodes(); // number of nodes in the mesh
    int dimensions = 0;
    double *forces, *displacements, *grid;
    int meshID, dID, fID;
    int *vertexIDs;
    if (interface != NULL)
    {
        dimensions = interface.getDimensions();
        forces = new double[n_elements*3];
        displacements = new double[n_elements*3];
        grid = new double[n_elements]; // TODO: WAS REPRÄSENTIERT grid IN preCICE WIRKLICH? WAS IST DER UNTERSCHIED FÜR preCICE ZWISCHEN node UND element?
                                       // PROBLEM IST NÄMLICH: WERTE WERDEN JA NICHT PRO ELEMENT SONDERN PRO KNOTEN GESPEICHERT...
        for (int i = 0; i < n_elements; i++)
        {
            forces[i*3] = 0.0;
            forces[i*3+1] = 0.0;
            forces[i*3+2] = 0.0;
            displacements[i*3] = 0.0;
            displacements[i*3+1] = 0.0;
            displacements[i*3+2] = 0.0;
            for (int dim = 0; dim < dimensions; dim++) // TODO: WAS MACH ICH HIER?
                grid[i*dimensions + dim] = i;
        }

        meshID = interface->getMeshID("Structure_Nodes");
        dID    = interface->getDataID("Displacements", meshID);
        fID    = interface->getDataID("Forces", meshID);

        vertexIDs = new int[n_elements];
        interface->setMeshVertices(meshID, n_elements, grid, vertexIDs);

        interface->initialize();

        if (interface.isActionRequired(actionWriteInitialData()))
        {
            interface->writeBlockScalarData(dID, n_elements, vertexIDs, displacements);
            interface->fulfilledAction(actionWriteInitialData());
        }

        interface->initializeData();

        if (interface->isReadDataAvailable())
        {
            interface->readBlockScalarData(fID, n_elements, vertexIDs, forces);
        }
    }

    // Create an equation systems object.
    EquationSystems equation_systems (mesh);

    // Declare the system and its variables.
    // Create a system named "Elasticity"
    LinearImplicitSystem& system = equation_systems.add_system<LinearImplicitSystem> ("Elasticity");

    // Add three displacement variables, u, v and w,
    // as well as three drilling variables theta_x, theta_y and theta_z to the system
    unsigned int u_var  = system.add_variable("u",  FIRST, LAGRANGE);
    unsigned int v_var  = system.add_variable("v",  FIRST, LAGRANGE);
    unsigned int w_var  = system.add_variable("w",  FIRST, LAGRANGE);
    unsigned int tx_var = system.add_variable("tx", FIRST, LAGRANGE);
    unsigned int ty_var = system.add_variable("ty", FIRST, LAGRANGE);
    unsigned int tz_var = system.add_variable("tz", FIRST, LAGRANGE);

    system.attach_assemble_function (assemble_elasticity);

    // Construct a Dirichlet boundary condition object
    // We impose a "clamped" boundary condition on the
    // nodes with bc_id = 0
    std::set<boundary_id_type> boundary_ids;
    boundary_ids.insert(1);
    // Create a vector storing the variable numbers which the BC applies to
    std::vector<unsigned int> variables(6);
    variables[0] = u_var;  variables[1] = v_var;  variables[2] = w_var;
    variables[3] = tx_var; variables[4] = ty_var; variables[5] = tz_var;
    // Create a ZeroFunction to initialize dirichlet_bc
    ConstFunction<Number> cf(0.0);
    DirichletBoundary dirichlet_bc(boundary_ids,variables,&cf);
    // We must add the Dirichlet boundary condition _before_ we call equation_systems.init()
    system.get_dof_map().add_dirichlet_boundary(dirichlet_bc);

    std::set<boundary_id_type> boundary_ids2;
    boundary_ids2.insert(0);
    std::vector<unsigned int> variables2(3);
    variables2[0] = u_var; variables2[1] = v_var; variables2[2] = w_var;
    ConstFunction<Number> cf2(0.0);
    DirichletBoundary dirichlet_bc2(boundary_ids2,variables2,&cf2);
    system.get_dof_map().add_dirichlet_boundary(dirichlet_bc2);

    // TODO: PROBLEM: andere Prozesse haben keinen Zugriff auf interface, müssen aber auch in die while-Schleife
    // eigene bool von rank 0 gesteuert über broadcast verteilt könnte funktionieren
    bool ongoing = false;
    if (global_processor_id() == 0)
        ongoing = interface->isCouplingOngoing();
    Communicator::broadcast(ongoing);
    while (ongoing)
    {
        // Initialize the data structures for the equation system.
        equation_systems.reinit();

        // Print information about the system to the screen.
        //equation_systems.print_info();

        //const Real tol = equation_systems.parameters.get<Real>("linear solver tolerance");
        //const unsigned int maxits = equation_systems.parameters.get<unsigned int>("linear solver maximum iterations");
        //equation_systems.parameters.set<unsigned int>("linear solver maximum iterations") = maxits*3;
        //equation_systems.parameters.set<Real>        ("linear solver tolerance") = tol/1000.0;

        /**
         * Solve the system
         **/
        equation_systems.solve();

        std::vector<Number> sols;
        equation_systems.build_solution_vector(sols);

        if (debug)
        {
            system.matrix->print(std::cout);
            system.rhs->print(std::cout);
        }

        // be sure that only the master process (id = 0) act on the solution, since the rest of the processes only see their own partial solution
        if (global_processor_id() == 0)
        {
            //std::cout << global_processor_id() << ": Solution: x=[";
            MeshBase::const_node_iterator no = mesh.nodes_begin();
            const MeshBase::const_node_iterator end_no = mesh.nodes_end();
            for (int i = 0 ; no != end_no; ++no,++i)
            {
               //std::cout << "uvw_" << i << " = " << sols[6*i] << ", " << sols[6*i+1] << ", " << sols[6*i+2] << "\n";
               // copy results into preCICE exchange array:
               displacements[3*i] = sols[6*i];
               displacements[3*i+1] = sols[6*i+1];
               displacements[3*i+2] = sols[6*i+2];
            }
            //std::cout << "]\n" << std::endl;

            interface->writeBlockScalarData(dID, n_elements, vertexIDs, displacements);
            interface->readBlockScalarData(fID, n_elements, vertexIDs, forces);

            if (interface->isActionRequired(actionReadIterationCheckpoint())) // i.e. not yet converged
            {
                interface->fulfilledAction(actionReadIterationCheckpoint());
            }
            else
            {
                // ???
            }
        }

        if (global_processor_id() == 0)
            ongoing = interface->isCouplingOngoing();
        Communicator::broadcast(ongoing);
    }

    /*
    // Plot the solution
    std::ostringstream file_name;
    file_name << "out/" << outfile << "_"
              << std::setw(3)
              << std::setfill('0')
              << std::right
              << ".e";

    ExodusII_IO (mesh).write_equation_systems(file_name.str(),equation_systems);
    */

    std::cout << "All done ;)\n";
    if (global_processor_id() == 0)
    {
        interface->finalize();
    }

    return 0;
}

void assemble_elasticity(EquationSystems& es,
                         const std::string& system_name)
{
    libmesh_assert_equal_to (system_name, "Elasticity");

    // get the mesh
    const MeshBase& mesh = es.get_mesh();

    // get a reference to the system
    LinearImplicitSystem& system = es.get_system<LinearImplicitSystem>("Elasticity");

    // get the ids for the unknowns
    const unsigned int u_var = system.variable_number ("u");
    const unsigned int v_var = system.variable_number ("v");
    const unsigned int w_var = system.variable_number ("w");

    // A reference to the DofMap object for this system.
    // The DofMap object handles the index translation from node and element numbers to degree of freedom numbers.
    const DofMap& dof_map = system.get_dof_map();

    // stiffness matrix Ke_m for element (membrane part),
    //                  Ke_p for element (plate part)
    DenseMatrix<Number> Ke;
    DenseVector<Number> Fe;

    DenseMatrix<Number> Ke_m, Ke_p;

    //
    std::vector<dof_id_type> dof_indices;
    std::vector<dof_id_type> dof_indices_u;
    std::vector<dof_id_type> dof_indices_v;
    std::vector<dof_id_type> dof_indices_w;

    // iterator for iterating through the elements of the mesh:
    MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

    Node *ndi = nullptr, *ndj = nullptr;
    Node U, V, W;
    DenseMatrix<Real> transTri;
    DenseMatrix<Real> transUV, dphi_p; // xij, yij
    std::vector<Real> sidelen; // lij^2
    DenseMatrix<Real> Hcoeffs; // ak, ..., ek
    Real area;
    DenseMatrix<Real> B;

    DenseMatrix<Real> Dp, Dm;
    Dp.resize(3,3);
    Dp(0,0) = 1.0; Dp(0,1) = nu;
    Dp(1,0) = nu;  Dp(1,1) = 1.0;
    Dp(2,2) = (1.0-nu)/2.0;
    Dm = Dp;
    Dp *= em*pow(thickness,3.0)/(12.0*(1.0-nu*nu)); // material matrix for plate part
    Dm *= em/(1.0-nu*nu); // material matrix for membrane part
    std::vector<std::vector<Real> > qps;

    std::unordered_set<dof_id_type> processedNodes;
    processedNodes.reserve(mesh.n_local_nodes());

    // for all local elements elem in mesh do:
    for ( ; el != end_el; ++el)
    {
        if (debug)
            std::cout << "START ELEMENT\n";

        const Elem* elem = *el;

        // get the local to global DOF-mappings for this element
        dof_map.dof_indices (elem, dof_indices);
        dof_map.dof_indices (elem, dof_indices_u, u_var);
        dof_map.dof_indices (elem, dof_indices_v, v_var);
        dof_map.dof_indices (elem, dof_indices_w, w_var);

        ElemType type = elem->type();
        if (type == TRI3)
        {
            /**
              * HIER KOMMT DIE BERECHNUNG VON STIFFNESS-MATRIX UND RHS FÜR DREIECKE REIN
              **/
            // resize the current element matrix and vector to an appropriate size
            Ke.resize (18, 18);
            Fe.resize (18);

            // transform arbirtrary 3d triangle down to xy-plane with node a at origin (implicit):
            ndi = elem->get_node(0); // node A
            std::cout << "node A:\n"; ndi->print_info(std::cout);
            ndj = elem->get_node(1); // node B
            std::cout << "node B:\n"; ndj->print_info(std::cout);
            U = (*ndj)-(*ndi); // U = B-A
            ndj = elem->get_node(2); // node C
            std::cout << "node C:\n"; ndj->print_info(std::cout);
            V = (*ndj)-(*ndi); // V = C-A
            transUV.resize(3,2);
            for (int i = 0; i < 3; i++)
            { // node A lies in local origin (per definition)
                transUV(i,0) = U(i); // node B in global coordinates (triangle translated s.t. A lies in origin)
                transUV(i,1) = V(i); // node C in global coordinates ( -"- )
            }
            /* transUV [ b_x, c_x ]
             *         [ b_y, c_y ]
             *         [ b_z, c_z ]
             */

            // area of triangle is half the length of the cross product of U and V
            W = U.cross(V);
            area = 0.5*W.size();

            U = U.unit();   // local x-axis unit vector
            W = W.unit();   // local z-axis unit vector, normal to triangle
            V = W.cross(U); // local y-axis unit vector (cross prod of 2 normalized vectors is automatically normalized)
        }
        else if (type == QUAD4)
        {
            /**
              * HIER KOMMT DIE BERECHNUNG VON STIFFNESS-MATRIX UND RHS FÜR VIERECKE REIN
              **/
            // resize the current element matrix and vector to an appropriate size
            Ke.resize (24, 24);
            Fe.resize (24);

            // transform planar 3d quadrilateral down to xy-plane with node A at origin:
            Node nI,nJ,nK,nL;
            ndi = elem->get_node(0); // node A
            //std::cout << "node A (" << ndi->processor_id() << "):\n"; ndi->print_info(std::cout);}
            ndj = elem->get_node(1); // node B
            //std::cout << "node B (" << ndj->processor_id() << "):\n"; ndj->print_info(std::cout);}
            nI = (*ndi) + 0.5*((*ndj)-(*ndi)); // nI = midpoint on edge AB
            ndi = elem->get_node(2); // node C
            //std::cout << "node C (" << ndi->processor_id() << "):\n"; ndi->print_info(std::cout);}
            nJ = (*ndj) + 0.5*((*ndi)-(*ndj)); // nJ = midpoint on edge BC
            ndj = elem->get_node(3); // node D
            //std::cout << "node D (" << ndj->processor_id() << "):\n"; ndj->print_info(std::cout);}
            nK = (*ndi) + 0.5*((*ndj)-(*ndi)); // nK = midpoint on edge CD
            ndi = elem->get_node(0); // node A
            nL = (*ndj) + 0.5*((*ndi)-(*ndj)); // nL = midpoint on edge DA
            ndj = elem->get_node(2);

            //std::cout << "node i:\n"; nI.print_info(std::cout);
            //std::cout << "node j:\n"; nJ.print_info(std::cout);
            //std::cout << "node k:\n"; nK.print_info(std::cout);
            //std::cout << "node l:\n"; nL.print_info(std::cout);

            transUV.resize(3,4); // ({x,y,z},{A,B,C,D})
            for (int i = 0; i < 4; i++)
            {
                ndi = elem->get_node(i);
                transUV(0,i) = (*ndi)(0); // coord x in global coordinates
                transUV(1,i) = (*ndi)(1); // coord y in global coordinates
                transUV(2,i) = (*ndi)(2); // coord z in global coordinates
            }
            /* transUV [ a_x, b_x, c_x, d_x ]
             *         [ a_y, b_y, c_y, d_y ]
             *         [ a_z, b_z, c_z, d_z ]
             */

            U = nJ-nL; // Vx
            U = U.unit(); // Vx normalized -> local x-axis unit vector
            W = nK-nI; // Vr
            W = U.cross(W); // Vz = Vx x Vr
            W = W.unit(); // Vz normalized -> local z-axis unit vector
            V = W.cross(U); // Vy = Vz x Vx -> local y-axis unit vector
        }
        else
        {
            /**
              * WIR HABEN EIN (NOCH) UNGÜLTIGES ELEMENT. ABBRUCH, DA WIR DAS MESH NICHT BENUTZEN KÖNNEN
              **/
            continue;
        }

        transTri.resize(3,3); // global to local transformation matrix
        for (int j = 0; j < 3; j++)
        {
            transTri(0,j) = U(j);
            transTri(1,j) = V(j);
            transTri(2,j) = W(j);
        }
        /* transTri [ u_x, u_y, u_z ]
         *          [ v_x, v_y, v_z ]
         *          [ w_x, w_y, w_z ]
         */

        // transform B and C (and D with QUAD4) to local coordinates and store results in the same place
        transUV.left_multiply(transTri);

        if (debug) {
            std::cout << "transUV:\n"; transUV.print(std::cout);
            std::cout << "\ntrafo:\n"; transTri.print(std::cout); std::cout << std::endl;
        }

        if (type == TRI3)
        {std::cout << "tri3 element\n";
            dphi_p.resize(3,2); // resizes matrix to 3 rows, 2 columns and zeros entries
            dphi_p(0,0) = -transUV(0,0); // x12 = x1-x2 = 0-x2 = -x2
            dphi_p(1,0) =  transUV(0,1); // x31 = x3-x1 = x3-0 = x3
            dphi_p(2,0) =  transUV(0,0)-transUV(0,1); // x23 = x2-x3
            dphi_p(0,1) = -transUV(1,0); // y12 = 0, stays zero, as node B and A lies on local x-axis and therefore y=0 for both
            dphi_p(1,1) =  transUV(1,1); // y31 = y3-y1 = y3-0 = y3
            dphi_p(2,1) =  transUV(1,0)-transUV(1,1); // y23 = y2-y3 = 0-y3 = -y3

            /*****************************************
             * BEGIN OF PLANE COMPUTATION            *
             *****************************************/

            DenseMatrix<Real> B_m(3,6);
            B_m(0,0) =  dphi_p(2,1); //  y23
            B_m(0,2) =  dphi_p(1,1); //  y31
            B_m(0,4) =  dphi_p(0,1); //  y12
            B_m(1,1) = -dphi_p(2,0); // -x23
            B_m(1,3) = -dphi_p(1,0); // -x31
            B_m(1,5) = -dphi_p(0,0); // -x12
            B_m(2,0) = -dphi_p(2,0); // -x23
            B_m(2,1) =  dphi_p(2,1); //  y23
            B_m(2,2) = -dphi_p(1,0); // -x31
            B_m(2,3) =  dphi_p(1,1); //  y31
            B_m(2,4) = -dphi_p(0,0); // -x12
            B_m(2,5) =  dphi_p(0,1); //  y12
            B_m *= 1.0/(2.0*area);

            // Ke_m = t*A* B^T * Dm * B
            Ke_m = Dm; // Ke_m = 3x3
            Ke_m.right_multiply(B_m); // Ke_m = 3x6
            Ke_m.left_multiply_transpose(B_m); // Ke_m = 6x6
            Ke_m *= thickness*area; // considered thickness and area is constant all over the element

            /*************************************************
             * END OF PLANE COMPUTATION                      *
             *************************************************/

            /*****************************************
             * BEGIN OF PLATE COMPUTATION            *
             *****************************************/

            qps.resize(3);
            for (unsigned int i = 0; i < qps.size(); i++)
                qps[i].resize(2);
            qps[0][0] = 1.0/6.0; qps[0][1] = 1.0/6.0;
            qps[1][0] = 2.0/3.0; qps[1][1] = 1.0/6.0;
            qps[2][0] = 1.0/6.0; qps[2][1] = 2.0/3.0;

            // side-lengths squared:
            sidelen.resize(3);
            sidelen[0] = pow(dphi_p(0,0), 2.0) + pow(dphi_p(0,1), 2.0); // side AB, x12^2 + y12^2 (=0) -> x12^2 = x2^2
            sidelen[1] = pow(dphi_p(1,0), 2.0) + pow(dphi_p(1,1), 2.0); // side AC, x31^2 + y31^2
            sidelen[2] = pow(dphi_p(2,0), 2.0) + pow(dphi_p(2,1), 2.0); // side BC, x23^2 + y23^2

            Hcoeffs.resize(1,3);
            for (int i = 0; i < 3; i++)
                Hcoeffs(0,i) = sidelen[i];

            // resize the current element matrix and vector to an appropriate size
            Ke_p.resize(9, 9);
            for (unsigned int i = 0; i < qps.size(); i++)
            {
                if (debug)
                    std::cout << "quadrature point (" << qps[i][0] << "," << qps[i][1] << ")\n";

                if (debug) {
                    std::cout << "Hcoeffs:\n";
                    Hcoeffs.print(std::cout);
                    std::cout << std::endl;
                }

                eval_B(Hcoeffs, qps[i][0], qps[i][1], dphi_p, B);

                DenseMatrix<Real> Y(3,3);
                Y(0,0) = pow(dphi_p(2,1),2.0);
                Y(0,1) = pow(dphi_p(1,1),2.0);
                Y(0,2) = dphi_p(2,1)*dphi_p(1,1);
                Y(1,0) = pow(dphi_p(2,0),2.0);
                Y(1,1) = pow(dphi_p(1,0),2.0);
                Y(1,2) = dphi_p(1,0)*dphi_p(2,0);
                Y(2,0) = -2.0*dphi_p(2,0)*dphi_p(2,1);
                Y(2,1) = -2.0*dphi_p(1,0)*dphi_p(1,0);
                Y(2,2) = -dphi_p(2,0)*dphi_p(1,1)-dphi_p(1,0)*dphi_p(2,1);
                Y *= 1.0/(4.0*pow(area,2.0));

                if (debug) {
                    std::cout << "B:\n";
                    B.print(std::cout);
                    std::cout << std::endl;
                }

                DenseMatrix<Real> temp;
                temp = Dp; // temp = 3x3
                temp.right_multiply(Y); // temp = 3x3
                temp.right_multiply(B); // temp = 9x3
                temp.left_multiply_transpose(Y); // temp = 9x3
                temp.left_multiply_transpose(B); // temp = 9x9

                temp *= 1.0/6.0; // gauss-weight

                Ke_p += temp;
            }

            Ke_p *= 2.0*area;

            /*****************************************
             * END OF PLATE COMPUTATION            *
             *****************************************/

            // copy values from submatrices into overall element matrix:
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    // submatrix K_ij [6x6]
                    Ke(6*i,  6*j)       = Ke_m(2*i,  2*j);   // uu
                    Ke(6*i,  6*j+1)     = Ke_m(2*i,  2*j+1); // uv
                    Ke(6*i+1,6*j)       = Ke_m(2*i+1,2*j);   // vu
                    Ke(6*i+1,6*j+1)     = Ke_m(2*i+1,2*j+1); // vv
                    Ke(2+6*i,  2+6*j)   = Ke_p(3*i,  3*j);   // ww
                    Ke(2+6*i,  2+6*j+1) = Ke_p(3*i,  3*j+1); // wx
                    Ke(2+6*i,  2+6*j+2) = Ke_p(3*i,  3*j+2); // wy
                    Ke(2+6*i+1,2+6*j)   = Ke_p(3*i+1,3*j);   // xw
                    Ke(2+6*i+1,2+6*j+1) = Ke_p(3*i+1,3*j+1); // xx
                    Ke(2+6*i+1,2+6*j+2) = Ke_p(3*i+1,3*j+2); // xy
                    Ke(2+6*i+2,2+6*j)   = Ke_p(3*i+2,3*j);   // yw
                    Ke(2+6*i+2,2+6*j+1) = Ke_p(3*i+2,3*j+1); // yx
                    Ke(2+6*i+2,2+6*j+2) = Ke_p(3*i+2,3*j+2); // yy
                }
            }

            Real max_value;
            for (int zi = 0; zi < 3; zi++)
            {
                for (int zj = 0; zj < 3; zj++)
                {
                    // search for max value in uv-matrix
                    max_value = Ke_m(2*zi,2*zj); // begin with uu value
                    max_value = std::max(max_value, Ke_m(2*zi+1,2*zj+1)); // test for vv
                    // search for max value in w-matrix
                    max_value = std::max(max_value, Ke_p(3*zi,  3*zj)); // test for ww
                    max_value = std::max(max_value, Ke_p(3*zi+1,3*zj+1)); // test for t_x t_x
                    max_value = std::max(max_value, Ke_p(3*zi+2,3*zj+2)); // test for t_y t_y
                    // take max from both and divide it by 1000
                    max_value /= 1000.0;
                    // set it at corresponding place
                    Ke(5+6*zi,5+6*zj) = max_value;
                }
            }

            // transform Ke from local back to global with transformation matrix T:
            DenseMatrix<Real> KeSub(6,6);
            DenseMatrix<Real> KeNew(18,18);
            DenseMatrix<Real> TSub(6,6);
            for (int k = 0; k < 2; k++)
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        TSub(3*k+i,3*k+j) = transTri(i,j);

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 6; k++)
                        for (int l = 0; l < 6; l++)
                            KeSub(k,l) = Ke(i*6+k,j*6+l);

                    KeSub.right_multiply(TSub);
                    KeSub.left_multiply_transpose(TSub);

                    for (int k = 0; k < 6; k++)
                        for (int l = 0; l < 6; l++)
                            KeNew(i*6+k,j*6+l) = KeSub(k,l);
                }
            }

            for (int alpha = 0; alpha < 6; alpha++)
                for (int beta = 0; beta < 6; beta++)
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                            Ke(3*alpha+i,3*beta+j) = KeNew(6*i+alpha,6*j+beta);
        }
        else if (type == QUAD4)
        {//std::cout << "quad4 element\n";
            // first partial derivatives x_ij, y_ij
            dphi_p.resize(6,2); // resizes matrix to 6 rows, 2 columns and zeros entries
            dphi_p(0,0) = transUV(0,0)-transUV(0,1); // x12 = x1-x2
            dphi_p(1,0) = transUV(0,1)-transUV(0,2); // x23 = x2-x3
            dphi_p(2,0) = transUV(0,2)-transUV(0,3); // x34 = x3-x4
            dphi_p(3,0) = transUV(0,3)-transUV(0,0); // x41 = x4-x1

            dphi_p(0,1) = transUV(1,0)-transUV(1,1); // y12 = y1-y2
            dphi_p(1,1) = transUV(1,1)-transUV(1,2); // y23 = y2-y3
            dphi_p(2,1) = transUV(1,2)-transUV(1,3); // y34 = y3-y4
            dphi_p(3,1) = transUV(1,3)-transUV(1,0); // y41 = y4-y1

            dphi_p(4,0) = transUV(0,2)-transUV(0,0); // x31 = x3-x1
            dphi_p(4,1) = transUV(1,2)-transUV(1,0); // y31 = y3-y1
            dphi_p(5,0) = transUV(0,3)-transUV(0,1); // x42 = x4-x2
            dphi_p(5,1) = transUV(1,3)-transUV(1,1); // y42 = y4-y2

            if (debug) {
                std::cout << "dphi_p:\n";
                dphi_p.print(std::cout);
                std::cout << "\n";
            }

            // quadrature points definitions:
            Real root = sqrt(1.0/3.0);

            // side-lengths squared:
            sidelen.resize(4);
            sidelen[0] = pow(dphi_p(0,0), 2.0) + pow(dphi_p(0,1), 2.0); // side AB, x12^2 + y12^2
            sidelen[1] = pow(dphi_p(1,0), 2.0) + pow(dphi_p(1,1), 2.0); // side BC, x23^2 + y23^2
            sidelen[2] = pow(dphi_p(2,0), 2.0) + pow(dphi_p(2,1), 2.0); // side CD, x34^2 + y34^2
            sidelen[3] = pow(dphi_p(3,0), 2.0) + pow(dphi_p(3,1), 2.0); // side DA, x41^2 + y41^2

            area = 0.0;
            for (int i = 0; i < 4; i++) // Gauss's area formula
                area += transUV(0,i)*transUV(1,(i+1)%4) - transUV(0,(i+1)%4)*transUV(1,i); // x_i*y_{i+1} - x_{i+1}*y_i = det((x_i,x_i+1),(y_i,y_i+1))
            area *= 0.5;

            if (debug)
                std::cout << "lij^2 = (" << sidelen[0] << ", " << sidelen[1] << ", " << sidelen[2] << ", " << sidelen[3] << ")\n";

            /*****************************************
             * BEGIN OF PLANE COMPUTATION            *
             *****************************************/
            DenseMatrix<Real> B_m, G(4,8);
            DenseMatrix<Real> J(2,2), Jinv(2,2);

            // we iterate over the 2x2 Gauss quadrature points (+- sqrt(1/3)) with weight 1
            Ke_m.resize(8,8);
            for (int ii = 0; ii < 2; ii++)
            {
                Real r = pow(-1.0, ii) * root; // +/- root
                for (int jj = 0; jj < 2; jj++)
                {
                    Real s = pow(-1.0, jj) * root; // +/- root

                    DenseVector<Real> shapeQ4(4);
                    shapeQ4(0) = 0.25*(1-r)*(1-s);
                    shapeQ4(1) = 0.25*(1+r)*(1-s);
                    shapeQ4(2) = 0.25*(1+r)*(1+s);
                    shapeQ4(3) = 0.25*(1-r)*(1+s);

                    DenseVector<Real> dhdr(4), dhds(4);
                    dhdr(0) = -0.25*(1-s);
                    dhdr(1) =  0.25*(1-s);
                    dhdr(2) =  0.25*(1+s);
                    dhdr(3) = -0.25*(1+s);

                    dhds(0) = -0.25*(1-r);
                    dhds(1) = -0.25*(1+r);
                    dhds(2) =  0.25*(1+r);
                    dhds(3) =  0.25*(1-r);

                    J.resize(2,2);
                    for (int i=0; i < 4; i++) {
                        J(0,0) += dhdr(i)*transUV(0,i);
                        J(0,1) += dhdr(i)*transUV(1,i);
                        J(1,0) += dhds(i)*transUV(0,i);
                        J(1,1) += dhds(i)*transUV(1,i);
                    }

                    Real detjacob = J.det();

                    B_m.resize(3,4);
                    B_m(0,0) =  J(1,1); B_m(0,1) = -J(0,1);
                    B_m(1,2) = -J(1,0); B_m(1,3) =  J(0,0);
                    B_m(2,0) = -J(1,0); B_m(2,1) =  J(0,0); B_m(2,2) = J(1,1); B_m(2,3) = -J(0,1);
                    B_m *= 1.0/detjacob;

                    for (int i = 0; i < 4; i++) {
                        G(0,2*i) = dhdr(i);
                        G(1,2*i) = dhds(i);
                        G(2,1+2*i) = dhdr(i);
                        G(3,1+2*i) = dhds(i);
                    }

                    B_m.right_multiply(G);

                    // Ke_m = t * B^T * Dm * B * |J|
                    DenseMatrix<Real> Ke_m_tmp;
                    Ke_m_tmp = Dm; // Ke_m = 3x3
                    Ke_m_tmp.left_multiply_transpose(B_m); // Ke_m = 8x8
                    Ke_m_tmp.right_multiply(B_m); // Ke_m = 3x8
                    Ke_m_tmp *= detjacob * thickness; // considered thickness and area is constant all over the element

                    Ke_m += Ke_m_tmp;
                }
            }

            if (debug) {
                std::cout << "Ke_m:\n";
                Ke_m.print(std::cout);
                std::cout << std::endl;
            }

            /*************************************************
             * END OF PLANE COMPUTATION                      *
             *************************************************/

            /*****************************************
             * BEGIN OF PLATE COMPUTATION            *
             *****************************************/
            Hcoeffs.resize(5,4); // [ a_k, b_k, c_k, d_k, e_k ], k=5,6,7,8
            for (int i = 0; i < 4; i++)
            {
                Hcoeffs(0,i) = -dphi_p(i,0)/sidelen[i]; // a_k
                Hcoeffs(1,i) = 0.75 * dphi_p(i,0) * dphi_p(i,1) / sidelen[i]; // b_k
                Hcoeffs(2,i) = (0.25 * pow(dphi_p(i,0), 2.0) - 0.5 * pow(dphi_p(i,1), 2.0))/sidelen[i]; // c_k
                Hcoeffs(3,i) = -dphi_p(i,1)/sidelen[i]; // d_k
                Hcoeffs(4,i) = (0.25 * pow(dphi_p(i,1), 2.0) - 0.5 * pow(dphi_p(i,0), 2.0))/sidelen[i]; // e_k
            }

            if (debug) {
                std::cout << "Hcoeffs:\n";
                Hcoeffs.print(std::cout);
                std::cout << std::endl;
            }

            // resize the current element matrix and vector to an appropriate size
            Ke_p.resize(12, 12);
            for (int ii = 0; ii < 2; ii++)
            {
                Real r = pow(-1.0, ii) * root; // +/- sqrt(1/3)
                for (int jj = 0; jj < 2; jj++)
                {
                    Real s = pow(-1.0, jj) * root; // +/- sqrt(1/3)

                    if (debug) std::cout << "(r,s) = " << r << ", " << s << "\n";

                    J(0,0) = (dphi_p(0,0)+dphi_p(2,0))*s - dphi_p(0,0) + dphi_p(2,0);
                    J(0,1) = (dphi_p(0,1)+dphi_p(2,1))*s - dphi_p(0,1) + dphi_p(2,1);
                    J(1,0) = (dphi_p(0,0)+dphi_p(2,0))*r - dphi_p(1,0) + dphi_p(3,0);
                    J(1,1) = (dphi_p(0,1)+dphi_p(2,1))*r - dphi_p(1,1) + dphi_p(3,1);
                    J *= 0.25;

                    if (debug) {
                        std::cout << "J:\n";
                        J.print(std::cout);
                        std::cout << std::endl;
                    }

                    Real det = J.det();
                    if (debug) std::cout << "J.det = " << det << "\n";

                    Jinv(0,0) =  J(1,1);
                    Jinv(0,1) = -J(0,1);
                    Jinv(1,0) = -J(1,0);
                    Jinv(1,1) =  J(0,0);
                    Jinv *= 1.0/det;

                    if (debug) {
                        std::cout << "Jinv:\n";
                        Jinv.print(std::cout);
                        std::cout << std::endl;
                    }

                    eval_B_quad(Hcoeffs, r, s, Jinv, B);

                    if (debug) {
                        std::cout << "B:\n";
                        B.print(std::cout);
                        std::cout << std::endl;
                    }

                    DenseMatrix<Real> temp;
                    temp = Dp; // temp = 3x3
                    temp.left_multiply_transpose(B); // temp = 12x3
                    temp.right_multiply(B); // temp = 12x12
                    temp *= det;

                    Ke_p += temp;

                    if (debug) {
                        std::cout << "temp:\n";
                        temp.print(std::cout);
                        std::cout << std::endl;
                    }
                }
            }

            if (debug) {
                std::cout << "Ke_p:\n";
                Ke_p.print(std::cout);
                std::cout << std::endl;
            }
            /*****************************************
             * END OF PLATE COMPUTATION            *
             *****************************************/

            // copy values from submatrices into overall element matrix:
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    // submatrix K_ij [6x6]
                    Ke(6*i,  6*j)       = Ke_m(2*i,  2*j);   // uu
                    Ke(6*i,  6*j+1)     = Ke_m(2*i,  2*j+1); // uv
                    Ke(6*i+1,6*j)       = Ke_m(2*i+1,2*j);   // vu
                    Ke(6*i+1,6*j+1)     = Ke_m(2*i+1,2*j+1); // vv
                    Ke(2+6*i,  2+6*j)   = Ke_p(3*i,  3*j);   // ww
                    Ke(2+6*i,  2+6*j+1) = Ke_p(3*i,  3*j+1); // wx
                    Ke(2+6*i,  2+6*j+2) = Ke_p(3*i,  3*j+2); // wy
                    Ke(2+6*i+1,2+6*j)   = Ke_p(3*i+1,3*j);   // xw
                    Ke(2+6*i+1,2+6*j+1) = Ke_p(3*i+1,3*j+1); // xx
                    Ke(2+6*i+1,2+6*j+2) = Ke_p(3*i+1,3*j+2); // xy
                    Ke(2+6*i+2,2+6*j)   = Ke_p(3*i+2,3*j);   // yw
                    Ke(2+6*i+2,2+6*j+1) = Ke_p(3*i+2,3*j+1); // yx
                    Ke(2+6*i+2,2+6*j+2) = Ke_p(3*i+2,3*j+2); // yy
                }
            }

            Real max_value;
            for (int zi = 0; zi < 4; zi++)
            {
                for (int zj = 0; zj < 4; zj++)
                {
                    // search for max value in uv-matrix
                    max_value = Ke_m(2*zi,2*zj); // begin with uu value
                    max_value = std::max(max_value, Ke_m(2*zi+1,2*zj+1)); // test for vv
                    // search for max value in w-matrix
                    max_value = std::max(max_value, Ke_p(3*zi,  3*zj)); // test for ww
                    max_value = std::max(max_value, Ke_p(3*zi+1,3*zj+1)); // test for t_x t_x
                    max_value = std::max(max_value, Ke_p(3*zi+2,3*zj+2)); // test for t_y t_y
                    // take max from both and divide it by 1000
                    max_value /= 1000.0;
                    // set it at corresponding place
                    Ke(5+6*zi,5+6*zj) = max_value;
                }
            }

            if (debug) {
                std::cout << "Ke PRE:\n";
                Ke.print(std::cout);
                std::cout << "\n";
            }

            // transform Ke from local back to global with transformation matrix T:
            DenseMatrix<Real> KeSub(6,6);
            DenseMatrix<Real> KeNew(24,24);
            DenseMatrix<Real> TSub(6,6);
            for (int k = 0; k < 2; k++) // copy transTri two times into TSub (cf comment beneath)
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        TSub(3*k+i,3*k+j) = transTri(i,j);
            /* TSub: [ux, vx, wx,  0,  0,  0]
             *       [uy, vy, wy,  0,  0,  0]
             *       [uz, vz, wz,  0,  0,  0]
             *       [0 ,  0,  0, ux, vx, wx]
             *       [0 ,  0,  0, uy, vy, wy]
             *       [0 ,  0,  0, uz, vz, wz] */

            if (debug) {
                std::cout << "TSub:\n";
                TSub.print(std::cout);
                std::cout << "\n";
            }

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int k = 0; k < 6; k++) // copy values into KeSub for right format to transformation
                        for (int l = 0; l < 6; l++)
                            KeSub(k,l) = Ke(i*6+k,j*6+l);

                    // the actual transformation step
                    KeSub.right_multiply(TSub);
                    KeSub.left_multiply_transpose(TSub);

                    // copy transformed values into temporal stiffness matrix
                    for (int k = 0; k < 6; k++)
                        for (int l = 0; l < 6; l++)
                            KeNew(i*6+k,j*6+l) = KeSub(k,l);
                }
            }

            // bring stiffness matrix into right format for libmesh equation system handling
            for (int alpha = 0; alpha < 6; alpha++)
                for (int beta = 0; beta < 6; beta++)
                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < 4; j++)
                            Ke(4*alpha+i,4*beta+j) = KeNew(6*i+alpha,6*j+beta);
        }
        else
            std::cout << "unknown element\n";

        if (debug) {
            std::cout << "K_m:\n";
            Ke_m.print(std::cout);
            std::cout << std::endl;
            std::cout << "K_p:\n";
            Ke_p.print(std::cout);
            std::cout << std::endl;
        }

        if (debug) {
            std::cout << "Ke global in libmesh ordering:\n";
            Ke.print(std::cout);
            std::cout << std::endl;
        }

        if (debug)
            std::cout << "area: " << area << "\n";

        unsigned int nsides = elem->n_sides();
        Fe.resize(6*nsides);
        DenseVector<Real> arg;
        for (unsigned int side = 0; side < nsides; side++)
        {
            Node* node = elem->get_node(side);
            dof_id_type id = node->id();
            if (node->processor_id() != global_processor_id())
                continue;

            if (processedNodes.find(id) == processedNodes.end())
            {
                processedNodes.insert(id);

                if (debug)
                    std::cout << "id_u = " << (id*6) << ", id_v = " << (id*6+1) << ", id_w = " << (id*6+2) << "\n";

                arg = forces[id];
                if (debug)
                    std::cout << "force = " << arg(0) << "," << arg(1) << "," << arg(2) << "\n";
                // forces don't need to be transformed since we bring the local stiffness matrix back to global co-sys
                // directly in libmesh-format:
                Fe(side)          = arg(0); // u_i (0,1,..,n-1)
                Fe(side+nsides)   = arg(1); // v_i (n,n+1,...,2n-1)
                Fe(side+nsides*2) = arg(2); // w_i (2n,2n+1,...,3n-1) nodal load
                //Fe(side+nsides*2) = area*arg(2)/(float)nsides; // w_i (2n,2n+1,...,3n-1) area load
                //Fe(side+nsides*3) = area*arg(2)/24.0; // theta_x_i (3n,3n+1,...,4n-1) area load
                //Fe(side+nsides*4) = area*arg(2)/24.0; // theta_y_i (4n,4n+1,...,5n-1) area load
            }
        }

        if (debug) {
            std::cout << "Fe:\n";
            Fe.print(std::cout);
            std::cout << std::endl;
        }

        dof_map.constrain_element_matrix_and_vector(Ke, Fe, dof_indices);

        system.matrix->add_matrix (Ke, dof_indices);
        system.rhs->add_vector    (Fe, dof_indices);

        if (debug)
            std::cout << "END ELEMENT\n";
    }

}

void eval_B_quad(DenseMatrix<Real> &Hcoeffs, Real xi, Real eta, DenseMatrix<Real> &Jinv, DenseMatrix<Real> &out)
{
    out.resize(3,12);

    DenseVector<Real> N_xi(8), N_eta(8);
    N_xi(0) =  0.25*(2.0*xi+eta)*(1.0-eta);
    N_xi(1) =  0.25*(2.0*xi-eta)*(1.0-eta);
    N_xi(2) =  0.25*(2.0*xi+eta)*(1.0+eta);
    N_xi(3) =  0.25*(2.0*xi-eta)*(1.0+eta);
    N_xi(4) = -xi  *(1.0-eta);
    N_xi(5) =  0.5 *(1.0-pow(eta,2.0));
    N_xi(6) = -xi  *(1.0+eta);
    N_xi(7) = -0.5 *(1.0-pow(eta,2.0));

    N_eta(0) =  0.25*(2.0*eta+xi)*(1.0-xi);
    N_eta(1) =  0.25*(2.0*eta-xi)*(1.0+xi);
    N_eta(2) =  0.25*(2.0*eta+xi)*(1.0+xi);
    N_eta(3) =  0.25*(2.0*eta-xi)*(1.0-xi);
    N_eta(4) = -0.5 *(1.0-pow(xi,2.0));
    N_eta(5) = -eta *(1.0+xi);
    N_eta(6) =  0.5 *(1.0-pow(xi,2.0));
    N_eta(7) = -eta *(1.0-xi);

    int  a = 0,  b = 1,  c = 2,  d = 3, e = 4;
    int i5 = 0, i6 = 1, i7 = 2, i8 = 3;

    DenseVector<Real> Hx_xi(12), Hy_xi(12), Hx_eta(12), Hy_eta(12);
    Hx_xi(0) = 1.5 * (   Hcoeffs(a,i5)*N_xi(4) - Hcoeffs(a,i8)*N_xi(7));
    Hx_xi(1) =           Hcoeffs(b,i5)*N_xi(4) + Hcoeffs(b,i8)*N_xi(7);
    Hx_xi(2) = N_xi(0) - Hcoeffs(c,i5)*N_xi(4) - Hcoeffs(c,i8)*N_xi(7);
    Hx_xi(3) = 1.5 * (   Hcoeffs(a,i6)*N_xi(5) - Hcoeffs(a,i5)*N_xi(4));
    Hx_xi(4) =           Hcoeffs(b,i6)*N_xi(5) + Hcoeffs(b,i5)*N_xi(4);
    Hx_xi(5) = N_xi(1) - Hcoeffs(c,i6)*N_xi(5) - Hcoeffs(c,i5)*N_xi(4);
    Hx_xi(6) = 1.5 * (   Hcoeffs(a,i7)*N_xi(6) - Hcoeffs(a,i6)*N_xi(5));
    Hx_xi(7) =           Hcoeffs(b,i7)*N_xi(6) + Hcoeffs(b,i6)*N_xi(5);
    Hx_xi(8) = N_xi(2) - Hcoeffs(c,i7)*N_xi(6) - Hcoeffs(c,i6)*N_xi(5);
    Hx_xi(9) = 1.5 * (   Hcoeffs(a,i8)*N_xi(7) - Hcoeffs(a,i7)*N_xi(6));
    Hx_xi(10)=           Hcoeffs(b,i8)*N_xi(7) + Hcoeffs(b,i7)*N_xi(6);
    Hx_xi(11)= N_xi(3) - Hcoeffs(c,i8)*N_xi(7) - Hcoeffs(c,i7)*N_xi(6);

    Hy_xi(0) = 1.5 * (   Hcoeffs(d,i5)*N_xi(4) - Hcoeffs(d,i8)*N_xi(7));
    Hy_xi(1) = -N_xi(0) +Hcoeffs(e,i5)*N_xi(4) + Hcoeffs(e,i8)*N_xi(7);
    Hy_xi(2) = -Hx_xi(1);
    Hy_xi(3) = 1.5 * (   Hcoeffs(d,i6)*N_xi(5) - Hcoeffs(d,i5)*N_xi(4));
    Hy_xi(4) = -N_xi(1) +Hcoeffs(e,i6)*N_xi(5) + Hcoeffs(e,i5)*N_xi(4);
    Hy_xi(5) = -Hx_xi(4);
    Hy_xi(6) = 1.5 * (   Hcoeffs(d,i7)*N_xi(6) - Hcoeffs(d,i6)*N_xi(5));
    Hy_xi(7) = -N_xi(2) +Hcoeffs(e,i7)*N_xi(6) + Hcoeffs(e,i6)*N_xi(5);
    Hy_xi(8) = -Hx_xi(7);
    Hy_xi(9) = 1.5 * (   Hcoeffs(d,i8)*N_xi(7) - Hcoeffs(d,i7)*N_xi(6));
    Hy_xi(10)= -N_xi(3) +Hcoeffs(e,i8)*N_xi(7) + Hcoeffs(e,i7)*N_xi(6);
    Hy_xi(11)= -Hx_xi(10);

    Hx_eta(0) = 1.5 * (   Hcoeffs(a,i5)*N_eta(4) - Hcoeffs(a,i8)*N_eta(7));
    Hx_eta(1) =           Hcoeffs(b,i5)*N_eta(4) + Hcoeffs(b,i8)*N_eta(7);
    Hx_eta(2) = N_eta(0) -Hcoeffs(c,i5)*N_eta(4) - Hcoeffs(c,i8)*N_eta(7);
    Hx_eta(3) = 1.5 * (   Hcoeffs(a,i6)*N_eta(5) - Hcoeffs(a,i5)*N_eta(4));
    Hx_eta(4) =           Hcoeffs(b,i6)*N_eta(5) + Hcoeffs(b,i5)*N_eta(4);
    Hx_eta(5) = N_eta(1) -Hcoeffs(c,i6)*N_eta(5) - Hcoeffs(c,i5)*N_eta(4);
    Hx_eta(6) = 1.5 * (   Hcoeffs(a,i7)*N_eta(6) - Hcoeffs(a,i6)*N_eta(5));
    Hx_eta(7) =           Hcoeffs(b,i7)*N_eta(6) + Hcoeffs(b,i6)*N_eta(5);
    Hx_eta(8) = N_eta(2) -Hcoeffs(c,i7)*N_eta(6) - Hcoeffs(c,i6)*N_eta(5);
    Hx_eta(9) = 1.5 * (   Hcoeffs(a,i8)*N_eta(7) - Hcoeffs(a,i7)*N_eta(6));
    Hx_eta(10)=           Hcoeffs(b,i8)*N_eta(7) + Hcoeffs(b,i7)*N_eta(6);
    Hx_eta(11)= N_eta(3) -Hcoeffs(c,i8)*N_eta(7) - Hcoeffs(c,i7)*N_eta(6);

    Hy_eta(0) = 1.5 * (   Hcoeffs(d,i5)*N_eta(4) - Hcoeffs(d,i8)*N_eta(7));
    Hy_eta(1) = -N_eta(0)+Hcoeffs(e,i5)*N_eta(4) + Hcoeffs(e,i8)*N_eta(7);
    Hy_eta(2) = -Hx_eta(1);
    Hy_eta(3) = 1.5 * (   Hcoeffs(d,i6)*N_eta(5) - Hcoeffs(d,i5)*N_eta(4));
    Hy_eta(4) = -N_eta(1)+Hcoeffs(e,i6)*N_eta(5) + Hcoeffs(e,i5)*N_eta(4);
    Hy_eta(5) = -Hx_eta(4);
    Hy_eta(6) = 1.5 * (   Hcoeffs(d,i7)*N_eta(6) - Hcoeffs(d,i6)*N_eta(5));
    Hy_eta(7) = -N_eta(2)+Hcoeffs(e,i7)*N_eta(6) + Hcoeffs(e,i6)*N_eta(5);
    Hy_eta(8) = -Hx_eta(7);
    Hy_eta(9) = 1.5 * (   Hcoeffs(d,i8)*N_eta(7) - Hcoeffs(d,i7)*N_eta(6));
    Hy_eta(10)= -N_eta(3)+Hcoeffs(e,i8)*N_eta(7) + Hcoeffs(e,i7)*N_eta(6);
    Hy_eta(11)= -Hx_eta(10);

    for (int i = 0; i < 12; i++)
    {
        out(0,i) = Jinv(0,0)*Hx_xi(i) + Jinv(0,1)*Hx_eta(i);
        out(1,i) = Jinv(1,0)*Hy_xi(i) + Jinv(1,1)*Hy_eta(i);
        out(2,i) = Jinv(0,0)*Hy_xi(i) + Jinv(0,1)*Hy_eta(i) + Jinv(1,0)*Hx_xi(i) + Jinv(1,1)*Hx_eta(i);
    }
}

void eval_B(DenseMatrix<Real>& C, Real L1, Real L2, DenseMatrix<Real>& dphi_p, DenseMatrix<Real> &out)
{
    out.resize(3,9);

    Real mu1 = (C(0,0)-C(0,1))/C(0,2);
    Real mu2 = (C(0,2)-C(0,0))/C(0,1);
    Real mu3 = (C(0,1)-C(0,2))/C(0,0);

    Real L3 = 1-L1-L2;
    Real f13mu1 = 1+3*mu1;
    Real f13mu2 = 1+3*mu2;
    Real f13mu3 = 1+3*mu3;
    Real f1m3mu3 = 1-3*mu3;
    Real fm13mu2 = -1+3*mu2;
    Real fm1m3mu3 = -1-3*mu3;
    Real f1mmu1 = 1-mu1;
    Real f1mmu2 = 1-mu2;
    Real f1mmu3 = 1-mu3;

    Real a = 3*f1mmu3*L1-f13mu3*L2+f13mu3*L3;
    Real b = 3*f1mmu2*L3-f13mu2*L1+f13mu2*L2;
    Real c = 3*f1mmu1*L2-f13mu1*L3+f13mu1*L1;

    out(0,0) = 6 + L2*(-4-2*a) + 4*f1m3mu3*(L2*L3-L1*L2) - 12*L1 + 2*L2*b + 8*(L2*L3-L1*L2);

    out(0,1) = -dphi_p(1,1)*(-2+6*L1+4*L2-L2*b-4*L2*L3+4*L1*L2)
               -dphi_p(0,1)*(2*L2-L2*a+L2*L3*2*f1m3mu3-L1*L2*2*f1m3mu3);

    out(0,2) =  dphi_p(1,0)*(-2+6*L1+4*L2-L2*b-4*L2*L3+4*L1*L2)
               +dphi_p(0,0)*(2*L2-L2*a+L2*L3*2*f1m3mu3-L1*L2*2*f1m3mu3);

    out(0,3) = -2*L2*c + 4*f13mu1*(L2*L3-L1*L2) - 4*L2 + 2*L2*a + 4*f1m3mu3*(-L2*L3+L1*L2);

    out(0,4) = -dphi_p(0,1)*(2*L2-L2*a+L2*L3*2*f1m3mu3-L1*L2*2*f1m3mu3)
               -dphi_p(2,1)*(-L2*c+L2*L3*2*f13mu1-L1*L2*2*f13mu1);

    out(0,5) = dphi_p(0,0)*(2*L2-L2*a+L2*L3*2*f1m3mu3-L1*L2*2*f1m3mu3)
              +dphi_p(2,0)*(-L2*c+L2*L3*2*f13mu1-L1*L2*2*f13mu1);

    out(0,6) = -6 + 12*L1 + 8*L2 - 2*L2*b + 8*(L1*L2-L2*L3) + 2*L2*c + 4*f13mu1*(L1*L2-L2*L3);

    out(0,7) = -dphi_p(2,1)*(-L2*c+L2*L3*2*f13mu1-L1*L2*2*f13mu1)
               -dphi_p(1,1)*(-4+6*L1+4*L2-L2*b-4*L2*L3+4*L1*L2);

    out(0,8) = dphi_p(2,0)*(-L2*c+L2*L3*2*f13mu1-L1*L2*2*f13mu1)
              +dphi_p(1,0)*(-4+6*L1+4*L2-L2*b-4*L2*L3+4*L1*L2);

    out(1,0) = -2*L1*a + 2*L1*L3*2*fm1m3mu3 - 2*L1*L2*2*fm1m3mu3 - 4*L1+2*L1*b - 2*L1*L3*2*fm13mu2 + 2*L1*L2*2*fm13mu2;

    out(1,1) = -dphi_p(1,1)*(2*L1-1*L1*b+1*L1*L3*2*fm13mu2-1*L1*L2*2*fm13mu2)
               -dphi_p(0,1)*(-1*L1*a+1*L1*L3*2*fm1m3mu3-1*L1*L2*2*fm1m3mu3);

    out(1,2) = dphi_p(1,0)*(2*L1-1*L1*b+1*L1*L3*2*fm13mu2-1*L1*L2*2*fm13mu2)
              +dphi_p(0,0)*(-1*L1*a+1*L1*L3*2*fm1m3mu3-1*L1*L2*2*fm1m3mu3);

    out(1,3) = 6 - 12*L2 - 4*L1-2*L1*c + 8*L3*L1 - 8*L1*L2 + 2*L1*a - 2*L1*L3*2*fm1m3mu3 + 2*L1*L2*2*fm1m3mu3;

    out(1,4) = -dphi_p(0,1)*(-1*L1*a+1*L1*L3*2*fm1m3mu3-1*L1*L2*2*fm1m3mu3)
               -dphi_p(2,1)*(-6*L2+2-2*L1-1*L1*c+4*L3*L1-4*L1*L2);

    out(1,5) = dphi_p(0,0)*(-1*L1*a+1*L1*L3*2*fm1m3mu3-1*L1*L2*2*fm1m3mu3)
              +dphi_p(2,0)*(-6*L2+2-2*L1-1*L1*c+4*L3*L1-4*L1*L2);

    out(1,6) = -6 + 8*L1 - 2*L1*b + 2*L1*L3*2*fm13mu2 - 2*L1*L2*2*fm13mu2 + 12*L2 + 2*L1*c - 8*L3*L1 +  8*L1*L2;

    out(1,7) = -dphi_p(2,1)*(-6*L2+4-2*L1-1*L1*c+4*L3*L1-4*L1*L2)
               -dphi_p(1,1)*(2*L1-1*L1*b+1*L1*L3*2*fm13mu2-1*L1*L2*2*fm13mu2);

    out(1,8) = dphi_p(2,0)*(-6*L2+4-2*L1-1*L1*c+4*L3*L1-4*L1*L2)
              +dphi_p(1,0)*(2*L1-1*L1*b+1*L1*L3*2*fm13mu2-1*L1*L2*2*fm13mu2);

    out(2,0) = 2 - 4*L1 + L3*a - L2*a + L2*L3*2*fm1m3mu3 - L1*a - L1*L2*2*fm1m3mu3 + L1*L3*2*f1m3mu3 - L1*L2*2*f1m3mu3
                 - 4*L2 - L3*b + L2*b - L2*L3*2*fm13mu2  + L1*b + L1*L2*2*fm13mu2  + 4*L3*L1         - 4*L1*L2;

    out(2,1) = -dphi_p(1,1)*(-1 + 4*L1 + 2*L2 + 0.5*L3*b - 0.5*L2*b + 0.5*L2*L3*2*fm13mu2
                             - 0.5*L1*b - 0.5*L1*L2*2*fm13mu2 - 2*L3*L1 + 2*L1*L2)
               -dphi_p(0,1)*(2*L1 + 0.5*L3*a - 0.5*L2*a + 0.5*L2*L3*2*fm1m3mu3 - 0.5*L1*a
                             - 0.5*L1*L2*2*fm1m3mu3 + 0.5*L1*L3*2*f1m3mu3 - 0.5*L1*L2*2*f1m3mu3);

    out(2,2) =  dphi_p(1,0)*(-1 + 4*L1 + 2*L2 + 0.5*L3*b - 0.5*L2*b + 0.5*L2*L3*2*fm13mu2
                             - 0.5*L1*b - 0.5*L1*L2*2*fm13mu2 - 2*L3*L1 + 2*L1*L2)
               +dphi_p(0,0)*(2*L1 + 0.5*L3*a - 0.5*L2*a + 0.5*L2*L3*2*fm1m3mu3 - 0.5*L1*a
                             - 0.5*L1*L2*2*fm1m3mu3 + 0.5*L1*L3*2*f1m3mu3 - 0.5*L1*L2*2*f1m3mu3);

    out(2,3) = 2 - 4*L2 + L3*c - L2*c + 4*L2*L3 - L1*c - 4*L1*L2 + L1*L3*2*f13mu1 - L1*L2*2*f13mu1
                 - 4*L1 - L3*a + L2*a + L1*a - L2*L3*2*fm1m3mu3 + L1*L2*2*fm1m3mu3 - L1*L3*2*f1m3mu3
                 + L1*L2*2*f1m3mu3;

    out(2,4) = -dphi_p(0,1)*(2*L1
                   +0.5*L3*a
                   -0.5*L2*a
                   +0.5*L2*L3*2*fm1m3mu3
                   -0.5*L1*a
                   -0.5*L1*L2*2*fm1m3mu3
                   +0.5*L1*L3*2*f1m3mu3
                   -0.5*L1*L2*2*f1m3mu3
                   -1)
             -dphi_p(2,1)*(-2*L2
                   +0.5*L3*c
                   -0.5*L2*c
                   +2*L2*L3
                   -0.5*L1*c
                   -2*L1*L2
                   +0.5*L1*L3*2*f13mu1
                   -0.5*L1*L2*2*f13mu1
                   );

    out(2,5) = dphi_p(0,0)*(2*L1
                  +0.5*L3*a
                  -0.5*L2*a
                  +0.5*L2*L3*2*fm1m3mu3
                  -0.5*L1*a
                  -0.5*L1*L2*2*fm1m3mu3
                  +0.5*L1*L3*2*f1m3mu3
                  -0.5*L1*L2*2*f1m3mu3
                  -1)
             +dphi_p(2,0)*(-2*L2
                   +0.5*L3*c
                   -0.5*L2*c
                   +2*L2*L3
                   -0.5*L1*c
                   -2*L1*L2
                   +0.5*L1*L3*2*f13mu1
                   -0.5*L1*L2*2*f13mu1
                   );

    out(2,6) = -4
             +8*L1
             +8*L2
             +L3*b
             -L2*b
             +L2*L3*2*fm13mu2
             -L1*b
             -L1*L2*2*fm13mu2
             -4*L3*L1
             +8*L1*L2
             -L3*c
             +L2*c
             -4*L2*L3
             +L1*c
             -L1*L3*2*f13mu1
             +L1*L2*2*f13mu1;

    out(2,7) = -dphi_p(2,1)*(-2*L2
                   +0.5*L3*c
                   -0.5*L2*c
                   +2*L2*L3
                   -0.5*L1*c
                   -2*L1*L2
                   +0.5*L1*L3*2*f13mu1
                   -0.5*L1*L2*2*f13mu1
                   +1
                   )
             -dphi_p(1,1)*(-2
                   +4*L1
                   +2*L2
                   +0.5*L3*b
                   -0.5*L2*b
                   +0.5*L2*L3*2*fm13mu2
                   -0.5*L1*b
                   -0.5*L1*L2*2*fm13mu2
                   -2*L3*L1
                   +2*L1*L2
                   );

    out(2,8) = dphi_p(2,0)*(-2*L2
                  +0.5*L3*c
                  -0.5*L2*c
                  +2*L2*L3
                  -0.5*L1*c
                  -2*L1*L2
                  +0.5*L1*L3*2*f13mu1
                  -0.5*L1*L2*2*f13mu1
                  +1
                  )
            +dphi_p(1,0)*(-2
                  +4*L1
                  +2*L2
                  +0.5*L3*b
                  -0.5*L2*b
                  +0.5*L2*L3*2*fm13mu2
                  -0.5*L1*b
                  -0.5*L1*L2*2*fm13mu2
                  -2*L3*L1
                  +2*L1*L2
                 );
    for (int i = 0; i < 9; i++)
        out(2,i) *= 2.0;
}
