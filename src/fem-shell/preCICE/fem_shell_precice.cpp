#include "fem-shell.h"
#include "precice/SolverInterface.hpp"

using namespace precice;
using namespace precice::constants;

// Begin the main program.
int main (int argc, char** argv)
{
    std::cout << "Starting Structure Solver..." << std::endl;

    // read command line arguments and initialize global variables
    read_parameters(argc, argv);

    // Initialize libMesh and any dependent libaries
    LibMeshInit init (argc, argv);

    // Initialize the mesh

    // Skip this program if libMesh was compiled as 1D-only.
    libmesh_example_requires(LIBMESH_DIM >= 2, "2D support");

    // Create a 2D mesh distributed across the default MPI communicator.
    Mesh mesh(init.comm(), 2);
    mesh.allow_renumbering(false);
    mesh.read(in_filename);

    if (isOutfileSet)
    {
        exo_io = new ExodusII_IO(mesh);
        exo_io->append(true);
    }

    // Print information about the mesh to the screen.
    mesh.print_info();

    /******************************
     *   preCICE Initialization   *
     ******************************/
    std::string solverName = "STRUCTURE";

    SolverInterface interface(solverName, global_processor_id(), global_n_processors());
    interface.configure(config_filename);
    std::cout << "preCICE configured..." << std::endl;

    // init data
    int n_nodes = 0;
    BoundaryInfo info = mesh.get_boundary_info();
    info.build_node_list_from_side_list();
    std::vector<const Node*> preCICEnodes;

    MeshBase::const_node_iterator           no = mesh.local_nodes_begin();
    const MeshBase::const_node_iterator end_no = mesh.local_nodes_end();
    for (; no != end_no; ++no)
    {
        const Node *nd = *no;
        if (info.has_boundary_id(nd,2) || info.has_boundary_id(nd,20) || info.has_boundary_id(nd,21))
            preCICEnodes.push_back(nd);
    }
    n_nodes = preCICEnodes.size();
    dimensions = interface.getDimensions();
    std::cout << "dims = " << dimensions << ", n_nodes = " << n_nodes << "\n";
    double *displ;
    displ  = new double[dimensions*n_nodes];  // Second dimension (only one cell deep) stored right after the first dimension: see SolverInterfaceImpl::setMeshVertices
    forces = new double[dimensions*n_nodes];
    double *grid;
    grid = new double[dimensions*n_nodes];

    //precice stuff
    int meshID  = interface.getMeshID("Structure_Nodes");
    int displID = interface.getDataID("Displacements", meshID);
    int forceID = interface.getDataID("Stresses", meshID);
    int *vertexIDs;
    vertexIDs = new int[n_nodes];

    std::vector<const Node*>::iterator iter = preCICEnodes.begin();
    ignoredAxis = 0; // we do not ignore any axis by default
    if (dimensions == 2)
    {
        const Node *firstNd = (*iter);
        Real xMin = (*firstNd)(0);
        Real xMax = xMin;
        Real yMin = (*firstNd)(1);
        Real yMax = yMin;
        Real zMin = (*firstNd)(2);
        Real zMax = zMin;
        iter++;
        for (; iter != preCICEnodes.end(); ++iter)
        {
            const Node *nd = *iter;
            Real cur;
            cur = (*nd)(0);
            if (cur < xMin)
                xMin = cur;
            if (cur > xMax)
                xMax = cur;
            cur = (*nd)(1);
            if (cur < yMin)
                yMin = cur;
            if (cur > yMax)
                yMax = cur;
            cur = (*nd)(2);
            if (cur < zMin)
                zMin = cur;
            if (cur > zMax)
                zMax = cur;
        }
        if (xMin != xMax)
            ignoredAxis += 1;
        if (yMin != yMax)
            ignoredAxis += 2;
        if (zMin != zMax)
            ignoredAxis += 4;
        // ignoredAxis = 0 -> x,y,z ignored, --> invalid (0D)
        //             = 1 -> x used, y,z ignored, --> invalid (1D)
        //             = 2 -> y used, x,z ignored, --> invalid (1D)
        //             = 3 -> x,y used, z ignored, --> valid (xy-2D)
        //             = 4 -> z used, x,y ignored, --> invalid (1D)
        //             = 5 -> x,z used, y ignored, --> valid (xz-2D)
        //             = 6 -> y,z used, x ignored, --> valid (yz-2D)
        //             = 7 -> x,y,z used           --> invalid (3D)

        if (ignoredAxis != 3 && ignoredAxis != 5 && ignoredAxis != 6)
            libmesh_error_msg("Error: preCICE expects 2D mesh, but mesh file does not provide this requirement.");
    }
    iter = preCICEnodes.begin();
    for (int i = 0 ; iter != preCICEnodes.end(); ++iter,++i)
    {
        const Node *nd = *iter;
        for (int dims = 0; dims < dimensions; dims++)
        {
            displ[i*dimensions+dims]  = 0.0;
            forces[i*dimensions+dims] = 0.0;
        }
        if (dimensions == 3)
        {
            grid[i*dimensions]   = (*nd)(0);
            grid[i*dimensions+1] = (*nd)(1);
            grid[i*dimensions+2] = (*nd)(2);
        }
        else
        {
            if (ignoredAxis == 3)
            {
                grid[i*dimensions]   = (*nd)(0);
                grid[i*dimensions+1] = (*nd)(1);
            }
            else if (ignoredAxis == 5)
            {
                grid[i*dimensions]   = (*nd)(0);
                grid[i*dimensions+1] = (*nd)(2);
            }
            else
            {
                grid[i*dimensions]   = (*nd)(1);
                grid[i*dimensions+1] = (*nd)(2);
            }
        }
    }
    int t = 0;
    interface.setMeshVertices(meshID, n_nodes, grid, vertexIDs);

    iter = preCICEnodes.begin();
    for (int i = 0 ; iter != preCICEnodes.end(); ++iter,++i)
    {
        std::pair<dof_id_type, int> pair( (*iter)->id(), vertexIDs[i] );
        id_map.insert(pair);
    }
    std::cout << "Structure: init precice..." << std::endl;
    interface.initialize();

    if ( interface.isActionRequired(actionWriteInitialData()) )
    {
        interface.writeBlockVectorData(displID, n_nodes, vertexIDs, displ);
        interface.fulfilledAction(actionWriteInitialData());
    }

    interface.initializeData();

    if ( interface.isReadDataAvailable() )
    {
        interface.readBlockVectorData(forceID, n_nodes, vertexIDs, forces);
    }
    /****************************
     *   libMesh System setup   *
     ****************************/
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
    // We impose a "simply supported" boundary condition
    // on the nodes with bc_id = 0 and 20
    std::set<boundary_id_type> boundary_ids;
    boundary_ids.insert(0);
    boundary_ids.insert(20);

    // Create a vector storing the variable numbers which the BC applies to
    std::vector<unsigned int> variables;
    variables.push_back(u_var);
    variables.push_back(v_var);
    variables.push_back(w_var);

    // Create a ZeroFunction to initialize dirichlet_bc
    ConstFunction<Number> cf(0.0);
    DirichletBoundary dirichlet_bc(boundary_ids, variables, &cf);

    // We impose a "clamped" boundary condition
    // on the nodes with bc_id = 1 and 21
    boundary_ids.clear();
    boundary_ids.insert(1);
    boundary_ids.insert(21);
    variables.push_back(tx_var);
    variables.push_back(ty_var);
    variables.push_back(tz_var);
    DirichletBoundary dirichlet_bc2(boundary_ids, variables, &cf);

    // We must add the Dirichlet boundary condition _before_ we call equation_systems.init()
    system.get_dof_map().add_dirichlet_boundary(dirichlet_bc);
    system.get_dof_map().add_dirichlet_boundary(dirichlet_bc2);

    initMaterialMatrices();

    // Initialize the data structures for the equation system.
    equation_systems.init();

    // Print information about the system to the screen.
    equation_systems.print_info();

    //const Real tol = equation_systems.parameters.get<Real>("linear solver tolerance");
    //const unsigned int maxits = equation_systems.parameters.get<unsigned int>("linear solver maximum iterations");
    //equation_systems.parameters.set<unsigned int>("linear solver maximum iterations") = maxits*3;
    //equation_systems.parameters.set<Real>        ("linear solver tolerance") = tol/1000.0;
    while ( interface.isCouplingOngoing() )
    {
        // When an implicit coupling scheme is used, checkpointing is required
        if ( interface.isActionRequired(actionWriteIterationCheckpoint()) )
        {
            interface.fulfilledAction(actionWriteIterationCheckpoint());
        }

        // here happens "the magic" of finding new displacements:
        //equation_systems.update();//equation_systems.reinit();
        equation_systems.solve();

        std::vector<Number> sols;
        equation_systems.build_solution_vector(sols);
        if (global_processor_id() > 0)
            sols.reserve(mesh.n_nodes()*6);
        if (global_n_processors() > 1)
            mesh.comm().broadcast(sols);

        /*std::vector<const Node*>::iterator */iter = preCICEnodes.begin();
        for (int i = 0 ; iter != preCICEnodes.end(); ++iter,++i)
        {
            int id = (*iter)->id();
            if (dimensions == 3)
            {
                displ[i*3]   = sols[6*id];
                displ[i*3+1] = sols[6*id+1];
                displ[i*3+2] = sols[6*id+2];
            }
            else
            {
                if (ignoredAxis == 3)
                {
                    displ[i*2]   = sols[6*id];
                    displ[i*2+1] = sols[6*id+1];
                }
                else if (ignoredAxis == 5)
                {
                    displ[i*2]   = sols[6*id];
                    displ[i*2+1] = sols[6*id+2];
                }
                else
                {
                    displ[i*2]   = sols[6*id+1];
                    displ[i*2+1] = sols[6*id+2];
                }
            }
            // add displacements to mesh:
            //Node *nd = *no;
            //(*nd)(0) += sols[6*i];
            //(*nd)(1) += sols[6*i+1];
            //(*nd)(2) += sols[6*i+2];
        }

        interface.writeBlockVectorData(displID, n_nodes, vertexIDs, displ);
        interface.advance(deltaT);
        interface.readBlockVectorData(forceID, n_nodes, vertexIDs, forces);

        if (interface.isActionRequired(actionReadIterationCheckpoint()))
        {
            std::cout << "Iterate" << std::endl;
            interface.fulfilledAction(actionReadIterationCheckpoint());
        }
        else
        {
            std::cout << "Advancing in time, finished timestep: " << t << std::endl;
            t++;

            if (global_processor_id() == 0)
            {
                MeshBase::const_node_iterator           no = mesh.nodes_begin();
                const MeshBase::const_node_iterator end_no = mesh.nodes_end();
                for (; no != end_no; ++no)
                {
                    Node* nd = *no;
                    int id = nd->id();
                    Real displ_x = sols[6*id];
                    Real displ_y = sols[6*id+1];
                    Real displ_z = sols[6*id+2];
                    (*nd)(0) += displ_x;
                    (*nd)(1) += displ_y;
                    (*nd)(2) += displ_z;
                }
            }
            writeOutput(mesh, equation_systems, t);
        }
    }

    // preCICE END
    interface.finalize();
    std::cout << "Exiting StructureSolver" << std::endl;
    // libMesh END

    if (exo_io != NULL)
        delete exo_io;
    exo_io = NULL;

    std::cout << "All done ;)\n";

    return 0;
}

void read_parameters(int argc, char **argv)
{
    if (argc < 9)
    {
        err << "Usage: " << argv[0] << " -nu -e -t -mesh -config -dt [-out] [-d]\n"
            << "-nu: Possion's ratio (required)\n"
            << "-e: Elastic modulus E (required)\n"
            << "-t: Thickness (required)\n"
            << "-mesh: Input mesh file (*.xda or *.msh, required)\n"
            << "-config: preCICE configuration file (required)\n"
            << "-dt: preCICE max time step length (required, same as in config XML)\n"
            << "-out: Output file name (without extension, optional)\n"
            << "-d: Additional messages (1=on, 0=off (default))\n";

        libmesh_error_msg("Error, must choose valid parameters.");
    }

    // Parse command line
    GetPot command_line (argc, argv);

    if ( command_line.search(1, "-d") )
        debug = (command_line.next(0) == 1? true : false);

    if ( command_line.search(1, "-nu") )
        nu = command_line.next(0.3);
    else
        libmesh_error_msg("ERROR: Poisson's ratio nu not specified!");

    if ( command_line.search(1, "-e") )
        em = command_line.next(1.0e6);
    else
        libmesh_error_msg("ERROR: Elastic modulus E not specified!");

    if ( command_line.search(1, "-t") )
        thickness = command_line.next(1.0);
    else
        libmesh_error_msg("ERROR: Mesh thickness t not specified!");

    if ( command_line.search(1, "-mesh") )
        in_filename = command_line.next("mesh.xda");
    else
        libmesh_error_msg("ERROR: Mesh file not specified!");

    if ( command_line.search(1, "-config") )
        config_filename = command_line.next("config");
    else
        libmesh_error_msg("ERROR: preCICE configuration file not specified!");

    if ( command_line.search(1, "-dt") )
        deltaT = command_line.next(0.01);
    else
        libmesh_error_msg("ERROR: preCICE max time step length not specified!");

    if ( command_line.search(1, "-out") )
    {
        out_filename = command_line.next("out");
        isOutfileSet = true;
    }
    else
        isOutfileSet = false;

    std::cout << "Run program with parameters: extra messages = " << (debug?"true":"false") << ", nu = " << nu << ", E = " << em << ", t = " << thickness;
    std::cout << ", config-file = " << config_filename << ", in-file = " << in_filename;
    if (isOutfileSet)
        std::cout << ", out-file = " << out_filename;
    std::cout << "\n";
}

void initMaterialMatrices()
{
    /*     /                   \
     *     | 1    nu      0    |
     * D = | nu   1       0    |
     *     | 0    0   (1-nu)/2 |
     *     \                   /
     */
    Dp.resize(3,3);
    Dp(0,0) = 1.0; Dp(0,1) = nu;
    Dp(1,0) = nu;  Dp(1,1) = 1.0;
    Dp(2,2) = (1.0-nu)/2.0;
    Dm = Dp; // base matrix is same for Dm and Dp
    //          E * t³
    // Dp = ------------- * D
    //       12 * (1-nu²)
    Dp *= em*pow(thickness,3.0)/(12.0*(1.0-nu*nu)); // material matrix for plate part
    //         E
    // Dm = ------- * D
    //       1-nu²
    Dm *= em/(1.0-nu*nu); // material matrix for membrane part
}

void initElement(const Elem **elem, DenseMatrix<Real> &transUV, DenseMatrix<Real> &trafo, DenseMatrix<Real> &dphi, Real *area)
{
    ElemType type = (*elem)->type();
    Node *ndi = NULL, *ndj = NULL;
    Node U,V,W;

    if (type == TRI3)
    {
        // transform arbirtrary 3d triangle down to xy-plane with node A at origin (implicit):
        ndi = (*elem)->get_node(0); // node A
        if (debug) { std::cout << "node A:\n"; ndi->print_info(std::cout); }
        ndj = (*elem)->get_node(1); // node B
        if (debug) { std::cout << "node B:\n"; ndj->print_info(std::cout); }
        U = (*ndj)-(*ndi); // U = B-A
        ndj = (*elem)->get_node(2); // node C
        if (debug) { std::cout << "node C:\n"; ndj->print_info(std::cout); }
        V = (*ndj)-(*ndi); // V = C-A

        transUV.resize(3,2);
        for (int i = 0; i < 3; i++)
        {                        // node A lies in local origin (per definition)
            transUV(i,0) = U(i); // node B in global coordinates (triangle translated s.t. A lies in origin)
            transUV(i,1) = V(i); // node C in global coordinates ( -"- )
        }
        /* transUV [ b_x, c_x ]
         *         [ b_y, c_y ]
         *         [ b_z, c_z ]
         */

        // area of triangle is half the length of the cross product of U and V
        W = U.cross(V);
        *area = 0.5*W.size();

        U = U.unit();   // local x-axis unit vector
        W = W.unit();   // local z-axis unit vector, normal to triangle
        V = W.cross(U); // local y-axis unit vector (cross prod of 2 normalized vectors is automatically normalized)
    }
    else if (type == QUAD4)
    {
        // transform planar 3d quadrilateral down to xy-plane with node A at origin:
        Node nI,nJ,nK,nL;
        ndi = (*elem)->get_node(0); // node A
        //std::cout << "node A (" << ndi->processor_id() << "):\n"; ndi->print_info(std::cout);}
        ndj = (*elem)->get_node(1); // node B
        //std::cout << "node B (" << ndj->processor_id() << "):\n"; ndj->print_info(std::cout);}
        nI = (*ndi) + 0.5*((*ndj)-(*ndi)); // nI = midpoint on edge AB
        ndi = (*elem)->get_node(2); // node C
        //std::cout << "node C (" << ndi->processor_id() << "):\n"; ndi->print_info(std::cout);}
        nJ = (*ndj) + 0.5*((*ndi)-(*ndj)); // nJ = midpoint on edge BC
        ndj = (*elem)->get_node(3); // node D
        //std::cout << "node D (" << ndj->processor_id() << "):\n"; ndj->print_info(std::cout);}
        nK = (*ndi) + 0.5*((*ndj)-(*ndi)); // nK = midpoint on edge CD
        ndi = (*elem)->get_node(0); // node A
        nL = (*ndj) + 0.5*((*ndi)-(*ndj)); // nL = midpoint on edge DA
        ndj = (*elem)->get_node(2);

        //std::cout << "node i:\n"; nI.print_info(std::cout);
        //std::cout << "node j:\n"; nJ.print_info(std::cout);
        //std::cout << "node k:\n"; nK.print_info(std::cout);
        //std::cout << "node l:\n"; nL.print_info(std::cout);

        transUV.resize(3,4); // ({x,y,z},{A,B,C,D})
        for (int i = 0; i < 4; i++)
        {
            ndi = (*elem)->get_node(i);
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

    trafo.resize(3,3); // global to local transformation matrix
    for (int j = 0; j < 3; j++)
    {
        trafo(0,j) = U(j);
        trafo(1,j) = V(j);
        trafo(2,j) = W(j);
    }
    /* trafo [ u_x, u_y, u_z ]
     *       [ v_x, v_y, v_z ]
     *       [ w_x, w_y, w_z ]
     */

    // transform B and C (and D with QUAD4) to local coordinates and store results in the same place
    transUV.left_multiply(trafo);

    if (debug) {
        std::cout << "transUV:\n"; transUV.print(std::cout);
        std::cout << "\ntrafo:\n"; trafo.print(std::cout); std::cout << std::endl;
    }

    if (type == TRI3)
    {
        dphi.resize(3,2); // resizes matrix to 3 rows, 2 columns and zeros entries
        dphi(0,0) = -transUV(0,0); // x12 = x1-x2 = 0-x2 = -x2
        dphi(1,0) =  transUV(0,1); // x31 = x3-x1 = x3-0 = x3
        dphi(2,0) =  transUV(0,0)-transUV(0,1); // x23 = x2-x3
        dphi(0,1) = -transUV(1,0); // y12 = 0, stays zero, as node B and A lies on local x-axis and therefore y=0 for both
        dphi(1,1) =  transUV(1,1); // y31 = y3-y1 = y3-0 = y3
        dphi(2,1) =  transUV(1,0)-transUV(1,1); // y23 = y2-y3 = 0-y3 = -y3
    }
    else if (type == QUAD4)
    {
        dphi.resize(6,2); // resizes matrix to 6 rows, 2 columns and zeros entries
        dphi(0,0) = transUV(0,0)-transUV(0,1); // x12 = x1-x2
        dphi(1,0) = transUV(0,1)-transUV(0,2); // x23 = x2-x3
        dphi(2,0) = transUV(0,2)-transUV(0,3); // x34 = x3-x4
        dphi(3,0) = transUV(0,3)-transUV(0,0); // x41 = x4-x1

        dphi(0,1) = transUV(1,0)-transUV(1,1); // y12 = y1-y2
        dphi(1,1) = transUV(1,1)-transUV(1,2); // y23 = y2-y3
        dphi(2,1) = transUV(1,2)-transUV(1,3); // y34 = y3-y4
        dphi(3,1) = transUV(1,3)-transUV(1,0); // y41 = y4-y1

        dphi(4,0) = transUV(0,2)-transUV(0,0); // x31 = x3-x1 // TODO: kann wahrscheinlich raus; wird nicht mehr gebraucht
        dphi(4,1) = transUV(1,2)-transUV(1,0); // y31 = y3-y1
        dphi(5,0) = transUV(0,3)-transUV(0,1); // x42 = x4-x2
        dphi(5,1) = transUV(1,3)-transUV(1,1); // y42 = y4-y2

        *area = 0.0;
        for (int i = 0; i < 4; i++) // Gauss's area formula
            *area += transUV(0,i)*transUV(1,(i+1)%4) - transUV(0,(i+1)%4)*transUV(1,i); // x_i*y_{i+1} - x_{i+1}*y_i = det((x_i,x_i+1),(y_i,y_i+1))
        *area *= 0.5;
    }
}

void calcPlane(ElemType type, DenseMatrix<Real> &transUV, DenseMatrix<Real> &dphi, Real *area, DenseMatrix<Real> &Ke_m)
{
    if (type == TRI3)
    {
        DenseMatrix<Real> B_m(3,6);
        B_m(0,0) =  dphi(2,1); //  y23
        B_m(0,2) =  dphi(1,1); //  y31
        B_m(0,4) =  dphi(0,1); //  y12
        B_m(1,1) = -dphi(2,0); // -x23
        B_m(1,3) = -dphi(1,0); // -x31
        B_m(1,5) = -dphi(0,0); // -x12
        B_m(2,0) = -dphi(2,0); // -x23
        B_m(2,1) =  dphi(2,1); //  y23
        B_m(2,2) = -dphi(1,0); // -x31
        B_m(2,3) =  dphi(1,1); //  y31
        B_m(2,4) = -dphi(0,0); // -x12
        B_m(2,5) =  dphi(0,1); //  y12
        B_m *= 1.0/(2.0*(*area));

        // Ke_m = t*A* B^T * Dm * B
        Ke_m = Dm; // Ke_m = 3x3
        Ke_m.right_multiply(B_m); // Ke_m = 3x6
        Ke_m.left_multiply_transpose(B_m); // Ke_m = 6x6
        Ke_m *= thickness*(*area); // considered thickness and area is constant all over the element
    }
    else if (type == QUAD4)
    {
        // quadrature points definitions:
        Real root = sqrt(1.0/3.0);

        DenseMatrix<Real> B_m, G(4,8);
        DenseMatrix<Real> J(2,2);
        DenseVector<Real> shapeQ4(4);
        DenseVector<Real> dhdr(4), dhds(4);
        // we iterate over the 2x2 Gauss quadrature points (+- sqrt(1/3)) with weight 1
        Ke_m.resize(8,8);
        for (int ii = 0; ii < 2; ii++)
        {
            Real r = pow(-1.0, ii) * root; // +/- root
            for (int jj = 0; jj < 2; jj++)
            {
                Real s = pow(-1.0, jj) * root; // +/- root

                shapeQ4(0) = 0.25*(1-r)*(1-s);
                shapeQ4(1) = 0.25*(1+r)*(1-s);
                shapeQ4(2) = 0.25*(1+r)*(1+s);
                shapeQ4(3) = 0.25*(1-r)*(1+s);

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

                for (int i = 0; i < 4; i++)
                {
                    G(0,2*i)   = dhdr(i);
                    G(1,2*i)   = dhds(i);
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
    }
}

void calcPlate(ElemType type, DenseMatrix<Real> &dphi, Real *area, DenseMatrix<Real> &Ke_p)
{
    std::vector<double> sidelen;
    DenseMatrix<Real> Hcoeffs;

    if (type == TRI3)
    {
        std::vector< std::vector<double> > qps(3);
        for (unsigned int i = 0; i < qps.size(); i++)
            qps[i].resize(2);
        qps[0][0] = 1.0/6.0; qps[0][1] = 1.0/6.0;
        qps[1][0] = 2.0/3.0; qps[1][1] = 1.0/6.0;
        qps[2][0] = 1.0/6.0; qps[2][1] = 2.0/3.0;

        // side-lengths squared:
        sidelen.resize(3);
        sidelen[0] = pow(dphi(0,0), 2.0) + pow(dphi(0,1), 2.0); // side AB, x12^2 + y12^2 (=0) -> x12^2 = x2^2
        sidelen[1] = pow(dphi(1,0), 2.0) + pow(dphi(1,1), 2.0); // side AC, x31^2 + y31^2
        sidelen[2] = pow(dphi(2,0), 2.0) + pow(dphi(2,1), 2.0); // side BC, x23^2 + y23^2

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

            DenseMatrix<Real> B;
            evalBTri(Hcoeffs, qps[i][0], qps[i][1], dphi, B);

            DenseMatrix<Real> Y(3,3);
            Y(0,0) = pow(dphi(2,1),2.0);
            Y(0,1) = pow(dphi(1,1),2.0);
            Y(0,2) = dphi(2,1)*dphi(1,1);
            Y(1,0) = pow(dphi(2,0),2.0);
            Y(1,1) = pow(dphi(1,0),2.0);
            Y(1,2) = dphi(1,0)*dphi(2,0);
            Y(2,0) = -2.0*dphi(2,0)*dphi(2,1);
            Y(2,1) = -2.0*dphi(1,0)*dphi(1,0);
            Y(2,2) = -dphi(2,0)*dphi(1,1)-dphi(1,0)*dphi(2,1);
            Y *= 1.0/(4.0*pow(*area,2.0));

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

        Ke_p *= 2.0*(*area);
    }
    else if (type == QUAD4)
    {
        // side-lengths squared:
        sidelen.resize(4);
        sidelen[0] = pow(dphi(0,0), 2.0) + pow(dphi(0,1), 2.0); // side AB, x12^2 + y12^2
        sidelen[1] = pow(dphi(1,0), 2.0) + pow(dphi(1,1), 2.0); // side BC, x23^2 + y23^2
        sidelen[2] = pow(dphi(2,0), 2.0) + pow(dphi(2,1), 2.0); // side CD, x34^2 + y34^2
        sidelen[3] = pow(dphi(3,0), 2.0) + pow(dphi(3,1), 2.0); // side DA, x41^2 + y41^2

        if (debug)
            std::cout << "lij^2 = (" << sidelen[0] << ", " << sidelen[1] << ", " << sidelen[2] << ", " << sidelen[3] << ")\n";

        Hcoeffs.resize(5,4); // [ a_k, b_k, c_k, d_k, e_k ], k=5,6,7,8
        for (int i = 0; i < 4; i++)
        {
            Hcoeffs(0,i) = -dphi(i,0)/sidelen[i]; // a_k
            Hcoeffs(1,i) = 0.75 * dphi(i,0) * dphi(i,1) / sidelen[i]; // b_k
            Hcoeffs(2,i) = (0.25 * pow(dphi(i,0), 2.0) - 0.5 * pow(dphi(i,1), 2.0))/sidelen[i]; // c_k
            Hcoeffs(3,i) = -dphi(i,1)/sidelen[i]; // d_k
            Hcoeffs(4,i) = (0.25 * pow(dphi(i,1), 2.0) - 0.5 * pow(dphi(i,0), 2.0))/sidelen[i]; // e_k
        }

        if (debug) {
            std::cout << "Hcoeffs:\n";
            Hcoeffs.print(std::cout);
            std::cout << std::endl;
        }

        // resize the current element matrix and vector to an appropriate size
        Ke_p.resize(12, 12);

        // quadrature points definitions:
        Real root = sqrt(1.0/3.0);

        DenseMatrix<Real> J(2,2), Jinv(2,2);

        for (int ii = 0; ii < 2; ii++)
        {
            Real r = pow(-1.0, ii) * root; // +/- sqrt(1/3)
            for (int jj = 0; jj < 2; jj++)
            {
                Real s = pow(-1.0, jj) * root; // +/- sqrt(1/3)

                if (debug)
                    std::cout << "(r,s) = " << r << ", " << s << "\n";

                J(0,0) = (dphi(0,0)+dphi(2,0))*s - dphi(0,0) + dphi(2,0);
                J(0,1) = (dphi(0,1)+dphi(2,1))*s - dphi(0,1) + dphi(2,1);
                J(1,0) = (dphi(0,0)+dphi(2,0))*r - dphi(1,0) + dphi(3,0);
                J(1,1) = (dphi(0,1)+dphi(2,1))*r - dphi(1,1) + dphi(3,1);
                J *= 0.25;

                if (debug) {
                    std::cout << "J:\n";
                    J.print(std::cout);
                    std::cout << std::endl;
                }

                Real det = J.det();
                if (debug)
                    std::cout << "|J| = " << det << "\n";

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

                DenseMatrix<Real> B;
                evalBQuad(Hcoeffs, r, s, Jinv, B);

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
            }
        }
    }
}

void evalBTri(DenseMatrix<Real>& C, Real L1, Real L2, DenseMatrix<Real> &dphi_p, DenseMatrix<Real> &out)
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

void evalBQuad(DenseMatrix<Real>& Hcoeffs, Real xi, Real eta, DenseMatrix<Real> &Jinv, DenseMatrix<Real> &out)
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

void constructStiffnessMatrix(ElemType type, DenseMatrix<Real> &Ke_m, DenseMatrix<Real> &Ke_p, DenseMatrix<Real> &K_out)
{
    int nodes = 3;
    if (type == TRI3)
        nodes = 3;
    else if (type == QUAD4)
        nodes = 4;

    K_out.resize(6*nodes, 6*nodes);

    // copy values from submatrices into overall element matrix:
    for (int i = 0; i < nodes; i++)
    {
        for (int j = 0; j < nodes; j++)
        {
            // submatrix K_ij [6x6]
            K_out(  6*i,    6*j)   = Ke_m(2*i,  2*j);   // uu
            K_out(  6*i,    6*j+1) = Ke_m(2*i,  2*j+1); // uv
            K_out(  6*i+1,  6*j)   = Ke_m(2*i+1,2*j);   // vu
            K_out(  6*i+1,  6*j+1) = Ke_m(2*i+1,2*j+1); // vv
            K_out(2+6*i,  2+6*j)   = Ke_p(3*i,  3*j);   // ww
            K_out(2+6*i,  2+6*j+1) = Ke_p(3*i,  3*j+1); // wx
            K_out(2+6*i,  2+6*j+2) = Ke_p(3*i,  3*j+2); // wy
            K_out(2+6*i+1,2+6*j)   = Ke_p(3*i+1,3*j);   // xw
            K_out(2+6*i+1,2+6*j+1) = Ke_p(3*i+1,3*j+1); // xx
            K_out(2+6*i+1,2+6*j+2) = Ke_p(3*i+1,3*j+2); // xy
            K_out(2+6*i+2,2+6*j)   = Ke_p(3*i+2,3*j);   // yw
            K_out(2+6*i+2,2+6*j+1) = Ke_p(3*i+2,3*j+1); // yx
            K_out(2+6*i+2,2+6*j+2) = Ke_p(3*i+2,3*j+2); // yy
        }
    }

    Real max_value;
    for (int zi = 0; zi < nodes; zi++)
    {
        for (int zj = 0; zj < nodes; zj++)
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
            K_out(5+6*zi,5+6*zj) = max_value;
        }
    }
}

void localToGlobalTrafo(ElemType type, DenseMatrix<Real> &trafo, DenseMatrix<Real> &Ke_inout)
{
    int nodes = 3;
    if (type == TRI3)
        nodes = 3;
    else if (type == QUAD4)
        nodes = 4;

    // transform Ke from local back to global with transformation matrix T:
    DenseMatrix<Real> KeSub(6,6);
    DenseMatrix<Real> KeNew(6*nodes,6*nodes);
    DenseMatrix<Real> TSub(6,6);
    for (int k = 0; k < 2; k++) // copy trafo two times into TSub (cf comment beneath)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                TSub(3*k+i,3*k+j) = trafo(i,j);
    /* TSub: [ux, vx, wx,  0,  0,  0]
     *       [uy, vy, wy,  0,  0,  0]
     *       [uz, vz, wz,  0,  0,  0]
     *       [0 ,  0,  0, ux, vx, wx]
     *       [0 ,  0,  0, uy, vy, wy]
     *       [0 ,  0,  0, uz, vz, wz] */

    for (int i = 0; i < nodes; i++)
    {
        for (int j = 0; j < nodes; j++)
        {
            for (int k = 0; k < 6; k++) // copy values into KeSub for right format to transformation
                for (int l = 0; l < 6; l++)
                    KeSub(k,l) = Ke_inout(i*6+k,j*6+l);

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
            for (int i = 0; i < nodes; i++)
                for (int j = 0; j < nodes; j++)
                    Ke_inout(nodes*alpha+i,nodes*beta+j) = KeNew(6*i+alpha,6*j+beta);
}

void contribRHS(const Elem **elem, DenseVector<Real> &Fe, std::unordered_set<unsigned int> *processedNodes)
{
    unsigned int nsides = (*elem)->n_sides();
    Fe.resize(6*nsides);

    for (unsigned int side = 0; side < nsides; side++)
    {
        Node* node = (*elem)->get_node(side);
        dof_id_type id = node->id();
        // do not process nodes that are owned by another process
        if (node->processor_id() != global_processor_id())
            continue;

        if (processedNodes->find(id) == processedNodes->end())
        {
            processedNodes->insert(id);

            std::unordered_map<libMesh::dof_id_type,int>::const_iterator preCICE_id = id_map.find(id);
            // forces don't need to be transformed since we bring the local stiffness matrix
            // back to global co-sys directly in libmesh-format:
            if (preCICE_id != id_map.end())
            {
                if (dimensions == 3)
                {
                    for (int i = 0; i < dimensions; i++)
                        Fe(side+nsides*i) = forces[preCICE_id->second*dimensions+i];
                    // u_i   (0, 1,   ..., n-1)
                    // v_i   (n, n+1, ...,2n-1)
                    // w_i   (2n,2n+1,...,3n-1)
                }
                else // dimensions == 2 otherwise program would have been exited at the beginning
                {
                    if (ignoredAxis == 3) // xy-plane
                    {
                        Fe(side)        = forces[preCICE_id->second*2];
                        Fe(side+nsides) = forces[preCICE_id->second*2+1];
                    }
                    else if (ignoredAxis == 5) // xz-plane
                    {
                        Fe(side)          = forces[preCICE_id->second*2];
                        Fe(side+nsides*2) = forces[preCICE_id->second*2+1];
                    }
                    else if (ignoredAxis == 6) // yz-plane
                    {
                        Fe(side+nsides)   = forces[preCICE_id->second*2];
                        Fe(side+nsides*2) = forces[preCICE_id->second*2+1];
                    }
                }
                if (debug)
                    std::cout << "force = " << Fe(side) << "," << Fe(side+nsides) << "," << Fe(side+nsides*2) << "\n";
            }
        }
    }
}

// Matrix and right-hand side assemble
void assemble_elasticity(EquationSystems &es, const std::string &system_name)
{
    libmesh_assert_equal_to (system_name, "Elasticity");

    // get the mesh
    const MeshBase& mesh = es.get_mesh();

    // get a reference to the system
    LinearImplicitSystem& system = es.get_system<LinearImplicitSystem>("Elasticity");

    // A reference to the DofMap object for this system.
    // The DofMap object handles the index translation from node and element numbers to degree of freedom numbers.
    const DofMap& dof_map = system.get_dof_map();

    // stiffness matrix Ke for overall shell element,
    //                  Ke_m for membrane part,
    //                  Ke_p for plate part
    DenseMatrix<Number> Ke, Ke_m, Ke_p;
    // RHS / force-momentum-vector Fe
    DenseVector<Number> Fe;

    // indices (positions) of node's variables in system matrix and RHS:
    std::vector<dof_id_type> dof_indices;

    DenseMatrix<Real> trafo; // global to local coordinate system transformation matrix
    DenseMatrix<Real> transUV; // saves the transformed vectors of the element's nodes
    DenseMatrix<Real> dphi; // contains the first partial derivatives of the element
    Real area = 0.0; // the area of the element

    // every node must contribute only once to the RHS. Since a node can be shared by many elements
    // 'processedNodes' keeps track of already used nodes and prevent further processing
    std::unordered_set<dof_id_type> processedNodes;
    processedNodes.reserve(mesh.n_local_nodes()); // we exactly need only as many nodes as the process got appointed

    // iterator for iterating through the elements of the mesh:
    MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();
    // for all local elements elem in mesh do:
    for ( ; el != end_el; ++el)
    {
        const Elem* elem = *el;

        // get the local to global DOF-mappings for this element
        dof_map.dof_indices (elem, dof_indices);

        ElemType type = elem->type();

        initElement(&elem, transUV, trafo, dphi, &area);

        calcPlane(type, transUV, dphi, &area, Ke_m);
        calcPlate(type, dphi, &area, Ke_p);

        constructStiffnessMatrix(type, Ke_m, Ke_p, Ke);

        localToGlobalTrafo(type, trafo, Ke);

        contribRHS(&elem, Fe, &processedNodes);

        dof_map.constrain_element_matrix_and_vector(Ke, Fe, dof_indices);

        system.matrix->add_matrix (Ke, dof_indices);
        system.rhs->add_vector    (Fe, dof_indices);
    }
}

void writeOutput(Mesh &mesh, EquationSystems &es, int timestep)
{
    if (!isOutfileSet)
        return;

    // Plot the solution
    std::ostringstream file_name;
    file_name << out_filename << "_"
              << std::setw(3)
              << std::setfill('0')
              << std::right
              << timestep
              //<< ".e";
              << ".pvtu";

    VTKIO (mesh).write_equation_systems(file_name.str(), es);
    //ExodusII_IO exo_io(mesh);//.write_equation_systems(file_name.str(), es);
    //if (timestep > 0)
    //    exo_io.append(true);
    //exo_io.write_timestep(file_name.str(), es, timestep, timestep/(float)220.0);
}
