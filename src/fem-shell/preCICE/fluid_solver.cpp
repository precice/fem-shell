/********************************************\
*  Fluid-Solver-Dummy for use with preCICE   *
*  by Stephan Herb                           *
*                                            *
* Note: This code is not intended to be      *
*       flexible for every imaginable        *
*       scenario to simulate. The coupling   *
*       of the structure solver through pre- *
*       CICE was developed together with     *
*       solver dummy.                        *
\********************************************/

#include <iostream>
#include <stdlib.h>

#include <math.h>
#include <mpi.h>

#include "precice/SolverInterface.hpp"

using namespace precice;
using namespace precice::constants;

int main (int argc, char **argv)
{
    int rank, size;

    MPI_Init (&argc, &argv);	/* starts MPI */
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* get current process id */
    MPI_Comm_size (MPI_COMM_WORLD, &size);	/* get number of processes */
    std::cout << "Starting Fluid Solver Dummy (MPI " << (rank+1) << "/" << size << ")..." << std::endl;

    if ( argc < 2 )
    {
        std::cout << "Usage: " << argv[0] <<  " configurationFileName N" << std::endl;
        std::cout << "configurationFileName: preCICE XML-configuration file" << std::endl;
        std::cout << "N: Number of coupling interface nodes" << std::endl;
        return -1;
    }

    std::string configFileName(argv[1]);
    int N = atoi(argv[2]);
    // WARNING: totally adapted to a special test scenario. If this code should be used
    //          further, this has to be modified!
    if (rank == 0)
        if (size == 1)
            N = 43;
        else
            N = 21;
    else
        N = 22;

    std::cout << "N: " << N << std::endl;

    std::string solverName = "FLUID";

    std::cout << "Configure preCICE..." << std::endl;
    // Initialize the solver interface with our name, our process index (rank) and the total number of processes (size).
    SolverInterface interface(solverName, rank, size);
    // Provide the configuration file to preCICE. After configuration a usable state of that SolverInterface is reached.
    // Reads the XML file and contacts the server, if used.
    interface.configure(configFileName);
    std::cout << "preCICE configured..." << std::endl;

    // init data
    double *f, *f_n, *d, *d_n;
    int dimensions = interface.getDimensions();

    f     = new double[N*dimensions]; // Force
    f_n   = new double[N*dimensions];
    d     = new double[N*dimensions]; // Displacements
    d_n   = new double[N*dimensions];

    //preCICE stuff
    int meshID = interface.getMeshID("Fluid_Nodes");
    int dID    = interface.getDataID("Displacements", meshID);
    int fID    = interface.getDataID("Stresses", meshID);
    int *vertexIDs;
    vertexIDs = new int[N];
    double *grid;
    grid = new double[dimensions*N];

    // initalize data fields:
    for (int i = 0; i < N; i++)
    {
        for (int dim = 0; dim < dimensions; dim++)
        {
            d[i*dimensions+dim]   = 1.0;
            d_n[i*dimensions+dim] = 1.0;
            f[i*dimensions+dim]   = 0.0;
            f_n[i*dimensions+dim] = 0.0;
        }
    }

    if (size == 1) // initialze grid for single-threaded run
    {
        for (int k = 0; k < 21; k++) // left edge of tower
        {
            grid[k*dimensions]   = 3.0;
            grid[k*dimensions+1] = k*0.1;
            if (dimensions == 3)
                grid[k*dimensions+2] = 0.0;
        }
        for (int k = 21; k < 42; k++) // right edge of tower
        {
            grid[k*dimensions]   = 3.25;
            grid[k*dimensions+1] = (k-21.0)*0.1;
            if (dimensions == 3)
                grid[k*dimensions+2] = 0.0;
        }
        for (int k = 42; k < 43; k++) // top edge of tower
        {
            grid[k*dimensions]   = 3.125;
            grid[k*dimensions+1] = 2.0;
            if (dimensions == 3)
                grid[k*dimensions+2] = 0.0;
        }
        // debug output
        for (int k = 0; k < 43; k++)
            std::cout << "grid [" << grid[k*2] << ", " << grid[k*2+1] << "]" << std::endl;
    }
    else // for multi-threaded run
    {
        if (rank == 0)
        {
            for (int k = 0; k < 21; k++) // left edge of tower
            {
                grid[k*dimensions]   = 3.0;
                grid[k*dimensions+1] = k*0.1;
                if (dimensions == 3)
                    grid[k*dimensions+2] = 0.0;
            }
        }
        else
        {
            for (int k = 0; k < 21; k++) // right edge of tower
            {
                grid[k*dimensions]   = 3.25;
                grid[k*dimensions+1] = k*0.1;
                if (dimensions == 3)
                    grid[k*dimensions+2] = 0.0;
            }
            for (int k = 21; k < 22; k++) // top edge of tower
            {
                grid[k*dimensions]   = 3.125;
                grid[k*dimensions+1] = 2.0;
                if (dimensions == 3)
                    grid[k*dimensions+2] = 0.0;
            }
        }
    }

    int t = 0; //number of timesteps

    interface.setMeshVertices(meshID, N, grid, vertexIDs);

    std::cout << "Fluid: init precice..." << std::endl;
    interface.initialize();


    if (interface.isActionRequired(actionWriteInitialData()))
    {
        interface.writeBlockVectorData(fID, N, vertexIDs, f);
        interface.fulfilledAction(actionWriteInitialData());
    }

    for (int i = 0; i < N; i++)
    {
        std::cout << "vertexIDs[" << i << "] = " << vertexIDs[i] << " at grid position (" << grid[2*i] << ", " << grid[2*i+1] << ")" << std::endl;
    }

    interface.initializeData();

    if (interface.isReadDataAvailable())
    {
        interface.readBlockVectorData(dID, N, vertexIDs, d);
    }

    while (interface.isCouplingOngoing())
    {
        // When an implicit coupling scheme is used, checkpointing is required
        if (interface.isActionRequired(actionWriteIterationCheckpoint()))
        {
            interface.fulfilledAction(actionWriteIterationCheckpoint());
        }

        if (size == 1)
        {
            // create "magic" forces:
            for (int i = 0; i < 21; i++) //left edge of tower
            {
                f[i*dimensions] = 1.0 + sin(t/25.01);
                if (dimensions == 3)
                    f[i*dimensions+1] = 0.0;//0.1 + sin(t/25.01)/10.0;
            }
            /*for (int i = 21; i < 42; i++) // right edge of tower
            {
                f[i*dimensions] = 0.0;//(-1.5+sin(t/25.01))*(1.0-0.006*(i-34)*(i-34));
                f[i*dimensions+1] = 0.0;//-(i-21)*0.01;
            }*/
        }
        else
        {
            if (rank == 0)
            {
                for (int i = 0; i < 21; i++) //left edge of tower
                {
                    f[i*dimensions] = 1.0+sin(t/25.01);
                    if (dimensions == 3)
                        f[i*dimensions+1] = 0.1 + sin(t/25.01)/10.0;
                }

            }
        }

        interface.writeBlockVectorData(fID, N, vertexIDs, f);
        interface.advance(0.01);
        interface.readBlockVectorData(dID, N, vertexIDs, d);

        if (interface.isActionRequired(actionReadIterationCheckpoint()))
        {   // i.e. not yet converged
            std::cout << "Iterate" << std::endl;
            interface.fulfilledAction(actionReadIterationCheckpoint());
        }
        else
        {
            std::cout << "Fluid: Advancing in time, finished timestep: " << t << std::endl;
            t++;

            for (int i = 0; i < N; i++)
            {
                for (int dim = 0; dim < dimensions; dim++)
                {
                    f_n[i*dimensions+dim] = f[i*dimensions+dim];
                    d_n[i*dimensions+dim] = d[i*dimensions+dim];
                }
            }
        }
    }

    interface.finalize();
    std::cout << "Exiting FluidSolver" << std::endl;

    MPI_Finalize();

    return 0;
}
