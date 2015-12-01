#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "precice/SolverInterface.hpp"

using std::cout;
using std::endl;

using namespace precice;
using namespace precice::constants;

void printData (const std::vector<double>& data)
{
  std::cout << "Received data = " << data[0];
  for (size_t i=1; i < data.size(); i++){
    std::cout << ", " << data[i];
  }
  std::cout << std::endl;
}

int main (int argc, char **argv)
{
    int rank, size;

    MPI_Init (&argc, &argv);	/* starts MPI */
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* get current process id */
    MPI_Comm_size (MPI_COMM_WORLD, &size);	/* get number of processes */
    cout << "Starting Fluid Solver Dummy (" << rank << "/" << size << ")..." << endl;

    if ( argc != 2 )
    {
        cout << endl;
        cout << "Usage: " << argv[0] <<  " configurationFileName" << endl;
        cout << endl;
        cout << "configurationFileName: preCICE XML-configuration file" << endl;
        return -1;
    }

    std::string configFileName(argv[1]);
    int N        = 0;
    if (rank == 0)
        if (size == 1)
            N = 25;
        else
            N = 11;
    else
        N = 14;

    std::cout << "N: " << std::endl;

    std::string solverName = "FLUID";

    cout << "Configure preCICE..." << endl;
    // Initialize the solver interface with our name, our process index (like rank) and the total number of processes.
    SolverInterface interface(solverName, rank, size);
    // Provide the configuration file to precice. After configuration a usuable state of that SolverInterface is reached.
    // Reads the XML file and contacts the server, if used.
    interface.configure(configFileName);
    cout << "preCICE configured..." << endl;

    // init data
    int i;
    double *f, *f_n, *d, *d_n;
    int dimensions = interface.getDimensions();

    f     = new double[N*dimensions]; // Force
    f_n   = new double[N*dimensions];
    d     = new double[N*dimensions]; // Displacements
    d_n   = new double[N*dimensions];

    //precice stuff
    int meshID = interface.getMeshID("Fluid_Nodes");
    int dID = interface.getDataID("Displacements", meshID);
    int fID = interface.getDataID("Stresses", meshID);
    int *vertexIDs;
    vertexIDs = new int[N];
    double *grid;
    grid = new double[dimensions*N];

    for (i = 0; i < N; i++)
    {
        for (int dim = 0; dim < dimensions; dim++)
        {
            d[i*dimensions+dim]   = 1.0;
            d_n[i*dimensions+dim] = 1.0;
            f[i*dimensions+dim]   = 0.0;
            f_n[i*dimensions+dim] = 0.0;
        }
    }

    if (size == 1)
    {
        for (int k = 0; k < 11; k++)
        {
            grid[k*dimensions]   = 0.45;
            //grid[k*dimensions+1] = -0.001;
            grid[k*dimensions+1] = 0.2-k*0.02;
        }
        for (int k = 11; k < 22; k++)
        {
            grid[k*dimensions]   = 0.55;
            //grid[k*dimensions+1] = -0.001;
            grid[k*dimensions+1] = (k-11.0)*0.02;
        }
        for (int k = 22; k < 25; k++)
        {
            grid[k*dimensions]   = 0.475 + (k-22.0)*0.025;
            //grid[k*dimensions+1] = -0.001;
            grid[k*dimensions+1] = 0.2;
        }
        for (int k = 0; k < 25; k++)
            cout << "grid [" << grid[k*2] << ", " << grid[k*2+1] << "]\n";
    }
    else
    {
        if (rank == 0)
        {
            for (int k = 0; k < 11; k++)
            {
                grid[k*dimensions]   = 0.45;
                //grid[k*dimensions+1] = -0.001;
                grid[k*dimensions+1] = k*0.02;
            }
        }
        else
        {
            for (int k = 0; k < 11; k++)
            {
                grid[k*dimensions]   = 0.55;
                //grid[k*dimensions+1] = -0.001;
                grid[k*dimensions+1] = k*0.02;
            }
            for (int k = 11; k < 14; k++)
            {
                grid[k*dimensions]   = 0.475 + (k-11.0)*0.025;
                //grid[k*dimensions+1] = -0.001;
                grid[k*dimensions+1] = 0.2;
            }
        }
    }

    int t = 0; //number of timesteps

    interface.setMeshVertices(meshID, N, grid, vertexIDs);

    cout << "Fluid: init precice..." << endl;
    interface.initialize();


    if (interface.isActionRequired(actionWriteInitialData()))
    {
        interface.writeBlockVectorData(fID, N, vertexIDs, f);
        //interface.initializeData();
        interface.fulfilledAction(actionWriteInitialData());
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
            for (int i = 0; i < 11; i++)
            {
                f[i*dimensions] = 1.0+sin(t/25.01);
            }
            for (int i = 11; i < 22; i++)
            {
                f[i*dimensions] = -1.0;
            }
        }
        else
        {
            if (rank == 0)
            {
                for (int i = 0; i < 11; i++)
                {
                    f[i*dimensions] = 1.0+sin(t/25.01);
                }

            }
            else
            {
                for (int i = 0; i < 11; i++)
                {
                    f[i*dimensions] = -1.0;
                }
            }
        }

        interface.writeBlockVectorData(fID, N, vertexIDs, f);
        interface.advance(0.01);
        interface.readBlockVectorData(dID, N, vertexIDs, d);

        if (interface.isActionRequired(actionReadIterationCheckpoint()))
        { // i.e. not yet converged
            cout << "Iterate" << endl;
            interface.fulfilledAction(actionReadIterationCheckpoint());
        }
        else
        {
            cout << "Fluid: Advancing in time, finished timestep: " << t << endl;
            t++;

            for ( i = 0; i < N; i++)
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
    cout << "Exiting FluidSolver" << endl;

    MPI_Finalize();

    return 0;
}
