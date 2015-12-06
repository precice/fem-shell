#!/bin/bash

echo "Check for FEM-shell..."
if [ -e FEM-shell ]
then
  echo "...FEM-shell executable found."
else
  echo "...FEM-shell executable NOT found."
  echo "...building FEM-shell..."
  scons
  if [ -e FEM-shell ]
  then
    echo "... building successful."
  else
    echo "... building failed."
    exit
  fi
fi
echo "Run examples..."
if [ -d example-out ]
then
  echo "...output folder found."
else
  mkdir example-out
  echo "...output folder created."
fi
if [ -d example-meshes ]
then
  echo "...example meshes folder found."
else
  echo "...example meshes folder NOT found."
  echo "Exiting"
  exit
fi
echo "Test A: "
./fem-shell -nu 0.25 -e 30000 -t 1.0 -mesh example-meshes/test_A_uv_t.xda -out example-out/test_A_uv_t
echo "Test B: "
./fem-shell -nu 0.25 -e 30000 -t 1.0 -mesh example-meshes/test_B_uv_q.xda -out example-out/test_B_uv_q
echo "Test C: "
./fem-shell -nu 0.3 -e 10.92 -t 1.0 -mesh example-meshes/test_C_w_tA16.xda -out example-out/test_C_w_tA16
echo "Test D: "
./fem-shell -nu 0.3 -e 1e7 -t 0.5 -mesh example-meshes/test_D_w_q_uni16.xda -out example-out/test_D_w_q_uni16
echo "Test E: "
./fem-shell -nu 0.25 -e 10000 -t 0.25 -mesh example-meshes/test_E_uvw_t.xda -out example-out/test_E_uvw_t
echo "Test F: "
./fem-shell -nu 0.3 -e 1.7472e7 -t 0.01 -mesh example-meshes/test_F_032_ss_uni.xda -out example-out/test_F_032_ss_uni
echo "Test G: "
mpirun -n 2 ./fem-shell -nu 0.3 -e 1e7 -t 0.5 -mesh example-meshes/test_G_mpi_64_q.xda -out example-out/test_G_mpi_64_q
echo "....all examples finished!"
