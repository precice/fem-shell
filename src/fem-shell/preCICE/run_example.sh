#!/bin/bash

echo "Check for FEM-shell-precice and FluidSolver..."
if [ -e FEM-shell-precice ]
then
  echo "...FEM-shell-precice executable found."
  if [ -e FluidSolver ]
  then
    echo "...FluidSolver executable found."
  else
    echo "...FluidSolver executable NOT found."
    echo "...building FluidSolver..."
    scons
    if [ -e FluidSolver ]
    then
      echo "... building successful."
    else
      echo "... building failed."
      exit
    fi
  fi
else
  echo "...FEM-shell-precice executable NOT found."
  echo "...building FEM-shell-precice..."
  scons
  if [ -e FEM-shell-precice ]
  then
    echo "... building successful."
  else
    echo "... building failed."
    exit
  fi
fi
echo "Run example..."
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
  echo "No mesh to test with. Exiting"
  exit
fi
echo "Start test: "
./FEM-shell-precice -nu 0.3 -e 1e6 -t 0.1 -mesh example-meshes/bending_tower_tri_test.xda -out example-out/bending_tower -config precice_config.xml -dt 0.01 -axis y 2>&1 &

./FluidSolver precice_config.xml 43 2>&1
