# Install Python Virtual Environment

sudo apt install python3.10-venv

# Clone LAMMPS

git clone https://github.com/lammps/lammps.git
cd lammps
mkdir build && cd build
cmake -D BUILD_MPI=OFF -D BUILD_SHARED_LIBS=ON -D PKG_PYTHON=ON ../cmake
make -j4
make install

cmake --build . --target install-python

export LD_LIBRARY_PATH=/home/XXXX/XXXX/lammps/build:$LD_LIBRARY_PATH

# Install PyMatgen

Allows easy .cif file parsing 

pip install pymatgen
pip install --upgrade pymatgen

# Install SciPy

pip install scipy
pip install --upgrade scipy

# Install PyVista

VTK package for python

pip install pyvista
pip install --upgrade pyvista