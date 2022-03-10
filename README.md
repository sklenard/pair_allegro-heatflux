# `pair_allegro`: LAMMPS pair style for Allegro

This pair style allows you to use Allegro models in LAMMPS simulations. Allegro is designed to enable parallelism, and so `pair_allegro` **supports MPI in LAMMPS**. It also supports OpenMP (better performance) or Kokkos (best performance) for accelerating the pair style.

## Pre-requisites

* PyTorch or LibTorch >= 1.10.0

## Usage in LAMMPS

```
pair_style	allegro
# or pair_style	allegro/kk for Kokkos
pair_coeff	* * deployed.pth <type name 1> <type name 2> ...
```
where `deployed.pth` is the filename of your trained model.

The names after the model path `deployed.pth` indicate, in order, the names of the Allegro model's atom types to use for LAMMPS atom types 1, 2, and so on. The number of names given must be equal to the number of atom types in the LAMMPS configuration (not the Allegro model!). 
The given names must be consistent with the names specified in the Allegro training YAML in `chemical_symbol_to_type` or `type_names`.

## Building LAMMPS with this pair style

### Download LAMMPS
```bash
git clone --depth 1 git@github.com:lammps/lammps
```
or your preferred method.
(`--depth 1` prevents the entire history of the LAMMPS repository from being downloaded.)

### Download this repository
```bash
git clone git@github.com:mir-group/pair_allegro
```
or by downloading a ZIP of the source.

### Patch LAMMPS
#### Automatically
From the `pair_allegro` directory, run:
```bash
./patch_lammps.sh /path/to/lammps/
```

#### Libtorch
If you have PyTorch installed and are **NOT** using Kokkos:
```bash
cd lammps
mkdir build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```
If you don't have PyTorch installed **OR** are using Kokkos, you need to download LibTorch from the [PyTorch download page](https://pytorch.org/get-started/locally/). **Ensure you download the cxx11 ABI version.** Unzip the downloaded file, then configure LAMMPS:
```bash
cd lammps
mkdir build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

#### MKL
CMake will look for MKL automatically. If it cannot find it (`MKL_INCLUDE_DIR` is not found) and you are using a Python environment, a simple solution is to run `conda install mkl-include` or `pip install mkl-include` and append:
```
-DMKL_INCLUDE_DIR="$CONDA_PREFIX/include"
```
to the `cmake` command if using a `conda` environment, or
```
-DMKL_INCLUDE_DIR=`python -c "import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()[\"include\"]).parent)"`
```
if using plain Python and `pip`.

#### CUDA
CMake will look for CUDA and cuDNN. You may have to explicitly provide the path for your CUDA installation (e.g. `-DCUDA_TOOLKIT_ROOT_DIR=/usr/lib/cuda/`).

Note that the CUDA that comes with PyTorch when installed with `conda` (the `cudatoolkit` package) is usually insufficient (see [here](https://github.com/pytorch/extension-cpp/issues/26), for example) and you may have to install full CUDA seperately. A minor version mismatch between the available full CUDA version and the version of `cudatoolkit` is usually *not* a problem, as long as the system CUDA is equal or newer. (For example, PyTorch's requested `cudatoolkit==11.3` with a system CUDA of 11.4 works, but a system CUDA 11.1 will likely fail.) cuDNN is also required by PyTorch.

#### With OpenMP (optional, better performance)
`pair_allegro` supports the use of OpenMP to accelerate certain parts of the pair style.

#### With Kokkos (GPU, optional, best performance)
`pair_allegro` supports the use of Kokkos to accelerate certain parts of the pair style on the GPU to avoid host-GPU transfers.
`pair_allegro` supports two setups for Kokkos: pair_style and model both on CPU, or both on GPU. Please ensure you build LAMMPS with the appropriate Kokkos backends enabled for your usecase. For example, to use CUDA GPUs, add:
```
-DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON
```
to your `cmake` command.

### Building LAMMPS
```bash
make -j$(nproc)
```
This gives `lammps/build/lmp`, which can be run as usual with `/path/to/lmp -in in.script`. If you specify `-DCMAKE_INSTALL_PREFIX=/somewhere/in/$PATH` (the default is `$HOME/.local`), you can do `make install` and just run `lmp -in in.script`.