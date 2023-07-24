# `pair_allegro-heatflux`

This pair style allows you to use Allegro models from the [`allegro`](https://github.com/mir-group/allegro) package in LAMMPS simulations. This implementation of `pair_allegro` includes heatflux and stress tensor calculation but requires that the mode computes partial forces. Therefore the allegro model should be deployed using https://github.com/sklenard/allegro-heatflux.

**Important note:** The current implementation may not include new features and/or bugfix from the original implementation (https://github.com/mir-group/pair_allegro)! In addition only the *standard* pair potential supports heatflux and virial calculations. They will be implemented in the kokkos pair potential in the future.

## Usage in LAMMPS

```
pair_style	allegro
pair_coeff	* * deployed.pth <Allegro type name for LAMMPS type 1> <Allegro type name for LAMMPS type 2> ...
```
where `deployed.pth` is the filename of your trained, **deployed** model.

The names after the model path `deployed.pth` indicate, in order, the names of the Allegro model's atom types to use for LAMMPS atom types 1, 2, and so on. The number of names given must be equal to the number of atom types in the LAMMPS configuration (not the Allegro model!). 
The given names must be consistent with the names specified in the Allegro training YAML in `chemical_symbol_to_type` or `type_names`.

## Building LAMMPS with this pair style

Please follow the instructions provided in the [pair_allegro README](https://github.com/mir-group/pair_allegro#building-lammps-with-this-pair-style).
