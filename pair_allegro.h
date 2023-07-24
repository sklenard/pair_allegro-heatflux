/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(allegro,PairAllegro)

#else

#ifndef LMP_PAIR_ALLEGRO_H
#define LMP_PAIR_ALLEGRO_H

#include "pair.h"

#include <torch/torch.h>
#include <vector>

namespace LAMMPS_NS {

class PairAllegro : public Pair {
public:
   PairAllegro(class LAMMPS *);
   virtual ~PairAllegro();
   virtual void compute(int, int);
   void settings(int, char **);
   virtual void coeff(int, char **);
   virtual double init_one(int, int);
   virtual void init_style();
   void allocate();

   double cutoff;
   torch::jit::Module model;
   torch::Device device = torch::kCPU;
   std::vector<int> type_mapper;

   int batch_size = -1;

   const torch::Tensor &get_partial_forces() const { return partial_forces_tensor; }
   const torch::Tensor &get_edge_vectors() const { return edge_vectors_tensor; }
   const torch::Tensor &get_edge_index() const { return edge_index_tensor; }
protected:
   int debug_mode = 0;

   torch::Tensor partial_forces_tensor;
   torch::Tensor edge_vectors_tensor;
   torch::Tensor edge_index_tensor;
};

}

#endif
#endif

