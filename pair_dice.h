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

PairStyle(dice,PairDICE)

#else

#ifndef LMP_PAIR_DICE_H
#define LMP_PAIR_DICE_H

#include "pair.h"

#include <torch/torch.h>
#include <vector>

namespace LAMMPS_NS {

class PairDICE : public Pair {
 public:
  PairDICE(class LAMMPS *);
  virtual ~PairDICE();
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


};

}

#endif
#endif

