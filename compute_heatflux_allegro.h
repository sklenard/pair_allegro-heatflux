#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(heat/flux/allegro,ComputeHeatFluxAllegro);
// clang-format on
#else

#ifndef LMP_COMPUTE_HEATFLUX_ALLEGRO_H
#define LMP_COMPUTE_HEATFLUX_ALLEGRO_H

#include "compute.h"
#include "pair_allegro.h"

namespace LAMMPS_NS {

class ComputeHeatFluxAllegro : public Compute {
 public:
  ComputeHeatFluxAllegro(class LAMMPS *, int, char **);
  ~ComputeHeatFluxAllegro();
  void init();
  void compute_vector();

 private:
  char *id_ke, *id_pe;
  class Compute *c_ke, *c_pe;
  PairAllegro *pair;
};

}    // namespace LAMMPS_NS

#endif
#endif
