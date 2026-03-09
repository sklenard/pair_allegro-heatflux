#include "compute_heatflux_allegro.h"

#include <cstring>
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "force.h"
#include "error.h"
#include "comm.h"

#include <torch/torch.h>
#include <iostream>

using namespace LAMMPS_NS;


/* ---------------------------------------------------------------------- */

ComputeHeatFluxAllegro::ComputeHeatFluxAllegro(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  id_ke(nullptr), id_pe(nullptr)
{
  if (narg != 5) error->all(FLERR,"Illegal compute heat/flux command");

  vector_flag = 1;
  size_vector = 9;
  extvector = 1;

  // store ke/atom, pe/atom, stress/atom IDs used by heat flux computation
  // insure they are valid for these computations
  id_ke = utils::strdup(arg[3]);
  id_pe = utils::strdup(arg[4]);

  pair = dynamic_cast<PairAllegro*>(force->pair);

  int ike = modify->find_compute(id_ke);
  int ipe = modify->find_compute(id_pe);

  if (ike < 0 || ipe < 0)
    error->all(FLERR,"Could not find compute heat/flux compute ID");
  if (strcmp(modify->compute[ike]->style,"ke/atom") != 0)
    error->all(FLERR,"Compute heat/flux compute ID does not compute ke/atom");
  if (modify->compute[ipe]->peatomflag == 0)
    error->all(FLERR,"Compute heat/flux compute ID does not compute pe/atom");
  if (pair == nullptr)
    error->all(FLERR,"Compute heat/flux requires NequIP pair potential");

  vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

ComputeHeatFluxAllegro::~ComputeHeatFluxAllegro()
{
  delete [] id_ke;
  delete [] id_pe;
  delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeHeatFluxAllegro::init()
{
  // error checks

  int ike = modify->find_compute(id_ke);
  int ipe = modify->find_compute(id_pe);
  if (ike < 0 || ipe < 0) 
    error->all(FLERR,"Could not find compute heat/flux compute ID");

  if (comm->ghost_velocity != 1)
    error->all(FLERR,"Velocity have to be communicated to ghost atoms (you should set: \"set comm_modify vel yes\")");

  c_ke = modify->compute[ike];
  c_pe = modify->compute[ipe];
}

/* ---------------------------------------------------------------------- */

void ComputeHeatFluxAllegro::compute_vector()
{
  invoked_vector = update->ntimestep;

  // invoke 3 computes if they haven't been already

  if (!(c_ke->invoked_flag & Compute::INVOKED_PERATOM)) {
    c_ke->compute_peratom();
    c_ke->invoked_flag |= Compute::INVOKED_PERATOM;
  }
  if (!(c_pe->invoked_flag & Compute::INVOKED_PERATOM)) {
    c_pe->compute_peratom();
    c_pe->invoked_flag |= Compute::INVOKED_PERATOM;
  }

  // heat flux vector = jc[3] + jv[3]
  double *ke = c_ke->vector_atom;
  double *pe = c_pe->vector_atom;

  double **x = atom->x;
  double **v = atom->v;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double eng;
  
  double jkin[3];
  double jpot[3];
  double jtot[3];

  //int nedges = atom->edge_index.size(1);

  auto vel_tensor = torch::zeros({nlocal, 3});
  auto edge_index_tensor = pair->get_edge_index();
  auto edge_vectors_tensor = pair->get_edge_vectors();
  auto partial_forces_tensor = pair->get_partial_forces();

  int nedges = edge_index_tensor.size(1);

  auto edge_index = edge_index_tensor.accessor<long,2>();  //atom->edge_index.accessor<long,2>();
  auto edge_vectors = edge_vectors_tensor.accessor<float,2>(); // atom->edge_vectors.accessor<float,2>();
  auto partial_forces = partial_forces_tensor.accessor<float,2>(); // atom->partial_forces.accessor<float,2>();
  auto vel = vel_tensor.accessor<float,2>();

  // collect kinetic term
  jkin[0] = 0;
  jkin[1] = 0;
  jkin[2] = 0;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      eng = pe[i] + ke[i];

      jkin[0] += eng * v[i][0];
      jkin[1] += eng * v[i][1];
      jkin[2] += eng * v[i][2];
    }
  }

  // collect potential term
  jpot[0] = 0;
  jpot[1] = 0;
  jpot[2] = 0;

  for (int idx=0; idx<nedges; idx++) {
    int i = edge_index[0][idx];
    int j = edge_index[1][idx];

    if (mask[i] & groupbit) {
      // require to set "comm_modify vel yes" in order to provide velocities to ghost atoms (i.e. comm->ghost_velocity = 1)
      double d = partial_forces[idx][0] * v[j][0] + partial_forces[idx][1] * v[j][1] + partial_forces[idx][2] * v[j][2];

      jpot[0] -= edge_vectors[idx][0] * d;
      jpot[1] -= edge_vectors[idx][1] * d;
      jpot[2] -= edge_vectors[idx][2] * d;
    }
  }

  jtot[0] = jkin[0] + jpot[0];
  jtot[1] = jkin[1] + jpot[1];
  jtot[2] = jkin[2] + jpot[2];

  // sum across all procs
  // 1st 3 terms are total heat flux
  // 2nd 3 terms are kinetic part
  // 3rd 3 terms are potential part
  double data[9] = { jtot[0], jtot[1], jtot[2], jkin[0], jkin[1], jkin[2], jpot[0], jpot[1], jpot[2] };

  MPI_Allreduce(data, vector, 9, MPI_DOUBLE, MPI_SUM, world);
}
