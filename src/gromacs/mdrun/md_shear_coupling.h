#ifndef MD_SHEAR_COUPLING_H
#define MD_SHEAR_COUPLING_H

#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/ga2la.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/real.h"

enum class Axis {
    X,
    Y,
    Z,
    NR
};

struct ShearVelOpts {
    Axis axis,                    /* Axis along which the areas that are coupled are defined */
         direction;               /* Directional axis of the velocity */ 
        
    ShearCouplStrategy strategy;  /* Strategy for determining the exchange areas */

    size_t num_groups,      /* Number of groups to include atoms from
                               Will be 1 or 2, if 1 both areas will include atoms from
                               the group, but if 2, area 0 and 1 will include atoms only 
                               from groups 0 and 1, respectively                      */

           step;            /* Couple the velocities at multiples of this step */

    real   size,            /* Area size along the axis */
           zedge_adj,       /* Adjust the zmin position by this amount from the edges */
           ref_velocity;    /* Reference velocity to target in the areas              */
                            /* Area 0: velocity = -ref_velocity                       */
                            /* Area 1: velocity = +ref_velocity                       */
};

ShearVelOpts init_shear_velocity_coupling_opts(const t_inputrec       *ir,
                                               const matrix            box,
                                               const SimulationGroups *groups,
                                               const t_commrec        *cr,
                                               const gmx::MDLogger    &mdlog);

void do_shear_velocity_coupling(t_state                *state,
                                const t_mdatoms        *mdatoms,
                                const int64_t           current_step,
                                const ShearVelOpts     &opts,
                                const SimulationGroups *groups,
                                const t_commrec        *cr);

#endif // MD_SHEAR_COUPLING_H