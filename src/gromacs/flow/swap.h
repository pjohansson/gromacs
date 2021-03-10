#ifndef FLOW_SWAP_H
#define FLOW_SWAP_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/real.h"

struct SwapZone {
    SwapZone(const gmx::RVec rmin, const gmx::RVec rmax) 
    :rmin{rmin}, 
     rmax{rmax} {
        for (size_t i = 0; i < DIM; i++)
        {
            r0[i] = (rmin[i] + rmax[i]) / 2.0; 
            max_distance[i] = (rmax[i] - rmin[i]) / 2.0;

            if (i == YY)
            {
                max_distance[i] = rmax[i] - rmin[i];
            }
        }
     }

    SwapZone(const gmx::RVec rmin, const gmx::RVec rmax, const gmx::RVec rmin2, const gmx::RVec rmax2)
    :is_split{true},
     rmin{rmin},
     rmax{rmax},
     rmin2{rmin2},
     rmax2{rmax2} {
        r0[XX] = rmin[XX];
        r0[YY] = rmin[YY];
        r0[ZZ] = (rmin[ZZ] + rmax[ZZ]) / 2.0;

        max_distance[XX] = rmax[XX];
        max_distance[YY] = rmax[YY];
        max_distance[ZZ] = (rmax[ZZ] - rmin[ZZ]) / 2.0;
     }
    
    void get_center(rvec center) const;

    bool is_split = false;
    gmx::RVec rmin, rmax,   // Minimum and maximum coordinates defining the zone
              rmin2, rmax2, // Coordinates for other zone (if split)
              r0, max_distance;
};

// A swap group consists of a set of molecules of identical type.
struct SwapGroup {
    /**< Dummy constructor for when no swapping is done */
    SwapGroup(const gmx::LocalAtomSet atom_set)
    :atom_set { atom_set } {}

    /**< Full constructor of object */
    SwapGroup(std::string name, size_t atoms_per_mol, gmx::LocalAtomSet atom_set)
    :name{name},
     atoms_per_mol{atoms_per_mol},
     atom_set{atom_set} 
    {
        snew(xs, atom_set.numAtomsGlobal());
    }

    std::string       name;             /**< Name of group or molecule type */
    size_t            atoms_per_mol;    /**< Number of atoms in each molecule */
    gmx::LocalAtomSet atom_set;         /**< Atom indices of swap group */

    // Preallocate space which all ranks can collect the full position data of the group into
    rvec             *xs = nullptr;     /**< Collective array of group positions (size: atom_set.numAtomsGlobal) */
};

struct CoupledSwapZones {
    gmx::RVec from, to;
};

struct FlowSwap {
    /**< Dummy constructor for when no swapping is done */
    FlowSwap(const SwapGroup swap, const SwapGroup fill)
    :swap { swap },
     fill { fill } {}

    /**< Full constructor of object */
    FlowSwap(const uint64_t nstswap, 
             const gmx::RVec zone_size,
             const std::vector<CoupledSwapZones> coupled_zones, 
             const SwapGroup swap, 
             const SwapGroup fill, 
             const size_t ref_num_atoms,
             const matrix box)
    :do_swap{true},
     nstswap{nstswap},
     zone_size{zone_size},
     coupled_zones{coupled_zones},
     swap{swap},
     fill{fill},
     ref_num_atoms{ref_num_atoms} 
    {
        for (size_t i = 0; i < DIM; i++)
        {
            if (zone_size[i] < 0.0)
            {
                max_distance[i] = box[i][i];
            }
            else 
            {
                max_distance[i] = zone_size[i] / 2.0;
            }
        }

        // The pbc struct will be filled in at every iteration for the current 
        // box, so just allocate the memory
        snew(pbc, 1);
    }

    //! Whether or not to do swapping
    gmx_bool do_swap = false;

    //! How often to swap
    uint64_t nstswap = 0;

    //! Size of swap zones (if < 0: span the entire box)
    gmx::RVec zone_size;

    //! Maximum distance from zone center for an atom to be inside (ie. half the zone size in each direction)
    gmx::RVec max_distance;

    //! Pairs of from-to coupled zone definitions
    std::vector<CoupledSwapZones> coupled_zones;

    //! Index group of molecules to swap and replace (fill) with
    SwapGroup swap, 
              fill;
    
    //! Minimum number of molecules inside the from zone to activate a swap
    size_t ref_num_atoms = 0;

    //! Periodic boundary condition information, updated every frame
    t_pbc *pbc = nullptr;
};

FlowSwap init_flowswap(t_commrec              *cr,
                       gmx::LocalAtomSetManager *atom_sets,
                       const t_inputrec       *ir,
                       const gmx_mtop_t       *top_global,
                       const SimulationGroups *groups,
                       const matrix            box,
                       const gmx::MDLogger    &mdlog);

gmx_bool do_flowswap(FlowSwap         &flow_swap,
                     t_state          *state,
                     const t_commrec  *cr,
                     const t_inputrec *ir,
                     gmx_wallcycle    *wcycle,
                     const int64_t     step,
                     const gmx_bool    bVerbose);

#endif // FLOW_SWAP_H