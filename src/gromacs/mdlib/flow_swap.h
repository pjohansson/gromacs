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
     rmax{rmax} {}

    SwapZone(const gmx::RVec rmin, const gmx::RVec rmax, const gmx::RVec rmin2, const gmx::RVec rmax2)
    :is_split{true},
     rmin{rmin},
     rmax{rmax},
     rmin2{rmin2},
     rmax2{rmax2} {}
    
    void get_center(rvec center) const;

    bool is_split = false;
    gmx::RVec rmin, rmax,   // Minimum and maximum coordinates defining the zone
              rmin2, rmax2; // Coordinates for other zone (if split)
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
    SwapZone min, max;
};

struct FlowSwap {
    /**< Dummy constructor for when no swapping is done */
    FlowSwap(const SwapGroup swap, const SwapGroup fill)
    :swap { swap },
     fill { fill } {}

    /**< Full constructor of object */
    FlowSwap(const uint64_t nstswap, 
             const std::vector<CoupledSwapZones> coupled_zones, 
             const SwapGroup swap, 
             const SwapGroup fill, 
             const size_t ref_num_atoms)
    :do_swap{true},
     nstswap{nstswap},
     coupled_zones{coupled_zones},
     swap{swap},
     fill{fill},
     ref_num_atoms{ref_num_atoms} 
    {
        // The pbc struct will be filled in at every iteration for the current 
        // box, so just allocate the memory
        snew(pbc, 1);
    }

    gmx_bool do_swap = false;
    uint64_t nstswap = 0;
    std::vector<CoupledSwapZones> coupled_zones;

    SwapGroup swap, 
              fill;
    
    size_t ref_num_atoms = 0;
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