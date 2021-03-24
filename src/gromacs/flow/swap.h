#ifndef FLOW_SWAP_H
#define FLOW_SWAP_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "gromacs/commandline/filenm.h"
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

struct FlowSwapAxis {
    //! Empty constructor when no swapping is performed.
    FlowSwapAxis()
    :swap{XX},
     zone{ZZ},
     normal{YY} {}
    
    //! Constructor for selected flow swap axis.
    //!
    //! Assumes to swap_axis != zone_axis (checked in readir.cpp).
    FlowSwapAxis(const size_t swap_axis,
                 const size_t zone_axis)
    :swap{swap_axis},
     zone{zone_axis}
    {
        // If swap != zone their sum will always be unique which means that 
        // we can easily get the plane normal by checking it.
        switch (swap + zone) 
        {
            case XX + YY: 
                normal = ZZ;
                break;

            case XX + ZZ:
                normal = YY;
                break;

            case YY + ZZ:
                normal = XX;
                break;
            
            default:
                gmx_fatal(
                    FARGS, 
                    "could not compute normal axis from swap axis %lu and zone axis %lu", 
                    swap, 
                    zone
                );
        }
    }

    //! Axis along which swaps are performed
    size_t swap;

    //! Axis along which zones are defined
    size_t zone;

    //! Axis normal to the swap and zone position axis.
    size_t normal;
};

struct FlowSwap {
    /**< Dummy constructor for when no swapping is done */
    FlowSwap(const SwapGroup swap, const SwapGroup fill)
    :swap { swap },
     fill { fill } {}

    /**< Full constructor of object */
    FlowSwap(const uint64_t  nstswap, 
             const gmx::RVec zone_size,
             const std::vector<CoupledSwapZones> coupled_zones, 
             const gmx_bool do_track_contact_line,
             const SwapGroup swap, 
             const SwapGroup fill, 
             const FlowSwapAxis axis,
             const size_t    ref_num_atoms,
             FILE           *fp,
             const matrix    box)
    :do_swap{true},
     do_track_contact_line{do_track_contact_line},
     nstswap{nstswap},
     zone_size{zone_size},
     init_coupled_zones{coupled_zones},
     swap{swap},
     fill{fill},
     axis{axis},
     ref_num_atoms{ref_num_atoms},
     fplog_zone{fp}
    {
        for (size_t i = 0; i < DIM; i++)
        {
            if (zone_size[i] < 0.0)
            {
                max_distance[i] = box[i][i] / 2.0;
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

    //! Whether or not to track the contact line
    //! 
    //! This is set by `eFlowSwapMethod::TwoPhaseContactLines` which requires
    //! exactly two `CoupledSwapZones` to be defined.
    gmx_bool do_track_contact_line = false;

    //! How often to swap
    uint64_t nstswap = 0;

    //! Size of swap zones (if < 0: span the entire box)
    gmx::RVec zone_size;

    //! Maximum distance from zone center for an atom to be inside (ie. half the zone size in each direction)
    gmx::RVec max_distance;

    //! Pairs of from-to coupled zone definitions
    std::vector<CoupledSwapZones> init_coupled_zones;

    //! Index group of molecules to swap and replace (fill) with
    SwapGroup swap, 
              fill;

    //! Axis definitions for swap zone positioning.
    FlowSwapAxis axis;
    
    //! Minimum number of molecules inside the from zone to activate a swap
    size_t ref_num_atoms = 0;

    //! Periodic boundary condition information, updated every frame
    t_pbc *pbc = nullptr;

    //! File for logging of zone positions
    FILE *fplog_zone = nullptr;
};

FlowSwap init_flowswap(gmx::LocalAtomSetManager *atom_sets,
                       t_commrec                *cr,
                       const t_inputrec         *ir,
                       const gmx_mtop_t         *top_global,
                       const SimulationGroups   *groups,
                       const matrix              box,
                       const int                 nfile,
                       const t_filenm            fnm[],
                       const gmx::MDLogger      &mdlog);

gmx_bool do_flowswap(FlowSwap         &flow_swap,
                     t_state          *state,
                     double            time,
                     const t_commrec  *cr,
                     const t_inputrec *ir,
                     gmx_wallcycle    *wcycle,
                     const int64_t     step,
                     const gmx_bool    bVerbose);

#endif // FLOW_SWAP_H