#include <algorithm>
#include <cmath>
#include <iterator>

#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/groupcoord.h"
#include "gromacs/swap/swapcoords.h"
#include "gromacs/topology/mtop_lookup.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"

#include "swap.h"

// #define FLOW_SWAP_DEBUG

#ifdef FLOW_SWAP_DEBUG

#include <chrono>
#include <thread>
using namespace std::chrono_literals;
#define MPI_RANK_SLEEP_DURATION 10ms

#define MPI_RANK_LOOP(commrec, body) \
for (int nodeid = 0; nodeid < commrec->nnodes; nodeid++) \
{ \
    if (nodeid == commrec->nodeid) \
    { \
        body \
    } \
    std::this_thread::sleep_for(MPI_RANK_SLEEP_DURATION); \
    MPI_Barrier(commrec->mpi_comm_mysim); \
}
#else
#define MPI_RANK_LOOP(commrec, body) 

#endif // FLOW_SWAP_DEBUG

#define MPI_RANK_LOOP_VERBOSE(commrec, body) MPI_RANK_LOOP(commrec, \
    fprintf(stderr, "rank %d:\n", commrec->nodeid); \
    body \
)

constexpr SimulationAtomGroupType FLOW_SWAP_GROUP = SimulationAtomGroupType::User2;


/*************************
 * Class implementations *
 *************************/

/**< Construct a dummy `FlowSwap` object
 * 
 * There is no empty constructor for `LocalAtomSet` handles, which means 
 * that we have to construct them for our `SwapGroup` objects.
 */
static FlowSwap init_empty_flow_swap(gmx::LocalAtomSetManager *atom_sets)
{
    const std::vector<int> inds { 0 };

    const SwapGroup swap { atom_sets->add(inds) };
    const SwapGroup fill { atom_sets->add(inds) };

    return FlowSwap { swap, fill };
}


/*************************
 * Group and zone set-up *
 *************************/

// Return the global atom indices of a given group.
static std::vector<int> get_group_inds(const int               num_atoms, 
                                       const int               group_index,
                                       const SimulationGroups *groups)
{
    std::vector<int> inds;

    for (int i = 0; i < num_atoms; ++i)
    {
        if (getGroupType(*groups, FLOW_SWAP_GROUP, i) == group_index)
        {
            inds.push_back(i);
        }
    }

    return inds;
}

static int get_atoms_per_mol_for_group(const std::vector<int> &atom_inds,
                                       const gmx_mtop_t       *mtop)
{
    if (atom_inds.empty())
    {
        return 0;
    }

    const auto atom_index = atom_inds.front();
    int mol_block_index = 0;
    mtopGetMolblockIndex(mtop, atom_index, &mol_block_index, nullptr, nullptr);

    return mtop->moleculeBlockIndices.at(mol_block_index).numAtomsPerMolecule;
}

static gmx::RVec get_zone_center(const real   swap_position,
                                 const real   zone_position, 
                                 const size_t swap_axis,
                                 const size_t zone_position_axis,
                                 const matrix box)
{
    gmx::RVec r = { 0.0, 0.0, 0.0 };

    for (size_t i = 0; i < DIM; i++)
    {
        if (i == swap_axis)
        {
            r[i] = swap_position * box[i][i];
        }
        else if (i == zone_position_axis)
        {
            r[i] = zone_position * box[i][i];
        }
        else
        {
            r[i] = box[i][i] / 2.0;
        }
    }

    return r;
}

static CoupledSwapZones create_swap_zones_at_height(const real   at_height,
                                                    const real   from_swap_position,
                                                    const real   to_swap_position,
                                                    const size_t swap_axis,
                                                    const size_t zone_position_axis,
                                                    const matrix box)
{
    const auto from = get_zone_center(
        from_swap_position, at_height, swap_axis, zone_position_axis, box);
    const auto to = get_zone_center(
        to_swap_position, at_height, swap_axis, zone_position_axis, box);

    return CoupledSwapZones { from, to };
}

static SwapGroup create_swap_group(const int                 group_index,
                                   const gmx_mtop_t         *mtop,
                                   gmx::LocalAtomSetManager *atom_sets,
                                   const SimulationGroups   *groups)
{
    // The group index in User2 is used (later) to check whether atoms belong to the group,
    // but the *global* group index is used here to get the name of the group 
    const int global_group_index = groups->groups[FLOW_SWAP_GROUP].at(group_index);
    const std::string group_name { *groups->groupNames.at(global_group_index) };

    const auto inds = get_group_inds(mtop->natoms, group_index, groups);
    const auto atoms_per_mol = get_atoms_per_mol_for_group(inds, mtop);

    // Add the group indices as a set to be managed by the domain decomposition code
    // and save the handle
    const auto atom_set = atom_sets->add(gmx::ArrayRef<const int>{ inds });

    return SwapGroup {
        group_name,
        static_cast<size_t>(atoms_per_mol),
        atom_set
    };
}


/*********************
 * Logging functions *
 *********************/

static void log_swap_group_info(const SwapGroup     &group,
                                const char          *group_type,
                                const gmx::MDLogger &mdlog)
{
    GMX_LOG(mdlog.warning)
        .appendTextFormatted(
            "  ---\n"
            "  %s group name: %s\n"
            "    Atoms per molecule: %lu\n"
            "    Atoms in group: %lu\n",
            group_type,
            group.name.c_str(),
            group.atoms_per_mol,
            group.atom_set.numAtomsGlobal()
        );
}

static void log_single_zone_info(const gmx::RVec     &position, 
                                 const gmx::RVec     &max_distance,
                                 const gmx::MDLogger &mdlog)
{
    gmx::RVec rmin, rmax;
    rvec_sub(position, max_distance, rmin);
    rvec_add(position, max_distance, rmax);

    GMX_LOG(mdlog.warning)
        .appendTextFormatted(
            "      Begin:           [%g, %g, %g]\n"
            "      End:             [%g, %g, %g]\n",
            rmin[XX], rmin[YY], rmin[ZZ],
            rmax[XX], rmax[YY], rmax[ZZ]
        );
}

static void log_swap_zone_info(const FlowSwap      &flow_swap,
                               const gmx::MDLogger &mdlog)
{
    int i = 1;

    for (const auto& zones : flow_swap.coupled_zones)
    {
        GMX_LOG(mdlog.warning)
            .appendTextFormatted(
                "  ---\n"
                "  Swap zone %d:\n"
                "    Move swap molecules from zone:",
                i
            );
        
        log_single_zone_info(zones.from, flow_swap.max_distance, mdlog);

        GMX_LOG(mdlog.warning)
            .appendText(
                "    To zone:"
            );

        log_single_zone_info(zones.to, flow_swap.max_distance, mdlog);
        
        i++;
    }
}

static void log_flow_swap_info(const FlowSwap      &flow_swap,
                               const gmx::MDLogger &mdlog)
{
    GMX_LOG(mdlog.warning).appendText("");

    GMX_LOG(mdlog.warning)
        .appendTextFormatted(
            "Swapping is turned on.\n"
            "  Frequency:                 %lu\n"
            "  Reference minimum atoms:   %lu\n"
            "  Zone size:                 [%g, %g, %g]\n"
            "  Max dist:                  [%g, %g, %g]\n",
            flow_swap.nstswap,
            flow_swap.ref_num_atoms,
            flow_swap.zone_size[XX],
            flow_swap.zone_size[YY],
            flow_swap.zone_size[ZZ],
            flow_swap.max_distance[XX],
            flow_swap.max_distance[YY],
            flow_swap.max_distance[ZZ] 
        );
    
    log_swap_group_info(flow_swap.swap, "Swap", mdlog);
    log_swap_group_info(flow_swap.fill, "Fill", mdlog);
    log_swap_zone_info(flow_swap, mdlog);
}

static void print_swap_info(FILE *fp,
                            const AtomGroupIndices &swap_mols,
                            const AtomGroupIndices &fill_mols,
                            const double time,
                            const char *comment)
{
    if (fp != nullptr) 
    {
        fprintf(fp, "%12.5g %10lu %10lu %s%s\n", 
            time, 
            swap_mols.size(), 
            fill_mols.size(), 
            strlen(comment) > 0 ? "# " : "", 
            comment);
    }
}


/****************** 
 * Position utils *
 ******************/

static real get_distance_pbc(const rvec x1, const rvec x2, const t_pbc *pbc)
{
    rvec dx;
    pbc_dx(pbc, x1, x2, dx);

    return norm(dx);
}


/*********************************
 * Zone and compartment checking *
 *********************************/

static bool zone_contains_atom(const rvec       x, 
                               const gmx::RVec &position, 
                               const gmx::RVec &max_distance, 
                               const t_pbc     *pbc)
{
    gmx::RVec dx;
    pbc_dx(pbc, x, position, dx);

    for (size_t i = 0; i < DIM; i++)
    {
        if (fabs(dx[i]) > max_distance[i])
        {
            return false;
        }
    }

    return true;
}

static void collect_group_positions_from_ranks(SwapGroup       &group,
                                               const t_commrec *cr,
                                               const rvec       xs_local[],
                                               const matrix     box)
{
    communicate_group_positions(
        cr, group.xs, nullptr, nullptr, FALSE, xs_local, 
        group.atom_set.numAtomsGlobal(), group.atom_set.numAtomsLocal(), 
        group.atom_set.localIndex().data(), group.atom_set.collectiveIndex().data(),
        nullptr, box);
}

// Return the collective indices (in group.xs) of all group atoms in the compartment.
static std::vector<int> find_molecules_in_compartment(const SwapGroup &group,
                                                      const gmx::RVec &position,
                                                      const gmx::RVec &max_distance,
                                                      const t_pbc     *pbc)
{
    std::vector<int> head_atoms_in_compartment;

    for (size_t i = 0; i < group.atom_set.numAtomsGlobal(); i += group.atoms_per_mol)
    {
        if (zone_contains_atom(group.xs[i], position, max_distance, pbc))
        {
            head_atoms_in_compartment.push_back(i);
        }
    }

    return head_atoms_in_compartment;
}

static bool check_compartment_condition(const std::vector<int> &mols_in_compartment,
                                        const FlowSwap         &flow_swap)
{
    return (mols_in_compartment.size() > flow_swap.ref_num_atoms);
}


/**********************
 * Swapping utilities *
 **********************/

static bool check_for_swap(const std::vector<std::vector<int>> &mols_in_low_compartment,
                           const FlowSwap                      &flow_swap)
{
    // Current condition is if any compartment has more swap molecules than a minimum amount
    for (const auto& inds : mols_in_low_compartment)
    {
        if (check_compartment_condition(inds, flow_swap))
        {
            return true;
        }
    }

    return false;
}

static int get_swap_molecule_index(const rvec              xs[],
                                   const std::vector<int> &inds,
                                   const gmx::RVec        &zone_center,
                                   const t_pbc            *pbc)
{
    auto i = inds.cbegin();

    auto best_index = *i;
    auto best_distance = get_distance_pbc(xs[best_index], zone_center, pbc);

    while (i != inds.cend())
    {
        const auto dx = get_distance_pbc(xs[*i], zone_center, pbc);

        if (dx < best_distance)
        {
            best_index = *i;
            best_distance = dx;
        }

        ++i;
    }

    return best_index;
}

static void add_swapped_inds_to_list(const int         head_index,
                                     const size_t      atoms_per_mol,
                                     std::vector<int> &inds)
{
    for (int i = head_index; i < head_index + static_cast<int>(atoms_per_mol); i++)
    {
        inds.push_back(i);
    }
}

template<typename T>
static bool vector_contains_value(const std::vector<T> vs, const T value)
{
    return std::find(vs.cbegin(), vs.cend(), value) != vs.cend();
}


/*********************
 * Molecule swapping *
 *********************/

static void apply_modified_positions(rvec                    xs_local[],
                                     const SwapGroup        &group,
                                     const std::vector<int> &swapped_collective_inds)
{
    const auto local_inds = group.atom_set.localIndex();
    const auto collective_inds = group.atom_set.collectiveIndex();

    for (size_t i = 0; i < group.atom_set.numAtomsLocal(); i++)
    {
        const auto ci = collective_inds.at(i);

        if (vector_contains_value(swapped_collective_inds, ci))
        {
            const auto li = local_inds.at(i);
            copy_rvec(group.xs[ci], xs_local[li]);
            // fprintf(stderr, "swapping collective index %d with local %d\n", ci, local_inds.at(i));
        }
    }
}

// Swap the molecules by swapping their center of mass positions.
static void swap_molecules_com(rvec xs1[],
                               const int i1,
                               const size_t atoms_per_mol_1,
                               rvec xs2[],
                               const int i2,
                               const size_t atoms_per_mol_2,
                               const t_pbc *pbc)
{
    rvec com_swap, com_fill;

    get_molecule_center(&xs1[i1], atoms_per_mol_1, nullptr, com_swap, pbc);
    get_molecule_center(&xs2[i2], atoms_per_mol_2, nullptr, com_fill, pbc);

    MPI_RANK_LOOP_VERBOSE(cr, 
        fprintf(stderr, 
            "swapping atoms %d-%d (at [%f, %f, %f]) with atoms %d-%d (at [%f, %f, %f])\n",
            i1, i1 + atoms_per_mol_1 - 1, com_swap[XX], com_swap[YY], com_swap[ZZ],
            i2, i2 + atoms_per_mol_2 - 1, com_fill[XX], com_fill[YY], com_fill[ZZ]);
    )

    translate_positions(&xs1[i1], atoms_per_mol_1, com_swap, com_fill, pbc);
    translate_positions(&xs2[i2], atoms_per_mol_2, com_fill, com_swap, pbc);
}

// Swap the molecules by exchanging their atom positions.
//
// Assumes that both molecules have the same number of atoms.
static void swap_molecules_atom_pos(rvec xs1[],
                                    const int i1,
                                    rvec xs2[],
                                    const int i2,
                                    const size_t atoms_per_mol)
{
    rvec rbuf;

    for (size_t i = 0; i < atoms_per_mol; i++)
    {
        copy_rvec(xs1[i1 + i], rbuf);
        copy_rvec(xs2[i2 + i], xs1[i1 + i]);
        copy_rvec(rbuf, xs2[i2 + i]);
    }
}
                        
static uint16_t do_swap(rvec                                 xs_local[],
                        const FlowSwap                      &flow_swap,
                        const std::vector<std::vector<int>> &swap_mols_in_low_department)
{
    GMX_RELEASE_ASSERT(flow_swap.coupled_zones.size() == swap_mols_in_low_department.size(),
        "Inconsistency.");

    auto zones = flow_swap.coupled_zones.cbegin();
    auto swap_mols = swap_mols_in_low_department.cbegin();

    std::vector<int> swap_atom_inds,
                     fill_atom_inds;
    
    uint16_t num_swaps = 0;

    while (zones != flow_swap.coupled_zones.cend())
    {
        if (check_compartment_condition(*swap_mols, flow_swap))
        {
            const auto fill_mols = find_molecules_in_compartment(flow_swap.fill, (*zones).to, flow_swap.max_distance, flow_swap.pbc);

            if (!fill_mols.empty()) {
                const auto swap_index = get_swap_molecule_index(
                    flow_swap.swap.xs, *swap_mols, (*zones).from, flow_swap.pbc);

                const auto fill_index = get_swap_molecule_index(
                    flow_swap.fill.xs,  fill_mols, (*zones).to, flow_swap.pbc);

                // swap_molecules_com(
                //     flow_swap.swap.xs, swap_index, flow_swap.swap.atoms_per_mol, 
                //     flow_swap.fill.xs, fill_index, flow_swap.fill.atoms_per_mol, 
                //     flow_swap.pbc);

                swap_molecules_atom_pos(
                    flow_swap.swap.xs, swap_index, 
                    flow_swap.fill.xs, fill_index, 
                    flow_swap.fill.atoms_per_mol);

                add_swapped_inds_to_list(
                    swap_index, flow_swap.swap.atoms_per_mol, swap_atom_inds);

                add_swapped_inds_to_list(
                    fill_index, flow_swap.fill.atoms_per_mol, fill_atom_inds);
                
                num_swaps++;
            }
        }

        ++zones;
        ++swap_mols;
    }

    apply_modified_positions(xs_local, flow_swap.swap, swap_atom_inds);
    apply_modified_positions(xs_local, flow_swap.fill, fill_atom_inds);

    return num_swaps;
}


/******************** 
 * Public functions *
 ********************/

FlowSwap init_flowswap(t_commrec                *cr,
                       gmx::LocalAtomSetManager *atom_sets,
                       const t_inputrec         *ir,
                       const gmx_mtop_t         *top_global,
                       const SimulationGroups   *groups,
                       const matrix              box,
                       const gmx::MDLogger      &mdlog)
{
    if (!ir->flow_swap->do_swap)
    {
        return init_empty_flow_swap(atom_sets);
    }

    if ((PAR(cr)) && !DOMAINDECOMP(cr))
    {
        gmx_fatal(FARGS, "Position swapping is only implemented for domain decomposition!");
    }

    const auto ref_num_mols = ir->flow_swap->ref_num_atoms;
    const auto nstswap = static_cast<uint64_t>(ir->flow_swap->nstswap);

    std::vector<CoupledSwapZones> coupled_zones;

    for (int i = 0; i < ir->flow_swap->num_positions; i++)
    {
        const auto height = ir->flow_swap->zone_positions[i];

        coupled_zones.push_back(
            create_swap_zones_at_height(
                height, 
                0.0, 
                0.5, 
                static_cast<size_t>(ir->flow_swap->swap_axis),
                static_cast<size_t>(ir->flow_swap->zone_position_axis),
                box)
        );
    }

    const auto swap_group = create_swap_group(0, top_global, atom_sets, groups);
    const auto fill_group = create_swap_group(1, top_global, atom_sets, groups);

    // If we are using domain decompositioning, we must update the local and global
    // atom indices of the sets now that we have added new ones to the manager
    if (DOMAINDECOMP(cr))
    {
        atom_sets->setIndicesInDomainDecomposition(*cr->dd->ga2la);
    }

    const FlowSwap flow_swap {
        nstswap,
        ir->flow_swap->zone_size,
        coupled_zones,
        swap_group,
        fill_group,
        static_cast<size_t>(ref_num_mols),
        box
    };

    log_flow_swap_info(flow_swap, mdlog);

    return flow_swap;
}

gmx_bool do_flowswap(FlowSwap         &flow_swap,
                     t_state          *state,
                     const t_commrec  *cr,
                     const t_inputrec *ir,
                     gmx_wallcycle    *wcycle,
                     const int64_t     step,
                     const gmx_bool    bVerbose)
{
    wallcycle_start(wcycle, ewcSWAP);

    set_pbc(flow_swap.pbc, ir->pbcType, state->box);

    // Collect data to all ranks and do swaps locally. If a swap is made, we will later
    // repartition the dd in the md loop.
    auto xs_local_ref = gmx::ArrayRef<gmx::RVec>(state->x);
    auto xs_local = as_rvec_array(xs_local_ref.data());

    collect_group_positions_from_ranks(flow_swap.swap, cr, xs_local, state->box);

    std::vector<std::vector<int>> mols_in_low_compartment;
    for (const auto& zones : flow_swap.coupled_zones)
    {
        mols_in_low_compartment.push_back(
            find_molecules_in_compartment(flow_swap.swap, zones.from, flow_swap.max_distance, flow_swap.pbc));
    }

    const bool bNeedSwap = check_for_swap(mols_in_low_compartment, flow_swap);
    if (bNeedSwap)
    {
        collect_group_positions_from_ranks(flow_swap.fill, cr, xs_local, state->box);

        const auto num_swaps = do_swap(
            xs_local, flow_swap, mols_in_low_compartment);
        
        if (bVerbose && (num_swaps > 0)) 
        {
            fprintf(stderr, "Performed %d swap%s in step %lu.\n", num_swaps, num_swaps != 1 ? "s" : "", step);
        }
    }

    wallcycle_stop(wcycle, ewcSWAP);
    
    return bNeedSwap;
}