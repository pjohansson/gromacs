#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>
#include <utility> // pair
#include <tuple>   // tie

#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/groupcoord.h"
#include "gromacs/swap/swapcoords.h"
#include "gromacs/topology/mtop_lookup.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
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
 * Contact Line Tracking *
 *************************/

using HistogramCounter = std::vector<double>;

struct Histogram {
    real dx;
    HistogramCounter count; 
};

static void
set_rvec_to_1dpos(rvec r, const real position, const size_t axis)
{
    for (size_t i = 0; i < DIM; i++)
    {
        if (i == axis)
        {
            r[i] = position;
        }
        else
        {
            r[i] = 0.0;
        }
    }
}

static bool
check_if_position_is_within_range_1d(const rvec    r0,
                                     const rvec    r1,
                                     const real    drmax,
                                     const size_t  axis,
                                     const t_pbc  *pbc)
{
    rvec dr;
    pbc_dx(pbc, r0, r1, dr);

    return (fabs(dr[axis]) <= drmax);
}

static void
add_atom_to_histogram_bins(HistogramCounter &count,
                           const rvec        position,
                           const size_t      axis,
                           const real        inv_bin_size)
{
    const auto x = position[axis];
    const auto bin = static_cast<size_t>(floor(x * inv_bin_size)) % count.size();

    count.at(bin) += 1.0;
}

struct ContactLineAxis {
    size_t contact_line;
    size_t height;
    size_t normal;
};

struct ContactLineDef {
    ContactLineDef(const FlowSwap &flow_swap, const matrix box)
    :axis{ flow_swap.axis.zone, flow_swap.axis.swap, flow_swap.axis.normal },
     dhmax{ flow_swap.zone_size[axis.height] / 2.0 }
    {
        height0 = flow_swap.init_coupled_zones.at(0).from[axis.height];
        height1 = flow_swap.init_coupled_zones.at(0).to[axis.height];
        normal_center = flow_swap.init_coupled_zones.at(0).to[axis.normal];

        if (dhmax < 0.0)
        {
            dhmax = box[axis.height][axis.height] / 4.0;
        }
    }

    ContactLineAxis axis;

    real height0, 
         height1,
         dhmax,
         normal_center;
};

static std::pair<HistogramCounter, HistogramCounter>
get_histogram_atom_counts(const ContactLineDef &cl_def,
                          const real            length,
                          const real            bin_size,
                          const size_t          num_bins,
                          const SwapGroup      &group, 
                          const t_pbc          *pbc)
{
    const auto inv_length = 1.0 / length;
    const auto inv_bin_size = 1.0 / bin_size;

    HistogramCounter count0(num_bins, 0.0),
                     count1(num_bins, 0.0);

    rvec r0, r1;
    set_rvec_to_1dpos(r0, cl_def.height0, cl_def.axis.height);
    set_rvec_to_1dpos(r1, cl_def.height1, cl_def.axis.height);

    for (size_t i = 0; i < group.atom_set.numAtomsGlobal(); i++)
    {
        const auto r = group.xs[i];

        if (check_if_position_is_within_range_1d(r, r0, cl_def.dhmax, cl_def.axis.height, pbc))
        {
            add_atom_to_histogram_bins(count0, r, cl_def.axis.contact_line, inv_bin_size);
        }

        if (check_if_position_is_within_range_1d(r, r1, cl_def.dhmax, cl_def.axis.height, pbc))
        {
            add_atom_to_histogram_bins(count1, r, cl_def.axis.contact_line, inv_bin_size);
        }
    }

    return std::pair(count0, count1);
}

static std::pair<Histogram, Histogram>
create_atom_histograms(const ContactLineDef &cl_def,
                       const real            target_resolution,
                       const SwapGroup      &group,
                       const t_pbc          *pbc,
                       const matrix          box)
{
    const auto length = box[cl_def.axis.contact_line][cl_def.axis.contact_line];
    const auto height = box[cl_def.axis.height][cl_def.axis.height];

    const auto num_bins = static_cast<size_t>(roundf(fabs(length / target_resolution)));
    const auto bin_size = length / static_cast<real>(num_bins);

    HistogramCounter count0, count1;
    std::tie(count0, count1) = 
        get_histogram_atom_counts(cl_def, length, bin_size, num_bins, group, pbc);

    const Histogram hist0 { bin_size, count0 },
                    hist1 { bin_size, count1 };

    return std::pair(hist0, hist1);
}

static void
log_histogram(FILE            *fp,
              const Histogram &hist)
{
    real x = 0.5 * hist.dx;

    for (const auto& v : hist.count)
    {
        fprintf(fp, "%g %g\n", x, v);
        x += hist.dx;
    }
}

static void
log_histogram_to_file(const std::string &fn,
                      const Histogram   &hist)
{
    FILE *fp = gmx_ffopen(fn, "w");

    log_histogram(fp, hist);

    gmx_ffclose(fp);
}

static Histogram
smooth_histogram(const Histogram &hist, const size_t num_smooth)
{
    const auto num_bins = hist.count.size();
    const double window_size = 2.0 * static_cast<double>(num_smooth) + 1.0;

    HistogramCounter smooth_count(num_bins, 0.0);

    for (size_t i = 0; i < num_bins; i++)
    {
        for (int j = -static_cast<int>(num_smooth); j <= static_cast<int>(num_smooth); j++)
        {
            auto n = (static_cast<int>(i) + j) % static_cast<int>(num_bins);

            while (n < 0)
            {
                n += num_bins;
            }

            smooth_count.at(static_cast<size_t>(n)) += hist.count.at(i);
        }
    }

    for (auto& v : smooth_count)
    {
        v /= window_size;
    }

    return Histogram {
        hist.dx,
        smooth_count
    };
}

struct HistRegion {
    //! Beginning and end of region, half-open range ([begin, end)).
    //! 
    //! Note that for begin < 0 the region stretches across 
    //! the periodic boundary.
    int begin, end;

    //! Accumulated counts inside region.
    double total_counts;
};

static void 
stitch_front_and_back_regions(std::vector<HistRegion> &regions,
                              const Histogram         &hist)
{
    if (regions.size() < 2)
    {
        return;
    }

    auto& front = regions.front();
    const auto& back = regions.back();

    if ((front.begin == 0) && (back.end == hist.count.size()))
    {
        const auto diff = static_cast<int>(hist.count.size()) - back.begin;

        front.begin = -diff;
        front.total_counts += back.total_counts;

        regions.pop_back();
    }
}

static std::vector<HistRegion>
find_hist_regions(const Histogram &hist,
                  const double     cutoff)
{
    std::vector<HistRegion> regions;

    int i = 0,
        begin = 0;

    double total_count = 0.0;
    bool in_region = false;

    for (const auto& v : hist.count)
    {
        if (v >= cutoff)
        {
            total_count += v;

            if (!in_region)
            {
                in_region = true;
                begin = i;
            }
        }
        // If we are exiting a region we save it and reset the counters
        else if (in_region)
        {
            regions.push_back(HistRegion { begin, i, total_count });

            in_region = false;
            total_count = 0.0;
        }

        i++;
    }

    if (in_region)
    {
        regions.push_back(HistRegion { begin, i, total_count });
    }

    stitch_front_and_back_regions(regions, hist);

    return regions;
}

//! Return the two regions with largest total_counts.
//!
//! Assumes that the given vector has at least two elements. 
//! Will panic if fewer elements are present.
static std::pair<HistRegion, HistRegion>
get_two_largest_hist_regions(std::vector<HistRegion> &regions)
{
    // Reminder: partial_sort ensures that the first N (here 2) elements
    // of the entire range are the "minimum" values, but does not sort 
    // the rest. Here, the "minimum" values are the largest counts so we 
    // supply a custom lambda function that ensures that sorting. 
    std::partial_sort(
        regions.begin(), 
        regions.begin() + 2, 
        regions.end(),
        [](HistRegion &reg0, HistRegion &reg1) {
            return reg0.total_counts > reg1.total_counts;
        }
    );

    return std::pair(regions.at(0), regions.at(1));
}

static size_t 
get_hist_index_pbc(const int        i,
                   const Histogram &hist)
{
    const auto num_bins = static_cast<int>(hist.count.size());

    int n = i % num_bins;
    while (n < 0)
    {
        n += num_bins;
    }

    return static_cast<size_t>(n);
}

static double
get_hist_value(const int        i,
               const Histogram &hist)
{
    return hist.count.at(get_hist_index_pbc(i, hist));
}

static double
get_hist_region_mean_position(const HistRegion &region,
                              const Histogram  &hist)
{
    double mean = 0.0;

    for (auto i = region.begin; i < region.end; i++)
    {
        const auto weight = get_hist_value(i, hist);

        mean += weight * static_cast<double>(i);
    }

    mean /= region.total_counts;

    const auto length = static_cast<double>(hist.count.size());
    while (mean < 0.0)
    {
        mean += length;
    }

    return fmod(mean, length) * hist.dx;
}

static std::pair<real, real>
find_two_peaks(std::vector<HistRegion> &regions,
               const Histogram         &hist)
{
    if (regions.size() < 2)
    {
        return std::pair(-1.0, -1.0);
    }

    HistRegion cl_region0, cl_region1;
    std::tie(cl_region0, cl_region1) = get_two_largest_hist_regions(regions);

    const auto x0 = get_hist_region_mean_position(cl_region0, hist);
    const auto x1 = get_hist_region_mean_position(cl_region1, hist);

    return std::pair(
        static_cast<real>(x0), 
        static_cast<real>(x1)
    );
}

static std::pair<real, real>
find_contact_lines_from_hist(const Histogram &hist)
{
    // To detect the contact lines we make the following assumptions: 
    //  - The histogram has one peak for each contact line (the peak is a normal distribution)
    //  - Each peak is separated from each other by a region with almost no counts
    // 
    // Thus we go through the histogram and detect continuous regions where 
    // the number of counts exceed a cutoff (determined from the maximum histogram value).
    // We then take the two contact lines regions as those with the highest integrated 
    // number of counts, which gets rid of noise. 
    // Finally, we get the average position in the regions.
    constexpr double rel_cutoff = 0.10;
    const auto max_value = *max_element(hist.count.cbegin(), hist.count.cend());
    const auto cutoff = rel_cutoff * max_value;

    auto regions = find_hist_regions(hist, cutoff);

    return find_two_peaks(regions, hist);
}

static gmx::RVec 
create_contact_line_zone_position(const real             contact_line_position,
                                  const real             height,
                                  const ContactLineDef  &cl_def)
{
    gmx::RVec r;

    r[cl_def.axis.contact_line] = contact_line_position;
    r[cl_def.axis.height]       = height;
    r[cl_def.axis.normal]       = cl_def.normal_center;

    return r;
}

static bool
verify_all_contact_lines_detected(const real from0,
                                  const real from1,
                                  const real to0,
                                  const real to1)
{
    return (from0 >= 0.0) && (from1 >= 0.0) && (to0 >= 0.0) && (to1 >= 0.0);
}

static void 
set_bad_contact_line_at_default_zones(real                                &from0,
                                      real                                &from1,
                                      real                                &to0, 
                                      real                                &to1,
                                      const ContactLineAxis               &axis,
                                      const std::vector<CoupledSwapZones> &default_zones)
{
    if (from0 < 0.0) {
        from0 = default_zones.at(0).from[axis.contact_line];
    }

    if (from1 < 0.0) {
        from1 = default_zones.at(1).from[axis.contact_line];
    }

    if (to0 < 0.0) {
        to0 = default_zones.at(0).to[axis.contact_line];
    }

    if (to1 < 0.0) {
        to1 = default_zones.at(1).to[axis.contact_line];
    }
}


static std::vector<CoupledSwapZones>
get_zones_at_contact_line_from_histograms(const Histogram                     &hist0, 
                                          const Histogram                     &hist1,
                                          const ContactLineDef                &cl_def,
                                          const std::vector<CoupledSwapZones> &default_zones)
{
    real pos_from0, pos_from1,
         pos_to0, pos_to1;

    std::tie(pos_from0, pos_to1) = find_contact_lines_from_hist(hist0);
    std::tie(pos_to0, pos_from1) = find_contact_lines_from_hist(hist1);

    if (!verify_all_contact_lines_detected(pos_from0, pos_from1, pos_to0, pos_to1))
    {
        gmx_warning("could not detect 4 contact lines");

        set_bad_contact_line_at_default_zones(
            pos_from0, pos_from1, pos_to0, pos_to1, cl_def.axis, default_zones
        );
    }

    // Order of contact lines: lower left, upper left, upper right, lower right 
    // We are swapping across the lower-upper axis, and the right swap direction 
    // is opposite to the left swap direction. Thus we order the from-to zones 
    // exactly like that, with one from-zone being at height 0 and the other 
    // at height 1, with opposite to-zones.
    const auto from0 = create_contact_line_zone_position(pos_from0, cl_def.height0, cl_def);
    const auto to0 = create_contact_line_zone_position(pos_to0, cl_def.height1, cl_def);
    const auto from1 = create_contact_line_zone_position(pos_from1, cl_def.height1, cl_def);
    const auto to1 = create_contact_line_zone_position(pos_to1, cl_def.height0, cl_def);

    return std::vector<CoupledSwapZones> {
        CoupledSwapZones { from0, to0 },
        CoupledSwapZones { from1, to1 }
    };
}

static std::vector<CoupledSwapZones> 
get_zones_at_contact_lines(const FlowSwap &flow_swap,
                           const matrix    box)
{
    constexpr size_t num_smooth = 10;
    constexpr real histogram_resolution = 0.1;

    const ContactLineDef cl_def(flow_swap, box);

    Histogram hist0, hist1;
    std::tie(hist0, hist1) = create_atom_histograms(
        cl_def, histogram_resolution, flow_swap.swap, flow_swap.pbc, box);

    // log_histogram_to_file("hist0.xvg", hist0);
    // log_histogram_to_file("hist1.xvg", hist1);
    
    if (num_smooth > 0)
    {
        hist0 = smooth_histogram(hist0, num_smooth);
        hist1 = smooth_histogram(hist1, num_smooth);

        // log_histogram_to_file("smooth_hist0.xvg", hist0);
        // log_histogram_to_file("smooth_hist1.xvg", hist1);
    }

    return get_zones_at_contact_line_from_histograms(
        hist0, hist1, cl_def, flow_swap.init_coupled_zones
    );
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
            r[i] = swap_position;
        }
        else if (i == zone_position_axis)
        {
            r[i] = zone_position;
        }
        else
        {
            r[i] = box[i][i] / 2.0;
        }
    }

    return r;
}

static CoupledSwapZones 
create_swap_zones_at_height(const real                 at_height,
                            const std::array<real, 2> &swap_positions,
                            const size_t               swap_axis,
                            const size_t               zone_position_axis,
                            const matrix               box)
{
    const auto from = get_zone_center(
        swap_positions[0], at_height, swap_axis, zone_position_axis, box);
    const auto to = get_zone_center(
        swap_positions[1], at_height, swap_axis, zone_position_axis, box);

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

static std::array<real, 2> 
get_positions_from_reals(const real    from_position,
                         const real    to_position,
                         const bool    is_relative,
                         const size_t  swap_axis,
                         const matrix  box)
{
    std::array<real, 2> positions = { from_position, to_position };
    const auto box_size = box[swap_axis][swap_axis];

    if (is_relative)
    {
        for (auto& v : positions)
        {
            v *= box_size;
        }
    }

    // Apply PBC to position
    for (auto& v : positions)
    {
        v -= floor(v / box_size) * box_size;
    }

    return positions;
}

//! For every zone position, create an array of swap axis positions
//! 
//! The created swap axis positions correspond to the swap method.
//! 
//! Note: Assumes that the given array has exactly two values,
//! which we check in either readir or before calling this.
static std::vector<std::array<real, 2>>
get_swap_positions_array(const t_flowswap *flow_swap,
                         const matrix      box)
{
    real from_position = 0.0,
         to_position   = 0.0;
    bool is_relative = false;

    std::vector<std::array<real, 2>> swap_positions;

    for (size_t n = 0; n < flow_swap->num_positions; n++)
    {
        size_t i, j;

        if (flow_swap->num_swap_zone_values == 2)
        {
            i = 0;
            j = 1;
        }
        else
        {
            GMX_RELEASE_ASSERT(
                flow_swap->num_swap_zone_values >= 2 * flow_swap->num_positions,
                "did not get 2 swap zone positions per zone position");

            i = 2 * n;
            j = i + 1;
        }

        switch (flow_swap->swap_method)
        {
            case eFlowSwapMethod::CenterEdge:
                from_position = 0.0;
                to_position = 0.5;
                break;

            case eFlowSwapMethod::Positions:
            case eFlowSwapMethod::TwoPhaseContactLines:
                from_position = flow_swap->swap_positions[i]; 
                to_position   = flow_swap->swap_positions[j];
                break;
            
            default:
                gmx_fatal(FARGS, "swap: unexpected 'flow-swap-method'");
                break;
        }
        const auto position = get_positions_from_reals(
            from_position, to_position, flow_swap->bRelativeSwapPositions, flow_swap->swap_axis, box
        );

        swap_positions.push_back(position);
    }

    return swap_positions;
}

//! Create from-to zone pairs for each position along the positional axis
static std::vector<CoupledSwapZones> 
create_coupled_swap_zones(const t_flowswap *flow_swap,
                          const matrix      box)
{
    std::vector<CoupledSwapZones> coupled_zones;

    const auto swap_positions = get_swap_positions_array(flow_swap, box);

    for (int i = 0; i < flow_swap->num_positions; i++)
    {
        const auto position_axis = flow_swap->zone_position_axis;
        const auto height = flow_swap->zone_positions[i] * box[position_axis][position_axis];

        coupled_zones.push_back(
            create_swap_zones_at_height(
                height, 
                swap_positions.at(i),
                static_cast<size_t>(flow_swap->swap_axis),
                static_cast<size_t>(flow_swap->zone_position_axis),
                box)
        );
    }

    return coupled_zones;
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

    for (const auto& zones : flow_swap.init_coupled_zones)
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

#ifdef FLOW_SWAP_DEBUG
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
#endif

static void print_single_zone_position(FILE *fp, const gmx::RVec position)
{
    for (size_t i = 0; i < DIM; i++)
    {
        fprintf(fp, "%9.3f ", position[i]);
    }
}

//! Log the zone positions to a file as [time X Y Z X Y Z ...\n]
//!
//! Order is: from, to, from, to order for each defined pair
static void print_zone_positions(FILE                                *fp,
                                 const std::vector<CoupledSwapZones> &coupled_zones,
                                 const double                         time)
{
    fprintf(fp, "%9.3g ", time);

    for (const auto zones : coupled_zones)
    {
        print_single_zone_position(fp, zones.from);
        print_single_zone_position(fp, zones.to);
    }

    fprintf(fp, "\n");
}

//! Log zone position header
static void print_zone_position_header(FILE                                *fp,
                                       const std::vector<CoupledSwapZones> &coupled_zones)
{
    fprintf(fp, "# Swap zone positions during a simulation\n");
    fprintf(fp, "# \n");
    fprintf(fp, "# For each time step, the current time and then the coordinates \n");
    fprintf(fp, "# from each from-to pair is printed in order as \n");
    fprintf(fp, "# 'X0 Y0 Z0 X1 Y1 Z1 X2 Y2 Z2 ...'\n");
    fprintf(fp, "# where (0, 1) is the first (from, to) pair, (2, 3) the second, etc.\n");
    fprintf(fp, "# \n");
    fprintf(fp, "# A total of %lu coupled zone pairs are defined.\n", coupled_zones.size());
    fprintf(fp, "# \n");

    fprintf(fp, "# %-7s ", "time");

    for (size_t n = 0; n < coupled_zones.size(); n++)
    {
        const auto i = 2 * n;
        const auto j = i + 1;

        fprintf(fp, "X%-8ld Y%-8ld Z%-8ld X%-8ld Y%-8ld Z%-8ld ", i, i, i, j, j, j);
    }

    fprintf(fp, "\n");
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
                        const std::vector<CoupledSwapZones> &coupled_zones,
                        const std::vector<std::vector<int>> &swap_mols_in_low_department)
{
    GMX_RELEASE_ASSERT(coupled_zones.size() == swap_mols_in_low_department.size(),
        "Inconsistency.");

    auto zones = coupled_zones.cbegin();
    auto swap_mols = swap_mols_in_low_department.cbegin();

    std::vector<int> swap_atom_inds,
                     fill_atom_inds;
    
    uint16_t num_swaps = 0;

    while (zones != coupled_zones.cend())
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

FlowSwap init_flowswap(gmx::LocalAtomSetManager *atom_sets,
                       t_commrec                *cr,
                       const t_inputrec         *ir,
                       const gmx_mtop_t         *top_global,
                       const SimulationGroups   *groups,
                       const matrix              box,
                       const int                 nfile,
                       const t_filenm            fnm[],
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

    const auto coupled_zones = create_coupled_swap_zones(ir->flow_swap, box);

    const auto swap_group = create_swap_group(0, top_global, atom_sets, groups);
    const auto fill_group = create_swap_group(1, top_global, atom_sets, groups);

    // If we are using domain decompositioning, we must update the local and global
    // atom indices of the sets now that we have added new ones to the manager
    if (DOMAINDECOMP(cr))
    {
        atom_sets->setIndicesInDomainDecomposition(*cr->dd->ga2la);
    }

    // Check whether a file to log swap zone information into 
    // was given and if so prepare it
    FILE *fplog = nullptr;

    if (opt2bSet("-flowswaplog", nfile, fnm))
    {
        const auto fn = opt2fn("-flowswaplog", nfile, fnm);

        fplog = gmx_ffopen(fn, "w");
        print_zone_position_header(fplog, coupled_zones);
    }

    const FlowSwap flow_swap {
        nstswap,
        ir->flow_swap->zone_size,
        coupled_zones,
        ir->flow_swap->swap_method == eFlowSwapMethod::TwoPhaseContactLines,
        swap_group,
        fill_group,
        FlowSwapAxis { 
            static_cast<size_t>(ir->flow_swap->swap_axis), 
            static_cast<size_t>(ir->flow_swap->zone_position_axis)
        },
        static_cast<size_t>(ref_num_mols),
        fplog,
        box
    };

    log_flow_swap_info(flow_swap, mdlog);

    return flow_swap;
}

gmx_bool do_flowswap(FlowSwap         &flow_swap,
                     t_state          *state,
                     double            time,
                     const t_commrec  *cr,
                     const t_inputrec *ir,
                     gmx_wallcycle    *wcycle,
                     const int64_t     step,
                     const gmx_bool    bVerbose)
{
    wallcycle_start(wcycle, ewcSWAP);

    set_pbc(flow_swap.pbc, ir->pbcType, state->box);

    // Collect data to all ranks and do swaps locally. If a swap is made, we will later
    // repartition the dd in the md loop (this is returned as the boolean from this function).
    //
    // Reminder: for each MPI rank we first get the local atom positions for all atoms 
    // in state->x, then collect all the group indices from *all ranks* into group.xs.
    //
    // Here, this is done for the swap group -- later on, we collect for the fill group
    // if needed.
    auto xs_local_ref = gmx::ArrayRef<gmx::RVec>(state->x);
    auto xs_local = as_rvec_array(xs_local_ref.data());
    collect_group_positions_from_ranks(flow_swap.swap, cr, xs_local, state->box);

    auto coupled_zones = flow_swap.init_coupled_zones;

    if (flow_swap.do_track_contact_line)
    {
        coupled_zones = get_zones_at_contact_lines(flow_swap, state->box);
    }

    std::vector<std::vector<int>> mols_in_from_zone_per_coupling;
    for (const auto& zones : coupled_zones)
    {
        mols_in_from_zone_per_coupling.push_back(
            find_molecules_in_compartment(flow_swap.swap, zones.from, flow_swap.max_distance, flow_swap.pbc));
    }

    const bool bNeedSwap = check_for_swap(mols_in_from_zone_per_coupling, flow_swap);
    if (bNeedSwap)
    {
        collect_group_positions_from_ranks(flow_swap.fill, cr, xs_local, state->box);

        const auto num_swaps = do_swap(
            xs_local, flow_swap, coupled_zones, mols_in_from_zone_per_coupling);
        
        if (bVerbose && (num_swaps > 0)) 
        {
            fprintf(stderr, "Performed %d swap%s in step %lu.\n", num_swaps, num_swaps != 1 ? "s" : "", step);
        }
    }

    if (flow_swap.fplog_zone != nullptr)
    {
        print_zone_positions(flow_swap.fplog_zone, coupled_zones, time);
    }

    wallcycle_stop(wcycle, ewcSWAP);
    
    return bNeedSwap;
}