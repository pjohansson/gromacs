/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#include "gmxpre.h"

#include <algorithm> // find, set_difference
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator> // distance
#include <numeric> // accumulate
#include <set>
#include <utility> // pair
#include <tuple> // tie
#include <vector>

#include "gromacs/commandline/pargs.h"
#include "gromacs/commandline/viewit.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pbcutil/rmpbc.h"
#include "gromacs/random.h"
#include "gromacs/random/seed.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxomp.h"
#include "gromacs/utility/smalloc.h"

#define DEBUG_CONTACTLINE

/****************************************************************************
 * This program analyzes how contact line molecules advance.                *
 * Petter Johansson, Stockholm 2017                                         *
 ****************************************************************************/

enum class Algorithm {
    ContactLine = 1,
    Bottom = 2,
    Hops = 3
};

using IndexVec = std::vector<size_t>;
using IndexSet = std::set<size_t>;

// rvec, but as an array instead of raw pointer
using RvecArray = std::array<real, DIM>;

struct CLConf {
    CLConf(t_pargs    pa[],
           const int  pasize,
           const rvec rmin_in,
           const rvec rmax_in,
           const int  alg_in,
           const t_filenm fnm[],
           const int fnmsize)
        :algorithm{ static_cast<Algorithm>(alg_in) },
         calc_jump_distance{ static_cast<bool>(opt2bSet("-dist", fnmsize, fnm)) },
         cutoff{ opt2parg_real("-co", pasize, pa) },
         cutoff2{ cutoff * cutoff },
         ball_radius{ opt2parg_real("-ball", pasize, pa) },
         precision{ opt2parg_real("-prec", pasize, pa) },
         dx{ opt2parg_real("-dx", pasize, pa) },
         hop_min{ opt2parg_real("-hmin", pasize, pa) },
         hop_min2{ hop_min * hop_min },
         hop_max{ opt2parg_real("-hmax", pasize, pa) },
         hop_max2{ hop_max * hop_max },
         nmin{ opt2parg_int("-nmin", pasize, pa) },
         nmax{ opt2parg_int("-nmax", pasize, pa) }
    {
        copy_rvec(rmin_in, rmin);
        copy_rvec(rmax_in, rmax);
        set_search_space_limits();

        const auto stride_buf = opt2parg_int("-stride", pasize, pa);
        if (stride_buf < 1)
        {
            gmx_fatal(FARGS, "Input stride must be positive.");
        }
        stride = static_cast<size_t>(stride_buf);

        const auto tmpseed = opt2parg_int("-seed", pasize, pa);
        if (tmpseed < 0)
        {
            seed = gmx::makeRandomSeed();
        }
        else
        {
            seed = static_cast<decltype(seed)>(tmpseed);
        }
    }

    void set_box_limits(const matrix box)
    {
        rmin[XX] = (rmin[XX] <= 0.0) ? 0.0 : rmin[XX];
        rmin[YY] = (rmin[YY] <= 0.0) ? 0.0 : rmin[YY];
        rmin[ZZ] = (rmin[ZZ] <= 0.0) ? 0.0 : rmin[ZZ];
        rmax[XX] = (rmax[XX] <= 0.0) ? box[XX][XX] : rmax[XX];
        rmax[YY] = (rmax[YY] <= 0.0) ? box[YY][YY] : rmax[YY];
        rmax[ZZ] = (rmax[ZZ] <= 0.0) ? box[ZZ][ZZ] : rmax[ZZ];
        set_search_space_limits();
    }

    void set_search_space_limits()
    {
        const rvec add_ss = {cutoff, cutoff, cutoff};
        rvec_sub(rmin, add_ss, rmin_ss);
        rvec_add(rmax, add_ss, rmax_ss);
    }

    Algorithm algorithm;
    bool calc_jump_distance;
    real cutoff,
         cutoff2,
         ball_radius,
         precision,
         dx,
         hop_min,
         hop_min2,
         hop_max,
         hop_max2;
    decltype(gmx::makeRandomSeed()) seed;
    int nmin,
        nmax;
    size_t stride;
    rvec rmin,
         rmax,
         rmin_ss,
         rmax_ss;
};

struct Interface {
    Interface(const rvec        *x0,
              const unsigned     num_atoms,
              const IndexVec     bottom_indices,
              const IndexVec     int_indices)
        :bottom{bottom_indices},
         interface{int_indices}
    {
        try {
            positions.reserve(num_atoms);

            for (unsigned i = 0; i < num_atoms; ++i)
            {
                positions.push_back({x0[i][XX], x0[i][YY], x0[i][ZZ]});
            }
        }
        catch (const std::bad_alloc& e) {
            gmx_fatal(FARGS, "Could not allocate memory for all frames (%s).",
                      e.what());
        }
    }

    IndexVec bottom,
             interface;
    IndexSet contact_line;
    std::vector<RvecArray> positions;

    // Vectors for absolute contact line positions from rolling
    // a ball over it and the corresponding closest molecular indices
    // for every position.
    std::vector<real> rolled_xs;
    IndexVec  rolled_inds;
};

template<typename T>
struct Counter {
    Counter ()
        :duration{static_cast<T>(0.0)} {}

    T total() const { return this->duration.count(); }

    void set() { start = std::chrono::system_clock::now(); }
    void stop() { duration += std::chrono::system_clock::now() - start; }

    std::chrono::system_clock::time_point start;
    std::chrono::duration<T> duration;
};

struct Timings {
    Counter<double> interface,
                    bottom,
                    contact_line,
                    loop,
                    traj_total;
};

static void
print_a_timing(const char    *label,
               const Counter<double>  counter,
               const Counter<double>  total)
{
    constexpr size_t MAXLEN = 6;
    const auto percent = 100 * (counter.total() / total.total());
    auto percent_str = new char[MAXLEN];
    snprintf(percent_str, MAXLEN, "%3.1f%%", percent);
    fprintf(stderr, "%-27s%12.3f%11s\n", label, counter.total(), percent_str);
}

static void
print_timings(Timings &timings)
{
    const auto subloop_total = timings.interface.duration
        + timings.bottom.duration + timings.contact_line.duration;
    timings.loop.duration -= subloop_total;
    const auto total = timings.traj_total;

    fprintf(stderr, "\n");
    fprintf(stderr, "Counter                        Time (s)   Of total\n");
    fprintf(stderr, "--------------------------------------------------\n");
    print_a_timing("Collect interface", timings.interface, total);
    print_a_timing("Collect bottom", timings.bottom, total);
    print_a_timing("Collect contact line", timings.contact_line, total);
    print_a_timing("Remainder loop", timings.loop, total);
    fprintf(stderr, "--------------------------------------------------\n");
    fprintf(stderr, "In total the trajectory analysis took %.3f seconds.\n",
            total.total());
}

static bool
is_inside_limits(const rvec x,
                 const rvec rmin,
                 const rvec rmax)
{
    return x[XX] >= rmin[XX] && x[XX] < rmax[XX]
        && x[YY] >= rmin[YY] && x[YY] < rmax[YY]
        && x[ZZ] >= rmin[ZZ] && x[ZZ] < rmax[ZZ];
}

static IndexVec
find_interface_indices(const rvec          *x0,
                       const int           *grpindex,
                       const int            grpsize,
                       const struct CLConf &conf,
                       const t_pbc         *pbc)
{
    // Find indices within search volume
    IndexVec search_space;
    IndexVec candidates;

    for (int i = 0; i < grpsize; ++i)
    {
        const auto n = grpindex[i];
        const auto x1 = x0[n];

        if (is_inside_limits(x1, conf.rmin_ss, conf.rmax_ss))
        {
            search_space.push_back(n);

            if (is_inside_limits(x1, conf.rmin, conf.rmax))
            {
                candidates.push_back(n);
            }
        }
    }

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "Kept %lu (%lu) indices within the search volume. ",
            candidates.size(), search_space.size());
#endif

    IndexVec interface_inds;
    interface_inds.reserve(candidates.size());

    for (const auto& i : candidates)
    {
        const auto x1 = x0[i];
        int count = 0;

        #pragma omp parallel for reduction(+:count)
        for (size_t j = 0; j < search_space.size(); ++j)
        {
            const auto k = search_space[j];
            rvec dx;

            if (i != k)
            {
                const auto x2 = x0[k];
                pbc_dx(pbc, x1, x2, dx);

                if (norm2(dx) <= conf.cutoff2)
                {
                    ++count;
                }
            }
        }

        if (count >= conf.nmin && count <= conf.nmax)
        {
            interface_inds.push_back(i);
        }
    }

    interface_inds.shrink_to_fit();

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "Found %lu interface ", interface_inds.size());
#endif

    return interface_inds;
}

static std::vector<IndexVec>
slice_system_along_dir(const rvec     *x0,
                       const IndexVec &interface,
                       const CLConf   &conf,
                       const int       dir)
{
    // Identify the floor by slicing the system and finding the lowest peak
    const unsigned num_slices = static_cast<unsigned>(ceil((conf.rmax[dir] - conf.rmin[dir]) / conf.precision));
    const real final_slice_precision = (conf.rmax[dir] - conf.rmin[dir]) / num_slices;

    std::vector<IndexVec> slice_indices (num_slices);
    for (const auto i : interface)
    {
        const auto x = x0[i][dir];
        const auto slice = static_cast<size_t>(
            (x - conf.rmin[dir]) / final_slice_precision
        );
        slice_indices.at(slice).push_back(i);
    }

    return slice_indices;
}

static IndexVec
find_bottom_layer_indices(const rvec     *x0,
                          const IndexVec &interface,
                          const CLConf   &conf)
{
    const auto slice_indices = slice_system_along_dir(x0, interface, conf, ZZ);

    // Track the maximum count.
    // When a decrease is found the peak was in the previous slice.
    int prev_slice = -1;
    unsigned max_count = 0;

    for (const auto& indices : slice_indices)
    {
        const auto count = indices.size();
        if (count < max_count)
        {
            break;
        }

        max_count = count;
        ++prev_slice;
    }

    return slice_indices.at(prev_slice);
}

using Coord2 = std::array<real, 2>;

static int
get_random_element_from_set(gmx::DefaultRandomEngine  &rng,
                            const std::set<size_t>    &list)
{
    if (list.size() == 0)
    {
        return -1;
    }

    gmx::UniformIntDistribution<> dist(0, list.size() - 1);
    const auto i = dist(rng);

    auto it = list.cbegin();
    std::advance(it, i);

    return *it;
}

static Coord2
gen_coordinate_in_box(gmx::DefaultRandomEngine &rng,
                      const Coord2             &xlim,
                      const Coord2             &ylim
                     )
{
    gmx::UniformRealDistribution<real> xdist(xlim[0], xlim[1]);
    gmx::UniformRealDistribution<real> ydist(ylim[0], ylim[1]);

    const auto x = xdist(rng);
    const auto y = ydist(rng);

    return Coord2 { x, y };
}

static Coord2
gen_coordinate_around_x(gmx::DefaultRandomEngine &rng,
                        const Coord2             &x0,
                        const real                rmin,
                        const Coord2             &xlim,
                        const Coord2             &ylim
                       )
{
    const auto rlim = Coord2 { -2 * rmin, 2 * rmin };
    while (true)
    {
        const auto candidate = gen_coordinate_in_box(rng, rlim, rlim);
        const auto dr = sqrt(pow(candidate[0], 2) + pow(candidate[1], 2));

        if (dr >= rmin && dr <= 2.0 * rmin)
        {
            const auto x = candidate[0] + x0[0];
            const auto y = candidate[1] + x0[1];

            if (x >= xlim[0] && x < xlim[1] && y >= ylim[0] && y < ylim[1])
            {
                return Coord2 { x, y };
            }
        }
    }
}

// Class which contains the required collections to construct a Poisson disk.
struct PoissonDiskGrid {
    // Initialize the variables and the `grid` with all-empty values (-1).
    PoissonDiskGrid(const real rmin,
                    const rvec rmin_vec,
                    const rvec rmax_vec)
    :rmin { rmin },
     origin { rmin_vec[YY], rmin_vec[ZZ] },
     size { rmax_vec[YY] - rmin_vec[YY], rmax_vec[ZZ] - rmin_vec[ZZ] }
    {
        const real cell_size = rmin / sqrt(2.0);
        ny = ceil(size[0] / cell_size);
        nz = ceil(size[1] / cell_size);
        grid = std::vector<int>(ny * nz, -1);
    }

    void add_coordinate(const Coord2 &x);
    size_t coord_to_index(const Coord2 &x) const;
    size_t index_from_2d(const size_t iy, const size_t iz) const;
    std::pair<size_t, size_t> index_to_2d(const size_t i) const;
    bool try_add_coordinate(const Coord2 &x);

    real rmin;
    size_t ny, nz;
    Coord2 origin, size;

    // List of coordinates that have been added.
    std::vector<Coord2> coordinates;

    // Background 2D grid with indices to each cell's index to its coordinate // in the `coordinates` vector. A cell with no coordinate has the value -1.
    std::vector<int> grid;

    // Indices of currently active samples in the `coordinates` vector.
    std::set<size_t> active;
};

// Add a coordinate to the `coordinates` vector and its index to the
// background grid and active list.
void PoissonDiskGrid::add_coordinate(const Coord2 &x)
{
    const auto i = coord_to_index(x);
    const auto n = coordinates.size();

    coordinates.push_back(x);
    grid.at(i) = n;
    active.insert(n);
}

// Return the index to the cell of the `grid` of an input coordinate.
size_t PoissonDiskGrid::coord_to_index(const Coord2 &x) const
{
    const auto dy = static_cast<real>(size[0] / ny);
    const auto dz = static_cast<real>(size[1] / nz);

    const auto iy = static_cast<size_t>(floor((x[0] - origin[0]) / dy));
    const auto iz = static_cast<size_t>(floor((x[1] - origin[1]) / dz));

    return index_from_2d(iy, iz);
}

// Return the `grid` index from its corresponding 2d indices.
size_t PoissonDiskGrid::index_from_2d(const size_t iy, const size_t iz) const
{
    if (iy >= ny || iz >= nz)
    {
        if (iy >= ny)
        {
            gmx_fatal(FARGS,
                      "Bad input index `iy` %lu (larger than maximum %lu)",
                      iy, ny - 1
            );
        }
        if (iz >= nz)
        {
            gmx_fatal(FARGS,
                      "Bad input index `iz` %lu (larger than maximum %lu)",
                      iz, nz - 1
            );
        }
    }

    return iy * nz + iz;
}

// Return a pair of the corresponding 2d indices of a 1d `grid` index.
std::pair<size_t, size_t> PoissonDiskGrid::index_to_2d(const size_t i) const
{
    const auto iz = i % nz;
    const auto iy = static_cast<size_t>(floor(i / nz));

    return std::pair<size_t, size_t> { iy, iz };
}

// Try to add a coordinate to the system: Do so only if there are no previously
// added `coordinates` within the minimum distance `rmin` of it.
//
// Return `true` if the coordinate was successfully added, `false` if not.
bool PoissonDiskGrid::try_add_coordinate(const Coord2 &x)
{
    const auto i = coord_to_index(x);
    size_t iy, iz;
    std::tie(iy, iz) = index_to_2d(i);

    // Check only a fixed area of the `grid`: since the cell size is bounded
    // at rmin / sqrt(2) we only have to check cells within 2 in each direction.
    const auto imin = static_cast<int>(iy) - 2 >= 0 ? iy - 2 : 0;
    const auto jmin = static_cast<int>(iz) - 2 >= 0 ? iz - 2 : 0;
    const auto imax = iy + 2 <= ny - 1 ? iy + 2 : ny - 1;
    const auto jmax = iz + 2 <= nz - 1 ? iz + 2 : nz - 1;

    for (auto i = imin; i <= imax; ++i)
    {
        for (auto j = jmin; j <= jmax; ++j)
        {
            const auto k = index_from_2d(i, j);
            const auto n = grid.at(k);

            if (n > -1)
            {
                const auto x1 = coordinates[n];
                const auto dr = sqrt(
                    pow(x1[0] - x[0], 2) + pow(x1[1] - x[1], 2)
                );

                if (dr < rmin)
                {
                    return false;
                }
            }
        }
    }

    add_coordinate(x);

    return true;
}

// When sampling the outer interface, use a poisson disk distribution
// to generate points in the yz plane along which to sample the outermost
// molecules. This method is efficient and should result in uniform sampling.
//
// See: http://www.cs.ubc.ca/%7Erbridson/docs/bridson-siggraph07-poissondisk.pdf
static std::vector<Coord2>
sample_points_poisson_disk(const CLConf &conf)
{
    constexpr unsigned kmax = 30; // suggested magic variable for the sampling

    static gmx::DefaultRandomEngine rng(conf.seed);

    const real rmin = conf.ball_radius / sqrt(2);
    PoissonDiskGrid poisson_disk (rmin, conf.rmin, conf.rmax);

    const auto ylim = Coord2 { conf.rmin[YY], conf.rmax[YY] };
    const auto zlim = Coord2 { conf.rmin[ZZ], conf.rmax[ZZ] };
    const auto x0 = gen_coordinate_in_box(rng, ylim, zlim);

    poisson_disk.add_coordinate(x0);

    while (poisson_disk.active.size() > 0)
    {
        const auto i = get_random_element_from_set(rng, poisson_disk.active);
        const auto x1 = poisson_disk.coordinates[i];

        bool found_new_sample = false;

        for (unsigned k = 0; k < kmax; ++k)
        {
            const auto x2 = gen_coordinate_around_x(rng, x1, rmin, ylim, zlim);

            if (poisson_disk.try_add_coordinate(x2))
            {
                found_new_sample = true;
                break;
            }
        }

        if (!found_new_sample)
        {
            poisson_disk.active.erase(i);
        }
    }

    return poisson_disk.coordinates;
}

static IndexSet
get_outer_interface(const Interface   &interface,
                    const rvec        *x0,
                    const CLConf      &conf)
{
    // Roll a ball along the interface to detect only the outer interface`
    const real r2 = conf.ball_radius * conf.ball_radius;
    const real dx_ball = conf.ball_radius / 10.0;

    IndexSet outer_interface {};

    const auto sample_coords = sample_points_poisson_disk(conf);

    for (const auto coord : sample_coords)
    {
        const auto y = coord[0];
        const auto z = coord[1];

        auto x = conf.rmax[XX];

        while (x >= conf.rmin[XX])
        {
            // For this new ball position, find all candidates that
            // are within the radius and *if any* are found,
            // take the closest one as the contact line molecule
            // and break the loop.

            IndexVec candidates;
            std::vector<real> dr2s;

            for (const auto index : interface.interface)
            {
                const RvecArray x1 {x, y, z};
                const auto dr2 = distance2(x1.data(), x0[index]);

                if (dr2 <= r2)
                {
                    candidates.push_back(index);
                    dr2s.push_back(dr2);
                }
                else
                {
                    const auto dr2 = distance2(x1.data(), x0[index + 1]);

                    if (dr2 <= r2)
                    {
                        candidates.push_back(index);
                        dr2s.push_back(dr2);
                    }
                    else
                    {
                        const auto dr2 = distance2(x1.data(), x0[index + 2]);

                        if (dr2 <= r2)
                        {
                            candidates.push_back(index);
                            dr2s.push_back(dr2);
                        }
                    }
                }
            }

            if (candidates.size() > 0)
            {
                const auto it = std::min_element(dr2s.cbegin(), dr2s.cend());
                const auto dn = std::distance(dr2s.cbegin(), it);
                const auto index = candidates[dn];

                outer_interface.insert(index);

                break;
            }

            x -= dx_ball;
        }
    }

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "\b Also found %lu outer interface indices.\n",
            outer_interface.size());
#endif

    return outer_interface;
}


static void
add_contact_line(Interface         &interface,
                 const rvec        *x0,
                 const IndexVec    &bottom,
                 const CLConf      &conf)
{
    // Roll a ball along the bottom of the interface to detect the contact line
    const real r2 = conf.ball_radius * conf.ball_radius;
    const real dx_ball = conf.ball_radius / 10.0;
    const real dy_ball = conf.ball_radius / 5.0;

    real y = conf.rmin[YY];

    while (y <= conf.rmax[YY])
    {
        // Just stupidly move towards the contact line (from the max)
        // until some molecule is found within the ball radius.

        real x = conf.rmax[XX];

        while (x >= conf.rmin[XX])
        {
            // For this new ball position, find all candidates that
            // are within the radius and *if any* are found,
            // take the closest one as the contact line molecule
            // and break the loop.

            IndexVec candidates;
            std::vector<real> dr2s;

            for (const auto index : bottom)
            {
                const auto dx = x - x0[index][XX];
                const auto dy = y - x0[index][YY];
                const auto dr2 = dx * dx + dy * dy;

                if (dr2 <= r2)
                {
                    candidates.push_back(index);
                    dr2s.push_back(dr2);
                }
            }

            if (candidates.size() > 0)
            {
                const auto it = std::min_element(dr2s.cbegin(), dr2s.cend());
                const auto dn = std::distance(dr2s.cbegin(), it);
                const auto index = candidates[dn];

                interface.rolled_xs.push_back(x);
                interface.rolled_inds.push_back(index);
                interface.contact_line.insert(index);

                break;
            }

            x -= dx_ball;
        }

        y += dy_ball;
    }

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "and %lu contact line atoms.\n",
            interface.contact_line.size());
#endif
}

template<typename T>
static IndexSet
find_shared_indices(const IndexSet &current,
                    const T        &prev)
{
    IndexSet shared_indices;

    for (const auto i : current)
    {
        if (find(prev.cbegin(), prev.cend(), i) != prev.cend())
        {
            shared_indices.insert(i);
        }
    }

    return shared_indices;
}

static IndexSet
contact_line_advancements(const Interface &current,
                          const Interface &previous,
                          const CLConf    &conf)
{
    IndexSet inds_advanced;

    if (current.rolled_xs.size() != previous.rolled_xs.size())
    {
        gmx_fatal(FARGS,
                  "Current and previous interface are of different size. "
                  "This should not happen.");
    }

    // The vectors contain X values and have identical order and spacing
    // of Y values, thus we can just iterate through them and compare.
    auto xcur = current.rolled_xs.cbegin();
    auto xprev = previous.rolled_xs.cbegin();
    auto icur = current.rolled_inds.cbegin();

    while (xcur != current.rolled_xs.cend())
    {
        auto dx = *xcur - *xprev;

        if (dx >= conf.dx)
        {
            // Assert that this delta is not due to an atom sidestepping
            // oh-so-slightly
            const auto x1 = current.positions[*icur];
            const auto x2 = previous.positions[*icur];
            const auto dr2 = distance2(x1.data(), x2.data());

            if (dr2 >= conf.hop_min2)
            {
                inds_advanced.insert(*icur);
            }
#ifdef DEBUG_CONTACTLINE
            else
            {
                fprintf(stderr, "Discarded a hop (dr: %.2f)\n", sqrt(dr2));
            }
#endif
        }

        ++xcur;
        ++xprev;
        ++icur;
    }

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "Atoms which advanced the contact line:");
    for (const auto i : inds_advanced)
    {
        fprintf(stderr, " %lu", i);
    }
    fprintf(stderr, "\n");
#endif

    return inds_advanced;
}

static IndexSet
at_previous_contact_line(const IndexSet  &indices,
                         const Interface &current,
                         const Interface &previous,
                         const CLConf    &conf)
{
    IndexSet prev_contact_line;

    switch (conf.algorithm)
    {
        case Algorithm::ContactLine:
            prev_contact_line = find_shared_indices(indices, previous.contact_line);
            break;

        case Algorithm::Bottom:
            prev_contact_line = find_shared_indices(indices, previous.bottom);
            break;

        case Algorithm::Hops:
            {
                const auto shared = find_shared_indices(indices, previous.bottom);

                for (const auto index : shared)
                {
                    const auto& x1 = current.positions[index];
                    const auto& x2 = previous.positions[index];

                    const auto dr2 = distance2(x1.data(), x2.data());

                    if ((dr2 >= conf.hop_min2) && (dr2 <= conf.hop_max2))
                    {
                        prev_contact_line.insert(index);
                    }
                }
            }
            break;

        default:
            gmx_fatal(FARGS, "Selected algorithm is not implemented.");
            break;
    }

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "Of which were at the previous contact line:");
    for (auto i : prev_contact_line)
    {
        fprintf(stderr, " %lu", i);
    }
    fprintf(stderr, "\n");
#endif

    return prev_contact_line;
}

static std::vector<real>
calculate_fractions(const IndexVec &from_previous,
                    const IndexVec &num_advanced)
{
    auto from = from_previous.cbegin();
    auto total = num_advanced.cbegin();

    real sum_total = 0.0,
         sum_previous = 0.0,
         sum_fractions = 0.0;

    std::vector<real> fractions;

    while (from != from_previous.cend())
    {
        const auto tot = static_cast<real>(*total++);
        const auto prev = static_cast<real>(*from++);

        sum_total += tot;
        sum_previous += prev;

        if (tot > 0.0)
        {
            fractions.push_back(prev / tot);
            sum_fractions += fractions.back();
        }
        else
        {
            fractions.push_back(0.0);
        }
    }

    const auto fraction = sum_previous / sum_total;
    const real mean = sum_fractions / static_cast<real>(fractions.size());
    real var = 0.0;

    for (const auto f : fractions)
    {
        var += (f - mean) * (f - mean);
    }
    var /= static_cast<real>(fractions.size());
    const auto std = sqrt(var);
    const auto err = std / sqrt(fractions.size());

    fprintf(stdout,
            "Fraction of contact line molecules that came "
            "from the previous contact line:\n%.3f +/- %.3f\n",
            fraction, err);

    return fractions;
}

/*
void
analyze_replacement(const IndexSet    &advanced_indices,
                    const Interface   &current,
                    const Interface   &previous,
                    const CLConf      &conf)
{
    for (auto index : advanced_indices)
    {
        // This is the previous position of an atom which
        // advanced the contact line. Find which atom in
        // the *current* bottom layer that is closest to
        // this position. Then find *that* atom's previous
        // position.
        const auto x0 = previous.positions[index];

        // If there are no atoms in the current bottom layer
        // we cannot do anything.!
        auto other = current.bottom.cbegin();
        if (other == current.bottom.cend())
        {
            gmx_fatal(FARGS,
                      "Inconsistent behaviour: No bottom layer of atoms is "
                      "present, but some contact line atoms were found. "
                      "This should not happen!");
        }

        auto x1 = current.positions[*other];
        //auto imin = *other;
        auto rmin2 = distance2(x1.data(), x0.data());

        while (other != current.bottom.cend())
        {
            if (*other != index)
            {
                x1 = current.positions[*other];
                const auto dr2 = distance2(x1.data(), x0.data());

                if (dr2 < rmin2)
                {
                    //imin = *other;
                    rmin2 = dr2;
                }
            }

            ++other;
        }
    }
}
*/

struct CLData {
    std::vector<real> times,
                      fractions,
                      distances;
};

template<typename T>
static int find_the_closest_neighbour(const RvecArray              &x0,
                                      const T                      &to,
                                      const std::vector<RvecArray> &positions)
{
    auto iter_to = to.cbegin();
    auto x1 = positions[*iter_to];

    auto index_min = *iter_to;
    auto min_dist = distance2(x1.data(), x0.data());

    while (++iter_to != to.cend())
    {
        x1 = positions[*iter_to];
        const auto dist = distance2(x1.data(), x0.data());

        if (dist < min_dist)
        {
            min_dist = dist;
            index_min = *iter_to;
        }
    }

    return index_min;
}

template<typename T1, typename T2>
static IndexSet find_second_layer(const T1                     &from,
                                  const T2                     &to,
                                  const std::vector<RvecArray> &positions)
{
    IndexSet second_layer;

    for (const auto i : from)
    {
        const auto x0 = positions[i];
        const auto index = find_the_closest_neighbour(x0, to, positions);
        second_layer.insert(index);
    }

    return second_layer;
}

template<typename T1, typename T2>
static IndexVec find_closest_neighbours(const T1                     &from,
                                        const T2                     &to,
                                        const std::vector<RvecArray> &positions)
{
    IndexVec neighbours;

    for (const auto i : from)
    {
        const auto x0 = positions[i];
        const auto index = find_the_closest_neighbour(x0, to, positions);
        neighbours.push_back(index);
    }

    return neighbours;
}

using DistAndHeight = std::pair<real, real>;

static DistAndHeight
calc_displacement_distance(const Interface &interface)
{
    // Find the set of indices which are at the interface
    // but not in the bottom layer
    IndexVec not_bottom_layer;
    set_difference(
        interface.interface.cbegin(), interface.interface.cend(),
        interface.bottom.cbegin(), interface.bottom.cend(),
        inserter(not_bottom_layer, not_bottom_layer.begin())
    );

    // Find all indices in the second layer
    const auto second_layer = find_second_layer(
        interface.contact_line, not_bottom_layer, interface.positions
    );

    // Find their closest neighbours among the contact line
    const auto closest_neighbours = find_closest_neighbours(
        second_layer, interface.contact_line, interface.positions
    );

    // Then calculate the distance along x to them
    std::vector<real> distances;
    std::vector<real> heights;
    auto from = second_layer.cbegin();
    auto to = closest_neighbours.cbegin();

    while (from != second_layer.cend())
    {
        const auto x0 = interface.positions[*from++];
        const auto x1 = interface.positions[*to++];

        // Calculate the movement in the xy plane as the distance
        const auto dx = x1[XX] - x0[XX];
        const auto dy = x1[YY] - x0[YY];
        const int sign = dx > 0.0 ? 1 : -1;
        const auto dr = sign * sqrt(pow(dx, 2) + pow(dy, 2));
        distances.push_back(dr);

        // Calculate the height difference
        const auto dz = x0[ZZ] - x1[ZZ];
        heights.push_back(dz);
    }

    // Calculate the mean and standard error
    const auto mean_dr = accumulate(distances.cbegin(), distances.cend(), 0.0)
                        / distances.size();
    const auto mean_dz = accumulate(heights.cbegin(), heights.cend(), 0.0)
                        / heights.size();

    return DistAndHeight {
        static_cast<real>(mean_dr), static_cast<real>(mean_dz)
    };
}

// Calculate the arithmetic mean and variance of all values in a vector.
// Excepts if less than two values are present in the vector.
template <typename T>
const static std::pair<T, T>
calc_mean_and_var(const std::vector<T> &values)
{
    const auto mean = std::accumulate(values.cbegin(), values.cend(), 0.0)
                        / values.size();
    const auto var = std::accumulate(values.cbegin(), values.cend(), 0.0,
                        [=] (const T& acc, const T& value) {
                            return acc + pow(value - mean, 2);
                        }
                    ) / (values.size() - 1);

    return std::pair<T, T> { mean, var };
}

// Calculate the mean distance and height between the contact line
// and their closest second layer molecules. Return the distances
// to use for plotting a distance figure.
static std::vector<real>
calculate_mean_distance(const std::vector<DistAndHeight> &values)
{
    std::vector<real> dists;
    std::vector<real> heights;
    real dr, dz;
    for (const auto v : values)

    {
        std::tie(dr, dz) = v;
        dists.push_back(dr);
        heights.push_back(dz);
    }

    real mean_dr, var_dr,
         mean_dz, var_dz;
    std::tie(mean_dr, var_dr) = calc_mean_and_var(dists);
    std::tie(mean_dz, var_dz) = calc_mean_and_var(heights);

    const auto err_dr = sqrt(var_dr) / (values.size() - 1);
    const auto err_dz = sqrt(var_dz) / (values.size() - 1);

    fprintf(stdout,
            "Average distance for second layer molecules to jump:\n"
            "%.3f +/- %.3f (std: %.3f)\n",
            mean_dr, err_dr, sqrt(var_dr));
    fprintf(stdout,
            "Average height above the first layer of the second layer molecules:\n"
            "%.3f +/- %.3f (std: %.3f)\n",
            mean_dz, err_dz, sqrt(var_dz));

    return dists;
}

static void
write_interface_coords(const char          *fn,
                       const Interface     &interface,
                       const rvec          *x0,
                       const struct CLConf &conf)
{
    const auto outer_interface = get_outer_interface(interface, x0, conf);

    // In turn, write:
    // 1. The number of coordinate points as an unsigned int.
    // 2. Each coordinate as 1 rvec.

    auto fp = gmx_ffopen(fn, "ab");

    const auto num_coords = static_cast<uint64_t>(outer_interface.size());
    fwrite(&num_coords, sizeof(num_coords), 1, fp);

    for (const auto index : outer_interface)
    {
        fwrite(x0[index], sizeof(x0[0]), 1, fp);
    }

    gmx_ffclose(fp);
}

static CLData
collect_contact_line_advancement(const char             *fn_traj,
                                 int                    *grpindex,
                                 int                     grpsize,
                                 struct CLConf          &conf,
                                 const char             *fnout_interface,
                                 const t_topology       *top,
                                 const int               ePBC,
                                 const gmx_output_env_t *oenv)
{
    unsigned num_atoms;
    rvec *x0;
    matrix box;
    t_trxstatus *status;
    real t;

    if ((num_atoms = read_first_x(oenv, &status, fn_traj, &t, &x0, box)) == 0)
    {
        gmx_fatal(FARGS, "Could not read coordinates from statusfile\n");
    }


    auto gpbc = gmx_rmpbc_init(&top->idef, ePBC, top->atoms.nr);
    auto pbc = new t_pbc;
    set_pbc(pbc, ePBC, box);
    conf.set_box_limits(box);

    // At each frame, count and save the number of atoms (ie. indices)
    // who advanced the contact line and the number of those who
    // came from the previous contact line. Keep them in a vector
    // to analyze the entire set at the end of the trajectory analysis.
    IndexVec num_advanced;
    IndexVec num_from_previous;
    std::vector<real> times;

    // Also save the distance from second layer atoms to their closest
    // contact line atom
    std::vector<DistAndHeight> distances;

    // To compare against previous frames when calculating which
    // indices advanced the contact line we need to save this
    // information. A deque gives us a window of the last-to-first data.
    std::deque<Interface> interfaces;

#ifdef DEBUG_CONTACTLINE
    constexpr size_t MAXLEN = 80;
    auto debug_filename = new char[MAXLEN];
    auto debug_title = new char[MAXLEN];
#endif

    // If the interface should be saved, open a file for it.
    // Start by writing the size of each vector element (`real`).
    FILE *fp_interface { nullptr };
    const bool save_interface_coords = fnout_interface != NULL;

    if (save_interface_coords)
    {
        fp_interface = gmx_ffopen(fnout_interface, "wb");
        constexpr auto sizeof_real { sizeof(real) };
        fwrite(&sizeof_real, sizeof(size_t), 1, fp_interface);
        gmx_ffclose(fp_interface);
    }

    Timings timings;
    timings.traj_total.set();

    do
    {
        try
        {
            times.push_back(t);

            timings.loop.set();
            gmx_rmpbc(gpbc, num_atoms, box, x0);

            timings.interface.set();
            const auto interface = find_interface_indices(
                x0, grpindex, grpsize, conf, pbc);
            timings.interface.stop();

            timings.bottom.set();
            const auto bottom = find_bottom_layer_indices(x0, interface, conf);
            timings.bottom.stop();

            Interface current {x0, num_atoms, bottom, interface};

            timings.contact_line.set();
            add_contact_line(current, x0, bottom, conf);
            timings.contact_line.stop();

            interfaces.push_back(current);

            if (conf.calc_jump_distance)
            {
                distances.push_back(calc_displacement_distance(current));
            }

            if (save_interface_coords)
            {
                write_interface_coords(fnout_interface, current, x0, conf);
            }

            if (interfaces.size() > conf.stride)
            {
                const auto& previous = interfaces.front();
                const auto inds_advanced = contact_line_advancements(
                    current, previous, conf);
                const auto from_previous = at_previous_contact_line(
                    inds_advanced, current, previous, conf);

                // This analysis function is no longer used, but may be
                // some time?
                // analyze_replacement(inds_advanced, current, previous, conf);

                num_advanced.push_back(inds_advanced.size());
                num_from_previous.push_back(from_previous.size());

                interfaces.pop_front();

#ifdef DEBUG_CONTACTLINE
                if (inds_advanced.size() > 0)
                {
                    const auto fraction =
                        static_cast<real>(num_from_previous.back())
                        / static_cast<real>(num_advanced.back());
                    fprintf(stderr, "%lu of %lu (%.2f) advancements were hops.\n",
                            num_from_previous.back(),
                            num_advanced.back(),
                            fraction);
                }
                else
                {
                    fprintf(stderr, "No advancement was made.\n");
                }
#endif
            }

#ifdef DEBUG_CONTACTLINE
            // write_sto_conf_indexed requires [int] data, not [size_t]
            // this is debugging so we don't care about performance,
            // just copy them
            const std::vector<int> int_inds(current.interface.cbegin(),
                                            current.interface.cend()),
                              cl_inds(current.contact_line.cbegin(),
                                      current.contact_line.cend());

            snprintf(debug_filename, MAXLEN, "contact_line_%05.1fps.gro", t);
            snprintf(debug_title, MAXLEN, "contact_line");
            write_sto_conf_indexed(debug_filename, debug_title,
                                   &top->atoms, x0, NULL, ePBC, box,
                                   cl_inds.size(),
                                   cl_inds.data());

            snprintf(debug_filename, MAXLEN, "interface_%05.1fps.gro", t);
            snprintf(debug_title, MAXLEN, "interface");
            write_sto_conf_indexed(debug_filename, debug_title,
                                   &top->atoms, x0, NULL, ePBC, box,
                                   int_inds.size(),
                                   int_inds.data());
#endif
            timings.loop.stop();
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
    }
    while (read_next_x(oenv, status, &t, x0, box));
    gmx_rmpbc_done(gpbc);

    timings.traj_total.stop();
    print_timings(timings);

#ifdef DEBUG_CONTACTLINE
    delete[] debug_filename;
    delete[] debug_title;
#endif

    close_trx(status);
    fprintf(stderr, "\nRead %d frames from trajectory.\n", 0);
    sfree(x0);
    delete pbc;

    const auto fractions = calculate_fractions(num_from_previous, num_advanced);

    std::vector<real> dists;
    if (conf.calc_jump_distance)
    {
        dists = calculate_mean_distance(distances);
    }

    const CLData result {
        times: times,
        fractions: fractions,
        distances: dists
    };

    return result;
}

static void
save_distances_figure(const CLData           &data,
                      const char             *filename,
                      const gmx_output_env_t *oenv)
{
    const auto title = "Molecule distance data";
    const auto xlabel = "Time (ps)";
    const auto ylabel = "Distance (nm)";

    auto file = xvgropen(filename, title, xlabel, ylabel, oenv);
    auto t = data.times.cbegin();
    auto d = data.distances.cbegin();

    while (t != data.times.cend())
    {

        fprintf(file, "%12.3f %1.3f\n", *t++, *d++);
    }

    xvgrclose(file);
}

static void
save_contact_line_figure(const CLData           &data,
                         const char             *filename,
                         const gmx_output_env_t *oenv)
{
    const auto title = "Contact line advancement data";
    const auto xlabel = "Time (ps)";
    const auto ylabel = "Fraction";

    auto file = xvgropen(filename, title, xlabel, ylabel, oenv);
    auto t = data.times.cbegin();
    auto f = data.fractions.cbegin();

    // Make sure that the time and data are correct
    // because we collect the fraction data only after
    // an initial stride (difference in frames)
    const auto diff = data.times.size() - data.fractions.size();
    for (unsigned i = 0; i < diff; ++i)
    {
        ++t;
    }

    while (t != data.times.cend())
    {

        fprintf(file, "%12.3f %1.3f\n", *t++, *f++);
    }

    xvgrclose(file);
}

int
gmx_contact_line(int argc, char *argv[])
{
    const char *desc[] = {
        "[THISMODULE] analyzes molecules at the contact line.",
        "",
    };

    static rvec rmin = { 0.0,  0.0,  0.0},
                rmax = {-1.0, -1.0, -1.0};
    static int nmin = 20,
               nmax = 100,
               stride = 1,
               seed = -1;
    static real cutoff = 1.0,
                precision = 0.3,
                dx = 0.3,
                hop_min = 0.25,
                hop_max = 0.35,
                ball_radius = 0.28;
    const char *algorithm[] = { NULL, "contact-line", "bottom", "hops", NULL };

    t_pargs pa[] = {
        { "-rmin", FALSE, etRVEC, { rmin },
          "Minimum coordinate values for atom positions to include." },
        { "-rmax", FALSE, etRVEC, { rmax },
          "Maximum coordinate values for atom positions to include." },
        { "-co", FALSE, etREAL, { &cutoff },
          "Cutoff distance for neighbour search." },
        { "-nmin", FALSE, etINT, { &nmin },
          "Minimum number of atoms within cutoff." },
        { "-nmax", FALSE, etINT, { &nmax },
          "Maximum number of atoms within cutoff." },
        { "-prec", FALSE, etREAL, { &precision },
          "Precision of slices along y and z." },
        { "-dx", FALSE, etREAL, { &dx },
          "Minimum distance for contact line advancement along x." },
        { "-hmin", FALSE, etREAL, { &hop_min },
          "Minimum distance for a hop to the contact line." },
        { "-hmax", FALSE, etREAL, { &hop_max },
          "Maximum distance for a hop to the contact line." },
        { "-stride", FALSE, etINT, { &stride },
          "Stride between contact line comparisons." },
        { "-al" , FALSE, etENUM, { &algorithm },
          "Algorithm for determining elegibility of atoms." },
        { "-ball" , FALSE, etREAL, { &ball_radius },
          "Radius of ball which is rolled over the interface." },
        { "-seed" , FALSE, etINT, { &seed },
          "Optional non-negative seed for the random number generation." },
    };

    const char *bugs[] = {
    };

    t_filenm fnm[] = {
        { efTRX, "-f", NULL,  ffREAD },
        { efNDX, NULL, NULL,  ffOPTRD },
        { efTPR, NULL, NULL,  ffREAD },
        { efXVG, "-o", "clfrac", ffWRITE },
        { efXVG, "-dist", "dist", ffOPTWR },
        { efDAT, "-interface", "interfaces", ffOPTWR },
    };

#define NFILE asize(fnm)

    gmx_output_env_t *oenv;
    if (!parse_common_args(&argc, argv, PCA_CAN_TIME | PCA_CAN_VIEW,
                           NFILE, fnm, asize(pa), pa, asize(desc), desc, asize(bugs), bugs,
                           &oenv))
    {
        return 0;
    }

    int ePBC;
    const auto top = read_top(ftp2fn(efTPR, NFILE, fnm), &ePBC);

    char **grpnames;
    int   *grpsizes;
    int  **index;

    snew(index, 1);
    snew(grpnames, 1);
    snew(grpsizes, 1);

    fprintf(stderr, "\nSelect group to analyze:\n");
    get_index(&top->atoms, ftp2fn_null(efNDX, NFILE, fnm),
              1, grpsizes, index, grpnames);

    struct CLConf conf {pa, asize(pa), rmin, rmax, nenum(algorithm), fnm, asize(fnm)};

    const auto contact_line_data = collect_contact_line_advancement(
        ftp2fn(efTRX, NFILE, fnm),
        *index,
        *grpsizes,
        conf,
        opt2fn("-interface", NFILE, fnm),
        top,
        ePBC,
        oenv);

    save_contact_line_figure(contact_line_data, opt2fn("-o", NFILE, fnm), oenv);

    if (conf.calc_jump_distance)
    {
        save_distances_figure(contact_line_data, opt2fn("-dist", NFILE, fnm), oenv);
    }

    view_all(oenv, NFILE, fnm);

    sfree(index);
    sfree(grpnames);
    sfree(grpsizes);

    return 0;
}
