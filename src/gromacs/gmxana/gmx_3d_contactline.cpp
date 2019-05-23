/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2017, by the GROMACS development team, led by
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

#include <algorithm> // set_difference
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <utility> // pair
#include <tuple> // tie

#include "gromacs/commandline/pargs.h"
#include "gromacs/commandline/viewit.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/matio.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/gmxana/gstat.h"
#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h"
#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/gmxomp.h"
#include "gromacs/utility/smalloc.h"

// Debug mode toggle
#define DEBUG3D

// Wrapper for debug mode-only execution (eg logging)
#ifdef DEBUG3D
    #define DEBUGLOG(x) x
#else
    #define DEBUGLOG(x)
#endif

#ifdef DEBUG3D
#define EPRINTLN(x) std::cerr << (#x) << " = " << (x) << '\n'; fflush(stderr);
#else
#define EPRINTLN(x);
#endif

constexpr double PI = 3.141592653589793;
constexpr double PI2 = 2.0 * 3.141592653589793;

template <typename T>
using Vec2 = std::array<T, 2>;
template <typename T>
using Vec3 = std::array<T, DIM>;

using RVec2 = Vec2<double>;
using RVec3 = Vec3<double>;
using Shape = Vec2<size_t>;

using Indices = std::vector<size_t>;

enum class DensityType {
    Mass,
    Number
};

enum class Dimensionality {
    Three,
    Two
};

struct DensMapBin {
    // Total mass in bin
    double mass;
    // Indices of atoms inside bin
    Indices indices;
};

struct DensMap {
    RVec3 bin_size;
    Shape shape;
    std::vector<DensMapBin> bins;
};

struct GraphXY {
    std::vector<double> x, y;
};

/*********************
 * Density Map Tools *
 *********************/

static size_t get_1d_index(const size_t ix, const size_t iy, const Shape &shape)
{
    return static_cast<size_t>((iy * shape[XX]) + ix);
}

static std::pair<size_t, size_t> get_2d_indices(const size_t i, const DensMap &densmap)
{
    const auto ix = i % densmap.shape[XX];
    const auto iy = static_cast<size_t>(
        floor(static_cast<double>(i) / static_cast<double>(densmap.shape[XX]))
    );

    return std::pair<size_t, size_t>{ix, iy};
}

static std::pair<size_t, size_t> get_2d_indices_from_pos(const real   x, 
                                                         const real   y,
                                                         const RVec3 &bin_size)
{
    return std::pair<size_t, size_t> { 
        static_cast<uint64_t>(floor(x / bin_size[XX])),
        static_cast<uint64_t>(floor(y / bin_size[YY]))
    };
}

static double get_1d_position(const size_t i, const double bin_size)
{
    return (static_cast<double>(i) + 0.5) * bin_size;
}

static std::pair<double, double> get_2d_position(const size_t   ix, 
                                                 const size_t   iy,
                                                 const DensMap &densmap)
{
    return std::pair<double, double> {
        get_1d_position(ix, densmap.bin_size[XX]),
        get_1d_position(iy, densmap.bin_size[YY])
    };
}


/****************************
 * Density Map Construction *
 ****************************/

static DensMap get_densmap(const rvec        *xs,
                           const real         bin_size,
                           const real         z0,
                           const real         dz,
                           const RVec3       &box_size,
                           const Indices     &indices,
                           const DensityType  type,
                           const t_topology  *top)
{
    const double z1 = z0 + dz;

    const auto bin_sizes = RVec3 { bin_size, bin_size, dz };

    const auto nx = static_cast<uint64_t>(ceil(box_size[XX] / bin_size));
    const auto ny = static_cast<uint64_t>(ceil(box_size[YY] / bin_size));
    const auto shape = Shape {nx, ny};

    std::vector<DensMapBin> bins(nx * ny, DensMapBin {0.0, Indices()});

    size_t ix, iy;

    for (const auto i : indices) 
    {
        const auto z = xs[i][ZZ];

        if ((z >= z0) && (z <= z1))
        {
            const auto x = xs[i][XX];
            const auto y = xs[i][YY];

            std::tie(ix, iy) = get_2d_indices_from_pos(x, y, bin_sizes);
            const auto index = get_1d_index(ix, iy, shape);

            switch (type)
            {
                case DensityType::Mass:
                    {
                        const auto mass = top->atoms.atom[i].m;
                        bins[index].mass += mass;
                    }
                    break;

                case DensityType::Number:
                    bins[index].mass += 1.0;
                    break;
            }

            bins[index].indices.push_back(i);
        }
    }

    return DensMap {
        bin_sizes,
        shape,
        bins
    };
}

// In order, write the density map to an input file:
// (The data format has been reused from `gmx_3d_analysis` which also 
// stores some data for a fitted spherical cap, these are blank here)
// 1. bin_size (3x double)
// 2. origin (2x double) (*blank*: always (0.0, 0.0))
// 3. shape (2x uint64)
// 4. center of spherical cap (2x double) (*blank*, always (0.0, 0.0))
// 5. simulation time (double)
// 6. data (NX * NY double)
static void write_densmap(const std::string &fnout,
                          const DensMap     &densmap,
                          const double       t)
{
    auto fp = gmx_ffopen(fnout.data(), "wb");

    const auto blank = RVec2 { 0.0, 0.0 };

    fwrite(densmap.bin_size.data(), sizeof(double), 3, fp);
    fwrite(blank.data(), sizeof(double), 2, fp);
    fwrite(densmap.shape.data(), sizeof(uint64_t), 2, fp);
    fwrite(blank.data(), sizeof(double), 2, fp);
    fwrite(&t, sizeof(double), 1, fp);

    for (const auto& bin : densmap.bins)
    {
        fwrite(&bin.mass, sizeof(double), 1, fp);
    }

    gmx_ffclose(fp);
}

static Indices get_indices_pbc(const size_t i, 
                               const size_t nsmooth, 
                               const size_t nx)
{
    const auto imin = static_cast<int>(i) - static_cast<int>(nsmooth);
    const auto imax = static_cast<int>(i + nsmooth);

    Indices indices;

    for (int i = imin; i <= imax; ++i)
    {
        auto j = i;

        while (j < 0)
        {
            j += nx;
        }

        while (static_cast<size_t>(j) >= nx)
        {
            j -= nx;
        }

        indices.push_back(j);
    }

    return indices;
}

static DensMap smooth_densmap(const DensMap &densmap, const uint64_t nsmooth)
{
    std::vector<DensMapBin> bins(densmap.bins.size(), {0.0, Indices()});

    size_t from_ix, from_iy;

    for (size_t from_index = 0; from_index < bins.size(); ++from_index)
    {
        std::tie(from_ix, from_iy) = get_2d_indices(from_index, densmap);
        const auto ixs = get_indices_pbc(from_ix, nsmooth, densmap.shape[XX]);
        const auto iys = get_indices_pbc(from_iy, nsmooth, densmap.shape[YY]);

        for (const auto to_ix : ixs) 
        {
            for (const auto to_iy : iys)
            {
                const auto to_index = get_1d_index(to_ix, to_iy, densmap.shape);
                bins[to_index].mass += densmap.bins[from_index].mass;
            }
        }
    }

    const uint64_t num_smoothing_bins = std::pow(((2 * nsmooth) + 1), 2);

    for (size_t index = 0; index < bins.size(); ++index)
    {
        bins[index].mass /= static_cast<double>(num_smoothing_bins);
        bins[index].indices = densmap.bins[index].indices;
    }

    return DensMap {
        densmap.bin_size,
        densmap.shape,
        bins
    };
}

static RVec2 calc_com(const DensMap &densmap)
{
    double xsum = 0.0, ysum = 0.0, sum = 0.0;
    size_t ix, iy;

    for (size_t i = 0; i < densmap.bins.size(); ++i)
    {
        const auto mass = densmap.bins[i].mass;

        if (mass > 0.0)
        {
            std::tie(ix, iy) = get_2d_indices(i, densmap);
            const auto x = static_cast<double>(ix) * densmap.bin_size[XX];
            const auto y = static_cast<double>(iy) * densmap.bin_size[YY];

            xsum += x * mass;
            ysum += y * mass;
            sum += mass;
        }
    }

    xsum /= sum;
    ysum /= sum;

    return RVec2 { xsum, ysum };
}

static double get_cutoff(const DensMap &densmap, const double frac)
{
    double mavg = 0.0;
    uint64_t n = 0;

    for (const auto& bin : densmap.bins)
    {
        if (bin.mass > 0.0)
        {
            mavg += bin.mass;
            n += 1;
        }
    }

    mavg /= static_cast<double>(n);

    return frac * mavg;
}

static DensMap apply_cutoff(const DensMap &densmap, const double cutoff)
{
    std::vector<DensMapBin> bins(densmap.bins.size(), {0.0, Indices()});

    for (size_t i = 0; i < bins.size(); ++i)
    {
        if (densmap.bins[i].mass >= cutoff)
        {
            bins[i].mass = densmap.bins[i].mass;
            bins[i].indices = densmap.bins[i].indices;
        }
    }

    return DensMap {
        densmap.bin_size,
        densmap.shape,
        bins
    };
}


/*******************************
 * Contact Line Identification *
 *******************************/

static double find_initial_radius(const DensMap &densmap, 
                                  const RVec2   &center, 
                                  const double   cutoff)
{
    const auto x0 = center[XX];
    const auto y0 = center[YY];

    size_t ix0, iy0;
    std::tie(ix0, iy0) = get_2d_indices_from_pos(x0, y0, densmap.bin_size);
    const auto i0 = get_1d_index(ix0, iy0, densmap.shape);

    size_t ix = ix0,
           i = i0;

    // Move along x until the first filled cell is found,
    // then find the edge from that point by moving outwards
    while ((densmap.bins[i].mass < cutoff) && (ix + 1 < densmap.shape[XX]))
    {
        ++ix;
        i = get_1d_index(ix, iy0, densmap.shape);
    }

    while ((densmap.bins[i].mass >= cutoff) && (ix + 1 < densmap.shape[XX]))
    {
        ++ix;
        i = get_1d_index(ix, iy0, densmap.shape);
    }

    --ix;

    const auto x = get_1d_position(ix, densmap.bin_size[XX]);

    return x - x0;
}

static Indices get_neighbouring_bins(const size_t  ix, 
                                     const size_t  iy,
                                     const size_t  nx,
                                     const size_t  ny,
                                     const Shape  &shape)
{
    Indices indices;
    indices.reserve(nx * ny);

    const auto ixmin = (static_cast<int>(ix) - static_cast<int>(nx) >= 0) ? ix - nx : 0;
    const auto ixmax = ((ix + nx) < shape[XX]) ? ix + nx : shape[XX] - 1;
    const auto iymin = (static_cast<int>(iy) - static_cast<int>(ny) >= 0) ? iy - ny : 0;
    const auto iymax = ((iy + ny) < shape[YY]) ? iy + ny : shape[YY] - 1;

    for (size_t i = ixmin; i <= ixmax; ++i)
    {
        for (size_t j = iymin; j <= iymax; ++j)
        {
            const auto index = get_1d_index(i, j, shape);
            indices.push_back(index);
        }
    }
    
    return indices;
}

static bool check_collision(const double                   x0,
                            const double                   y0,
                            const Indices                 &neighbours,
                            const std::vector<DensMapBin> &bins,
                            const double                   dr2_min,
                            const rvec                    *xs)
{

    for (const auto n : neighbours)
    {
        const auto& bin = bins[n];

        for (const auto i : bin.indices)
        {
            const auto x1 = xs[i][XX];
            const auto y1 = xs[i][YY];

            const auto dr2 = std::pow(x1 - x0, 2) + std::pow(y1 - y0, 2);

            if (dr2 < dr2_min)
            {
                return true;
            }
        }
    }

    return false;
}

static Indices get_collisions(const double                   x0,
                              const double                   y0,
                              const Indices                 &neighbours,
                              const std::vector<DensMapBin> &bins,
                              const double                   dr2_min,
                              const rvec                    *xs)
{
    Indices indices;

    for (const auto n : neighbours)
    {
        const auto& bin = bins[n];

        for (const auto i : bin.indices)
        {
            const auto x1 = xs[i][XX];
            const auto y1 = xs[i][YY];

            const auto dr2 = std::pow(x1 - x0, 2) + std::pow(y1 - y0, 2);

            if (dr2 < dr2_min)
            {
                indices.push_back(i);
            }
        }
    }

    return indices;
}

static std::pair<Indices, GraphXY> 
get_boundary_atoms(const rvec    *xs,
                   const DensMap &densmap, 
                   const RVec2   &center, 
                   const double   cutoff_mass_fraction,
                   const double   ball_radius)
{
    constexpr double da_precision = 0.1; // Try to revolve around r in increments of 0.1 nm
    constexpr double dr_precision = 0.1;

    // The boundary atoms are collected like: 
    //
    // 1. A cutoff is used to remove atoms in "empty" (below-cutoff) bins. 
    //    This takes care of loose atoms in the system.
    //
    // 2. The initial radius of the ball is determined by detecting the 
    //    (outer) contact line position.
    //
    // 3. The ball is rolled to find r(theta): atoms which hinder it from 
    //    moving inwards are collected as the contact line atoms.

    const auto cutoff = get_cutoff(densmap, cutoff_mass_fraction);
    const auto cutoff_densmap = apply_cutoff(densmap, cutoff);

    const auto r0 = find_initial_radius(cutoff_densmap, center, cutoff);
    const auto x0 = center[XX];
    const auto y0 = center[YY];

    // We limit the search space to bins within the ball radius from 
    // the current bin. These are how many extra bins we need to check
    // in each direction.
    const auto nx = static_cast<size_t>(ceil(ball_radius / densmap.bin_size[XX]));
    const auto ny = static_cast<size_t>(ceil(ball_radius / densmap.bin_size[YY]));

    double r = r0,
           angle = 0.0,
           dr2_min = std::pow(ball_radius, 2);

    size_t ix, iy,
           ix_prev, iy_prev;

    const auto xinit = x0 + r * cos(angle);
    const auto yinit = y0 + r * sin(angle);
    std::tie(ix_prev, iy_prev) = get_2d_indices_from_pos(xinit, yinit, densmap.bin_size);

    auto neighbours = get_neighbouring_bins(ix_prev, iy_prev, nx, ny, densmap.shape);

    // Control over whether the radius is increasing or decreasing until a collision
    bool update_angle = true,
         increase_radius = true;

    std::set<size_t> boundary_indices;
    std::vector<double> avals, rvals;

    while (angle < 2.0 * PI)
    {
        auto x = x0 + r * cos(angle);
        auto y = y0 + r * sin(angle);
        std::tie(ix, iy) = get_2d_indices_from_pos(x, y, densmap.bin_size);

        if ((ix != ix_prev) || (iy != iy_prev))
        {
            neighbours = get_neighbouring_bins(ix, iy, nx, ny, densmap.shape);
        }

        const auto collision = check_collision(
            x, y, neighbours, cutoff_densmap.bins, dr2_min, xs
        );

        if (update_angle)
        {
            increase_radius = collision;
            update_angle = false;
        }

        if (increase_radius)
        {
            if (!collision)
            {
                update_angle = true;

                // The previous radius was the hit since we are increasing
                r -= dr_precision;
                x = x0 + r * cos(angle);
                y = y0 + r * sin(angle);
            }
            else 
            {
                r += dr_precision;
            }
        }
        else 
        {
            if (collision)
            {
                update_angle = true;
            }
            else 
            {
                r -= dr_precision;
            }
        }

        if (update_angle)
        {
            avals.push_back(angle);
            rvals.push_back(r);

            const auto colliding_indices = 
                get_collisions(x, y, neighbours, cutoff_densmap.bins, dr2_min, xs);
            for (const auto i : colliding_indices)
            {
                boundary_indices.insert(i);
            }

            const double da = (r > 0.0) ? da_precision / r : 0.01;
            angle += da;
        }

        ix_prev = ix;
        iy_prev = iy;
    }

    Indices boundary(boundary_indices.cbegin(), boundary_indices.cend());
    std::sort(boundary.begin(), boundary.end());

    const GraphXY boundary_line { avals, rvals };

    return std::pair<Indices, GraphXY> {
        boundary,
        boundary_line
    };
}

static std::pair<Indices, GraphXY> 
get_boundary_atoms_2d(const rvec    *xs,
                      const DensMap &densmap, 
                      const RVec2   &center, 
                      const RVec3   &box_size,
                      const double   cutoff_mass_fraction,
                      const double   ball_radius)
{
    constexpr double dx_precision = 0.05;
    constexpr double dy_precision = 0.05;

    // The boundary atoms are collected like: 
    //
    // 1. A cutoff is used to remove atoms in "empty" (below-cutoff) bins. 
    //    This takes care of loose atoms in the system.
    //
    // 2. The initial x position of the ball is determined by detecting the 
    //    (outer) contact line position.
    //
    // 3. The ball is rolled to find x(y): atoms which hinder it from 
    //    moving inwards are collected as the contact line atoms.

    const auto cutoff = get_cutoff(densmap, cutoff_mass_fraction);
    const auto cutoff_densmap = apply_cutoff(densmap, cutoff);

    const double x0 = find_initial_radius(cutoff_densmap, center, cutoff);
    const auto y0 = 0.0;

    // We limit the search space to bins within the ball radius from 
    // the current bin. These are how many extra bins we need to check
    // in each direction.
    const auto nx = static_cast<size_t>(ceil(ball_radius / densmap.bin_size[XX]));
    const auto ny = static_cast<size_t>(ceil(ball_radius / densmap.bin_size[YY]));

    double x = x0 + center[XX],
           y = y0,
           dr2_min = std::pow(ball_radius, 2);

    size_t ix, iy,
           ix_prev, iy_prev;

    std::tie(ix_prev, iy_prev) = get_2d_indices_from_pos(x, y, densmap.bin_size);
    auto neighbours = get_neighbouring_bins(ix_prev, iy_prev, nx, ny, densmap.shape);

    // // Control over whether the position is increasing or decreasing until a collision
    bool update_y = true,
         increase_x = true;

    std::set<size_t> boundary_indices;
    std::vector<double> xvals, yvals;

    while (y <= box_size[YY])
    {
        std::tie(ix, iy) = get_2d_indices_from_pos(x, y, densmap.bin_size);

        if ((ix != ix_prev) || (iy != iy_prev))
        {
            neighbours = get_neighbouring_bins(ix, iy, nx, ny, densmap.shape);
        }

        const auto collision = check_collision(
            x, y, neighbours, cutoff_densmap.bins, dr2_min, xs
        );

        if (update_y)
        {
            increase_x = collision;
            update_y = false;
        }

        if (increase_x)
        {
            if (!collision)
            {
                update_y = true;

                // The previous x was the hit since we are increasing
                x -= dx_precision;
            }
            else 
            {
                x += dx_precision;
            }
        }
        else 
        {
            if (collision)
            {
                update_y = true;
            }
            else 
            {
                x -= dx_precision;
            }
        }

        if (update_y)
        {
            xvals.push_back(x);
            yvals.push_back(y);

            const auto colliding_indices = 
                get_collisions(x, y, neighbours, cutoff_densmap.bins, dr2_min, xs);
            for (const auto i : colliding_indices)
            {
                boundary_indices.insert(i);
            }

            y += dy_precision;
        }

        ix_prev = ix;
        iy_prev = iy;
    }

    Indices boundary(boundary_indices.cbegin(), boundary_indices.cend());
    std::sort(boundary.begin(), boundary.end());

    const GraphXY boundary_line { xvals, yvals };

    return std::pair<Indices, GraphXY> {
        boundary,
        boundary_line
    };
}


/******************
 * Analysis Tools *
 ******************/

struct BoundaryAtoms {
    Indices indices;
    std::vector<RVec3> positions;
    double radius, time;
};

// Assumes that the boundary indices in all sets are sorted
static GraphXY calc_autocorrelation(const std::vector<BoundaryAtoms> boundary)
{
    std::vector<double> times, 
                        values(boundary.size(), 0.0);
    double t0 = 0.0;

    if (!boundary.empty())
    {
        t0 = boundary[0].time;
    }

    for (size_t i = 0; i < boundary.size(); ++i)
    {
        const auto from = boundary[i].indices;
        times.push_back(boundary[i].time - t0);

        for (size_t j = i + 1; j < boundary.size(); ++j)
        {
            const auto lag = j - i;
            const auto to = boundary[j].indices;

            Indices intersection;
            std::set_intersection(
                from.cbegin(), from.cend(),
                to.cbegin(), to.cend(),
                std::back_inserter(intersection)
            );

            const auto ac = 
                static_cast<double>(intersection.size()) / static_cast<double>(from.size());
            
            values[lag] += ac;
        }
    }

    if (!values.empty())
    {
        values[0] = 1.0;
    }

    // Take the mean for all correlations
    for (size_t i = 1; i < values.size(); ++i)
    {
        values[i] /= static_cast<double>(values.size() - i);
    }

    return GraphXY {
        times,
        values
    };
}

static double calc_mean_radius(const std::vector<RVec3> &positions, 
                               const RVec2              &center,
                               const Dimensionality     &dim)
{
    double sum = 0.0;

    const auto x0 = center[XX];
    const auto y0 = center[YY];

    for (const auto pos : positions)
    {
        const auto x = pos[XX];
        const auto y = pos[YY];

        switch (dim)
        {
            case Dimensionality::Three:
                sum += sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2));
                break;
            case Dimensionality::Two:
                sum += x - x0;
                break;
        }
    }

    return sum / static_cast<double>(positions.size());
}

static GraphXY 
calc_advancement_intersections(const std::vector<BoundaryAtoms> &boundary,
                               const double                      dr_min)
{
    std::vector<double> adv_times,
                        fractions;

    if (boundary.size() > 0)
    {
        // For each boundary, get the first next boundary which advances 
        // the line by the input amount.
        for (size_t i = 0; i < boundary.size(); ++i)
        {
            const auto& previous = boundary[i];

            for (size_t j = i; j < boundary.size(); ++j)
            {
                const auto& current = boundary[j];
                const auto dr = current.radius - previous.radius;   

                if (dr >= dr_min)
                {
                    const auto dt = current.time - previous.time;

                    Indices intersection;
                    std::set_intersection(
                        previous.indices.cbegin(), previous.indices.cend(),
                        current.indices.cbegin(), current.indices.cend(),
                        std::back_inserter(intersection)
                    );

                    const auto frac = static_cast<double>(intersection.size()) 
                        / static_cast<double>(previous.indices.size());

                    adv_times.push_back(dt);
                    fractions.push_back(frac);

                    break;
                }
            }
        }
    }

    return GraphXY { 
        adv_times,
        fractions
    };
}

static GraphXY count_contact_line_atoms(const std::vector<BoundaryAtoms> boundary)
{
    std::vector<double> times, values;
    double t0 = 0.0;

    if (!boundary.empty())
    {
        t0 = boundary[0].time;
    }

    for (const auto& b : boundary)
    {
        times.push_back(b.time - t0);
        values.push_back(static_cast<double>(b.indices.size()));
    }

    return GraphXY {
        times,
        values
    };
}

static double calc_graph_mean(const GraphXY &graph)
{
    double sum = 0.0;

    for (const auto& v : graph.y)
    {
        sum += v;
    }

    return sum / static_cast<double>(graph.y.size());
}

static double calc_graph_median(const GraphXY &graph)
{
    auto values = std::vector<double>(graph.y.cbegin(), graph.y.cend());
    std::sort(values.begin(), values.end());

    const auto i = static_cast<size_t>(floor(values.size() / 2));

    if (values.size() > 0)
    {
        return values[i];
    }
    else
    {
        return 0.0;
    }
}


/**********************
 * Main Functionality *
 **********************/

static std::vector<RVec3> copy_positions(const rvec *xs, const Indices indices)
{
    std::vector<RVec3> positions;
    positions.reserve(indices.size());

    for (const auto i : indices)
    {
        positions.push_back(RVec3 { xs[i][XX], xs[i][YY], xs[i][ZZ] });
    }

    return positions;
}

static std::string get_fnbase(const char *fnarg, 
                              const char *sep,
                              const int fntype, 
                              const t_filenm fnm[], 
                              const int nfnm)
{
    const std::string ext { ftp2ext_with_dot(fntype) };

    std::string fnbase { opt2fn(fnarg, nfnm, fnm) };
    fnbase.resize(fnbase.size() - ext.size());

    return std::string { fnbase + sep };
}

static std::string get_fnend(const char *pre, const int fntype)
{
    return std::string { pre + std::string(ftp2ext_with_dot(fntype)) };
}

static void write_graph_to_xvg(const GraphXY          &graph,
                               const char             *fn,
                               const char             *title,
                               const char             *xlabel,
                               const char             *ylabel,
                               const gmx_output_env_t *oenv)
{
    auto fp = xvgropen_type(fn, title, xlabel, ylabel, exvggtXNY, oenv);

    auto xiter = graph.x.cbegin();
    auto yiter = graph.y.cbegin();

    const auto xend = graph.x.cend();
    const auto yend = graph.y.cend();

    while ((xiter != xend) && (yiter != yend))
    {
        fprintf(fp, "%12g %12g\n", *xiter++, *yiter++);
    }

    xvgrclose(fp);
}

int gmx_3d_contactline(int argc, char *argv[])
{
    const char        *desc[] = {
        "[THISMODULE] analyzes the contact line atomic motions of 3D droplets.",
    };
    static real ball_radius = 0.3,
                bin_size = 0.25,
                dr_adv = 0.25,
                mfrac = 0.5,
                z0 = 0.0,
                dz = 0.25;
    static int nsmooth = 1;
    static rvec center { -1, -1, -1 };

    enum DimensionalitySel
    {
        eDimSel,
        eDim3d,
        eDim2d,
        eDimN
    };
    const char *dim_opt[eDimN + 1] = { nullptr, "3d", "2d", nullptr };

    t_pargs pa[] = {
        { "-bin", FALSE, etREAL, {&bin_size},
          "Grid size for interface search (nm)" },
        { "-center", FALSE, etRVEC, {&center},
          "Center of droplet" },
        { "-mfrac", FALSE, etREAL, {&mfrac},
          "Fraction of mean bin mass to use as cutoff" },
        { "-z", FALSE, etREAL, {&z0},
          "Position along z of bottom interface (nm)" },
        { "-dz", FALSE, etREAL, {&dz},
          "Height of bottom interface (nm)" },
        { "-dr", FALSE, etREAL, {&dr_adv},
          "Mean distance for contact line advancement (nm)" },
        { "-br", FALSE, etREAL, {&ball_radius},
          "Radius of ball used to roll interface (nm)"},
        { "-nsmooth", FALSE, etINT, {&nsmooth},
          "Smoothing window size for density map" },
        { "-dim", FALSE, etENUM, { &dim_opt }, 
          "Dimensionality of system" }
    };

    t_trxstatus       *status;
    t_topology         top;
    int                ePBC = -1;
    rvec              *x;
    matrix             box;
    real               t;
    char             **grpname;
    int                ngrps, anagrp, *gnx = nullptr, nindex;
    int              **ind = nullptr, *index;
    gmx_output_env_t  *oenv;
    t_filenm           fnm[]   = {
        { efTRX, "-f",   nullptr,       ffREAD },
        { efTPS, nullptr,   nullptr,       ffOPTRD },
        { efNDX, nullptr,   nullptr,       ffOPTRD },
        { efXVG, "-oa", "autocorrelation", ffWRITE },
        { efXVG, "-on", "number", ffWRITE },
        { efXVG, "-oi", "histogram", ffWRITE },
        { efDAT, "-dens", "densmap", ffWRITE },
        { efDAT, "-smooth", "smooth_densmap", ffWRITE }
    };

#define NFILE asize(fnm)

    const int npargs = asize(pa);
    if (!parse_common_args(&argc, argv, PCA_CAN_TIME | PCA_CAN_VIEW,
                           NFILE, fnm, npargs, pa, asize(desc), desc, 0, nullptr, &oenv))
    {
        return 0;
    }

    // Use number density if we cannot read a topology which will have the masses
    DensityType type = DensityType::Number;
    if (ftp2bSet(efTPS, NFILE, fnm) || !ftp2bSet(efNDX, NFILE, fnm))
    {
        if (read_tps_conf(ftp2fn(efTPS, NFILE, fnm), &top, &ePBC, &x, nullptr, box, false)) 
        {
            type = DensityType::Mass;
        };
    } 

    // Parse the dimensionality
    auto dim_enum = Dimensionality::Three;
    if (nenum(dim_opt) == eDim2d)
    {
        dim_enum = Dimensionality::Two;
    }

    const auto bWriteDensmap = static_cast<bool>(opt2bSet("-dens", NFILE, fnm));
    const auto bWriteSmooth = static_cast<bool>(opt2bSet("-smooth", NFILE, fnm));
    const auto fnbase_densmap = get_fnbase("-dens", "_", efDAT, fnm, NFILE);
    const auto fnbase_smooth = get_fnbase("-smooth", "_", efDAT, fnm, NFILE);
    const auto fndat_end = get_fnend("ps", efDAT);

    /* Either use an input center or the center of mass */
    const auto use_com_center = !static_cast<bool>(opt2parg_bSet("-center", npargs, pa));
    RVec2 r0 {center[XX], center[YY]};

    ngrps = 1;
    fprintf(stderr, "\nSelect an analysis group\n");
    snew(gnx, ngrps);
    snew(grpname, ngrps);
    snew(ind, ngrps);
    get_index(&top.atoms, ftp2fn_null(efNDX, NFILE, fnm), ngrps, gnx, ind, grpname);
    anagrp = ngrps - 1;
    nindex = gnx[anagrp];
    index  = ind[anagrp];
    const Indices indices(index, index + nindex);

    std::vector<BoundaryAtoms> boundaries_traj;
    read_first_x(oenv, &status, ftp2fn(efTRX, NFILE, fnm), &t, &x, box);

    do
    {
        const auto box_size = RVec3 { box[XX][XX], box[YY][YY], box[ZZ][ZZ] };
        const auto densmap = get_densmap(x, bin_size, z0, dz, box_size, indices, type, &top);
        const auto smooth = smooth_densmap(densmap, static_cast<uint64_t>(nsmooth));

        if (use_com_center) 
        {
            r0 = calc_com(densmap);
        }

        if (bWriteDensmap)
        {
            char buf[16];
            snprintf(buf, 16, "%09.3f", t);
            std::string fnmap_current { fnbase_densmap + buf + fndat_end };
            write_densmap(fnmap_current, densmap, t);
        }

        if (bWriteSmooth)
        {
            char buf[16];
            snprintf(buf, 16, "%09.3f", t);
            std::string fnmap_current { fnbase_smooth + buf + fndat_end };
            write_densmap(fnmap_current, smooth, t);
        }

        Indices boundary_indices;
        GraphXY boundary_line;

        switch (dim_enum)
        {
            case Dimensionality::Three:
                std::tie(boundary_indices, boundary_line) = 
                    get_boundary_atoms(x, smooth, r0, mfrac, ball_radius);
                break;

            case Dimensionality::Two:
                std::tie(boundary_indices, boundary_line) = 
                    get_boundary_atoms_2d(x, smooth, r0, box_size, mfrac, ball_radius);
                break;
        }

        DEBUGLOG(
            std::vector<double> xvals; 
            std::vector<double> yvals;

            char buf[48];
            snprintf(buf, 48, "debug_boundary_%05.1fps.xvg", t);

            switch (dim_enum)
            {
                case Dimensionality::Three:
                    for (size_t i = 0; i < boundary_line.x.size(); ++i)
                    {
                        const auto a = boundary_line.x[i];
                        const auto r = boundary_line.y[i];

                        const auto x = r * cos(a);
                        const auto y = r * sin(a);

                        xvals.push_back(x);
                        yvals.push_back(y);
                    }

                    break;

                case Dimensionality::Two:
                    xvals = std::vector<double>(
                        boundary_line.x.cbegin(), boundary_line.x.cend()
                    );
                    yvals = std::vector<double>(
                        boundary_line.y.cbegin(), boundary_line.y.cend()
                    );
                    break;
            }

            GraphXY tmp_graph;
            tmp_graph.x = xvals;
            tmp_graph.y = yvals;

            write_graph_to_xvg(tmp_graph, buf, "Boundary", "x (nm)", "y (nm)", oenv);
        )

        // const auto boundary_indices = get_boundary_atoms(x, smooth, r0, mfrac, 0.3);
        const auto positions = copy_positions(x, boundary_indices);
        const auto mean_radius = calc_mean_radius(positions, r0, dim_enum);
        boundaries_traj.push_back(
            BoundaryAtoms { boundary_indices, positions, mean_radius, t }
        );

        // TODO: Figure out a criteria for when the contact line advances 
        // and how to determine which atom did it

        // DEBUGLOG(
        //     // Write the boundary as a .gro file for debugging
        //     char buf[20];
        //     snprintf(buf, 20, "debug_cl_%05.1f.gro", t);

        //     const std::vector<int> boundary_ints(
        //         boundary_indices.cbegin(), boundary_indices.cend()
        //     );

        //     write_sto_conf_indexed(buf, "contact_line",
        //                            &top.atoms, x, NULL, ePBC, box,
        //                            boundary_ints.size(),
        //                            boundary_ints.data());
        // )
    }
    while (read_next_x(oenv, status, &t, x, box));
    close_trx(status);

    const auto autocorrelation = calc_autocorrelation(boundaries_traj);
    write_graph_to_xvg(autocorrelation, opt2fn("-oa", NFILE, fnm), 
                       "Autocorrelation", "t (ps)", "Autocorrelation", 
                       oenv);

    const auto adv_fractions = calc_advancement_intersections(boundaries_traj, dr_adv);
    write_graph_to_xvg(adv_fractions, opt2fn("-oi", NFILE, fnm), 
                       "Intersection fraction", "t (ps)", "Fraction", 
                       oenv);
    
    const auto fraction_mean = calc_graph_mean(adv_fractions);
    const auto fraction_median = calc_graph_median(adv_fractions);

    std::cout 
        << "Advancing jump fraction mean:   " << fraction_mean << '\n'
        << "Advancing jump fraction median: " << fraction_median << '\n';

    const auto num_atoms = count_contact_line_atoms(boundaries_traj);
    write_graph_to_xvg(num_atoms, opt2fn("-on", NFILE, fnm), 
                       "Number of contact line atoms", "t (ps)", "N", 
                       oenv);

    do_view(oenv, opt2fn("-oa", NFILE, fnm), nullptr);

    return 0;
}


/*****************
 * Indices tools *
 *****************/

static void dipole_atom2molindex(int *n, int *index, const t_block *mols)
{
    int nmol, i, j, m;

    nmol = 0;
    i    = 0;
    while (i < *n)
    {
        m = 0;
        while (m < mols->nr && index[i] != mols->index[m])
        {
            m++;
        }
        if (m == mols->nr)
        {
            gmx_fatal(FARGS, "index[%d]=%d does not correspond to the first atom of a molecule", i+1, index[i]+1);
        }
        for (j = mols->index[m]; j < mols->index[m+1]; j++)
        {
            if (i >= *n || index[i] != j)
            {
                gmx_fatal(FARGS, "The index group is not a set of whole molecules");
            }
            i++;
        }
        /* Modify the index in place */
        index[nmol++] = m;
    }
    printf("There are %d molecules in the selection\n", nmol);

    *n = nmol;
}

static Indices get_molecule_indices_for_atoms(const Indices    &molecule_inds,
                                              const t_topology *top)
{
    const auto mols = &(top->mols);
    const auto natoms = top->atoms.nr;

    Indices atom2molecule (natoms, 0);

    for (const auto i : molecule_inds)
    {
        const auto i0 = static_cast<size_t>(mols->index[i]);
        const auto i1 = static_cast<size_t>(mols->index[i + 1]);

        for (size_t j = i0; j < i1; ++j)
        {
            atom2molecule[j] = i;
        }
    }

    return atom2molecule;
}

static std::set<size_t> get_molecules(const Indices &boundary_indices,
                                      const Indices &atom2mol)
{
    std::set<size_t> indices;

    for (const auto i : boundary_indices)
    {
        indices.insert(atom2mol[i]);
    }

    return indices;
} 

static Indices molecule_to_atom_indices(const std::set<size_t> &molecule_inds,
                                        const t_topology       *top)
{
    Indices indices;
    const auto mols = &(top->mols);

    for (const auto i : molecule_inds)
    {
        const auto i0 = static_cast<size_t>(mols->index[i]);
        const auto i1 = static_cast<size_t>(mols->index[i + 1]);

        for (size_t j = i0; j < i1; ++j)
        {
            indices.push_back(j);
        }
    }

    return indices;
}

static void write_index_file(const std::string &fn,
                             const Indices     &inds,
                             const std::string &grpname)
{
    auto fp = gmx_ffopen(fn.data(), "w");

    fprintf(fp, "[ %s ]\n", grpname.c_str());

    size_t n = 0;

    for (const auto i : inds)
    {
        fprintf(fp, "%lu ", i + 1);

        if (++n % 15 == 0)
        {
            fprintf(fp, "\n");
        }
    }

    if (n % 15 != 0)
    {
        fprintf(fp, "\n");
    }

    gmx_ffclose(fp);

    return;
}

static Indices get_densmap_indices(const DensMap &densmap)
{
    Indices inds;

    for (const auto& bin : densmap.bins)
    {
        for (const auto i : bin.indices)
        {
            inds.push_back(i);
        }
    }

    return inds;
}

int gmx_contactline_indices(int argc, char *argv[])
{
    const char        *desc[] = {
        "[THISMODULE] gets the indices from molecules at the contact line.",
    };
    static real ball_radius = 0.3,
                bin_size = 0.25,
                dr_adv = 0.25,
                mfrac = 0.5,
                z0 = 0.0,
                dz = 0.25;
    static int nsmooth = 1;
    static rvec center { -1, -1, -1 };

    enum DimensionalitySel
    {
        eDimSel,
        eDim3d,
        eDim2d,
        eDimN
    };
    const char *dim_opt[eDimN + 1] = { nullptr, "3d", "2d", nullptr };

    t_pargs pa[] = {
        { "-bin", FALSE, etREAL, {&bin_size},
          "Grid size for interface search (nm)" },
        { "-center", FALSE, etRVEC, {&center},
          "Center of droplet" },
        { "-mfrac", FALSE, etREAL, {&mfrac},
          "Fraction of mean bin mass to use as cutoff" },
        { "-z", FALSE, etREAL, {&z0},
          "Position along z of bottom interface (nm)" },
        { "-dz", FALSE, etREAL, {&dz},
          "Height of bottom interface (nm)" },
        { "-dr", FALSE, etREAL, {&dr_adv},
          "Mean distance for contact line advancement (nm)" },
        { "-br", FALSE, etREAL, {&ball_radius},
          "Radius of ball used to roll interface (nm)"},
        { "-nsmooth", FALSE, etINT, {&nsmooth},
          "Smoothing window size for density map" },
        { "-dim", FALSE, etENUM, { &dim_opt }, 
          "Dimensionality of system" }
    };

    t_topology         top;
    int                ePBC = -1;
    rvec              *x;
    matrix             box;
    char             **grpname;
    int                ngrps, *gnx = nullptr;
    int              **ind = nullptr, *grpindex;
    gmx_output_env_t  *oenv;
    t_filenm           fnm[]   = {
        { efTPS, "-f",     nullptr,             ffREAD },
        { efNDX, nullptr,  nullptr,             ffOPTRD },
        { efNDX, "-oi",    "index_contactline", ffWRITE },
        { efNDX, "-ob",    "index_bottom",      ffWRITE }
    };

#define NFILE asize(fnm)

    const int npargs = asize(pa);
    if (!parse_common_args(&argc, argv, PCA_CAN_TIME | PCA_CAN_VIEW,
                           NFILE, fnm, npargs, pa, asize(desc), desc, 0, nullptr, &oenv))
    {
        return 0;
    }

    // Use number density if we cannot read a topology which will have the masses
    DensityType type = DensityType::Number;
    if (ftp2bSet(efTPS, NFILE, fnm) || !ftp2bSet(efNDX, NFILE, fnm))
    {
        if (read_tps_conf(ftp2fn(efTPS, NFILE, fnm), &top, &ePBC, &x, nullptr, box, false)) 
        {
            type = DensityType::Mass;
        };
    } 

    // Parse the dimensionality
    auto dim_enum = Dimensionality::Three;
    if (nenum(dim_opt) == eDim2d)
    {
        dim_enum = Dimensionality::Two;
    }

    /* Either use an input center or the center of mass */
    const auto use_com_center = !static_cast<bool>(opt2parg_bSet("-center", npargs, pa));
    RVec2 r0 {center[XX], center[YY]};

    ngrps = 1;
    fprintf(stderr, "\nSelect an analysis group\n");
    snew(gnx, ngrps);
    snew(grpname, ngrps);
    snew(ind, ngrps);
    get_index(&top.atoms, ftp2fn_null(efNDX, NFILE, fnm), ngrps, gnx, ind, grpname);
    grpindex  = ind[0];

    const Indices atom_indices(grpindex, grpindex + gnx[0]);

    // Get atom index ranges for all molecules and vice versa: 
    // after getting the boundary atom indices we want to get which molecules
    // they belong to, then finally write all indices for those molecules to disk.
    dipole_atom2molindex(&gnx[0], grpindex, &(top.mols));
    const Indices molecule_indices(grpindex, grpindex + gnx[0]);
    const auto atom2mol = get_molecule_indices_for_atoms(molecule_indices, &top);

    const auto box_size = RVec3 { box[XX][XX], box[YY][YY], box[ZZ][ZZ] };

    // Bottom substrate-bonded molecules
    const auto bottom_densmap = get_densmap(x, bin_size, z0, dz, box_size, atom_indices, type, &top);
    const auto bottom_indices = get_densmap_indices(bottom_densmap);
    const auto bottom_mols = get_molecules(bottom_indices, atom2mol);
    const auto final_bottom_indices = molecule_to_atom_indices(bottom_mols, &top);

    // Second layer boundary molecules
    const auto densmap = get_densmap(x, bin_size, z0 + dz, dz, box_size, atom_indices, type, &top);
    const auto smooth = smooth_densmap(densmap, static_cast<uint64_t>(nsmooth));

    if (use_com_center) 
    {
        r0 = calc_com(densmap);
    }

    Indices boundary_indices;
    GraphXY boundary_line;

    switch (dim_enum)
    {
        case Dimensionality::Three:
            std::tie(boundary_indices, boundary_line) = 
                get_boundary_atoms(x, smooth, r0, mfrac, ball_radius);
            break;

        case Dimensionality::Two:
            std::tie(boundary_indices, boundary_line) = 
                get_boundary_atoms_2d(x, smooth, r0, box_size, mfrac, ball_radius);
            break;
    }

    const auto boundary_mols = get_molecules(boundary_indices, atom2mol);

    // Molecules cannot both be at the (upper) boundary and at the bottom: 
    // Do a set difference for the boundary!
    std::set<size_t> final_mols;
    std::set_difference(
        boundary_mols.cbegin(), boundary_mols.cend(),
        bottom_mols.cbegin(), bottom_mols.cend(),
        std::inserter(final_mols, final_mols.begin())
    );

    const auto all_boundary_indices = molecule_to_atom_indices(final_mols, &top);

    write_index_file(opt2fn("-oi", NFILE, fnm), all_boundary_indices, "contact_line");
    write_index_file(opt2fn("-ob", NFILE, fnm), final_bottom_indices, "bottom");

    return 0;
}
