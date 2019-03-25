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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>

#include <libqhullcpp/RboxPoints.h>
#include <libqhullcpp/QhullError.h>
#include <libqhullcpp/QhullQh.h>
#include <libqhullcpp/QhullFacet.h>
#include <libqhullcpp/QhullFacetList.h>
#include <libqhullcpp/QhullLinkedList.h>
#include <libqhullcpp/QhullPoint.h>
#include <libqhullcpp/QhullVertex.h>
#include <libqhullcpp/QhullVertexSet.h>
#include <libqhullcpp/Qhull.h>

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

// #define DEBUG3D

template <typename T>
using Vec2 = std::array<T, 2>;
template <typename T>
using Vec3 = std::array<T, DIM>;
using Normal = Vec3<double>;
using Point = Vec3<double>;
using UVec = Vec3<size_t>;
using VecIndices = std::vector<size_t>;

struct SphericalCap {
    SphericalCap(const real a, // base radius
                 const real h, // height above base
                 const real R, // sphere radius
                 const Vec3<real> center_input)
    :base_radius { a },
     height { h },
     sphere_radius { R },
     bottom { center_input[ZZ] + R - h },
     center { center_input } {}

    real base_radius,
         height,
         sphere_radius,
         bottom; // z position of the cap bottom in system coordinates
    Vec3<real> center;
};

struct FacetInfo {
    Normal normal;
    std::vector<Point> vertex_points;
    double area;
};

struct XLim {
    XLim (void)
    :bMin { false },
     bMax { false },
     min { -1 },
     max { -1 } {}

    bool bMin, bMax;
    real min, max;
};

static real get_x_in_box(real x, const real box)
{
    while (x >= box)
    {
        x -= box;
    }

    while (x < 0.0)
    {
        x += box;
    }

    return x;
}

static VecIndices cut_system(const rvec       *x,
                             const VecIndices &all_indices,
                             const Vec3<XLim> &xlims)
{
    VecIndices indices;
    indices.reserve(all_indices.size());

    for (const auto i : all_indices)
    {
        const auto x0 = x[i];

        bool keep = true;

        for (size_t e = 0; e < DIM; ++e)
        {
            if (!((!xlims[e].bMin || (x0[e] >= xlims[e].min))
                && (!xlims[e].bMax || (x0[e] <= xlims[e].max))))
            {
                keep = false;
            }
        }

        if (keep)
        {
            indices.push_back(i);
        }
    }

    return indices;
}

static VecIndices hit_and_count(const rvec       *x,
                                const VecIndices &all_indices,
                                const matrix      box,
                                const real        bin_size,
                                const UVec       &min_counts)
{
    VecIndices indices {all_indices};

    for (size_t e = 0; e < DIM; ++e)
    {
        const auto nbins = static_cast<size_t>(ceil(box[e][e] / bin_size));
        std::vector<size_t> counts (nbins, 0);

        // Assert that if we are limiting the coordinates in this direction,
        // that it is within the set limit
        // if (xlims[e].bMin || xlims[e].bMax)
        // {
        //     const auto xmax = xlims[e].bMax ? xlims[e].max : box[e][e];
        //
        //     for (const auto i : indices)
        //     {
        //         const auto x0 = get_x_in_box(x[i][e], box[e][e]);
        //
        //         if ((x0 >= xlims[e].min) && (x0 <= xmax))
        //         {
        //             const auto bin = static_cast<size_t>(
        //                 floor(x0 / box[e][e] * static_cast<real>(nbins)));
        //             ++counts[bin];
        //         }
        //     }
        // }
        // else
        {
            for (const auto i : indices)
            {
                const auto x0 = get_x_in_box(x[i][e], box[e][e]);

                const auto bin = static_cast<size_t>(
                    floor(x0 / box[e][e] * static_cast<real>(nbins)));
                ++counts[bin];
            }
        }

        // Select the range of indices to keep by looking for the largest
        // connected set of filled bins in this direction
        size_t ibegin_best = 0, ibegin_cur = 0,
               nmax = 0, ncur = 0;
        bool in_connected = false;

        for (size_t i = 0; i < counts.size(); ++i)
        {
            if (counts.at(i) >= min_counts[e])
            {
                // Found a new section: initialize measurements
                if (!in_connected)
                {
                    in_connected = true;
                    ibegin_cur = i;
                    ncur = 1;
                }
                // Found the next connected cell: increase counter for section
                else
                {
                    ++ncur;
                }
            }
            else
            {
                // Found the end of the current section: check and update max
                if (in_connected)
                {
                    in_connected = false;
                    if (ncur > nmax)
                    {
                        nmax = ncur;
                        ibegin_best = ibegin_cur;
                    }
                }
            }
        }

        GMX_RELEASE_ASSERT(nmax > 0, "could not detect droplet");

        const size_t imin = ibegin_best,
                     imax = ibegin_best + nmax;
        const auto xmin = bin_size * static_cast<real>(imin);
        const auto xmax = bin_size * static_cast<real>(imax);

        VecIndices keep;
        keep.reserve(indices.size());

        for (const auto i : indices)
        {
            const auto x1 = get_x_in_box(x[i][e], box[e][e]);

            if ((x1 >= xmin) && (x1 < xmax))
            {
                keep.push_back(i);
            }
        }

        indices = keep;
    }

    return indices;
}

static SphericalCap fit_spherical_cap(const rvec       *x,
                                      const VecIndices &indices)
{
    auto it = indices.cbegin();
    auto xmin = x[*it][XX], xmax = x[*it][XX],
         ymin = x[*it][YY], ymax = x[*it][YY],
         zmin = x[*it][ZZ], zmax = x[*it][ZZ];

    for (++it; it != indices.cend(); ++it)
    {
        if (x[*it][XX] < xmin)
        {
            xmin = x[*it][XX];
        }
        if (x[*it][XX] > xmax)
        {
            xmax = x[*it][XX];
        }
        if (x[*it][YY] < ymin)
        {
            ymin = x[*it][YY];
        }
        if (x[*it][YY] > ymax)
        {
            ymax = x[*it][YY];
        }
        if (x[*it][ZZ] < zmin)
        {
            zmin = x[*it][ZZ];
        }
        if (x[*it][ZZ] > zmax)
        {
            zmax = x[*it][ZZ];
        }
    }

#ifdef DEBUG3D
    std::cerr 
        << "xmin: " << xmin << ", xmax: " << xmax << '\n'
        << "ymin: " << ymin << ", ymax: " << ymax << '\n'
        << "zmin: " << zmin << ", zmax: " << zmax << '\n';
#endif

    // Calculate the (halved) maximum extent of the droplet, use the max along x and y.
    const real dx_max_half = std::max(xmax - xmin, ymax - ymin) / 2.0;

    // The height of the spherical cap.
    const real h = zmax - zmin;

    // If the contact angle is larger than 90 degrees, the droplet extent
    // along x or y represents the (double) radius r of a spherical cap. Meanwhile,
    // if the contact angle is smaller it represents the (double) base radius a.
    //
    // The former condition corresponds to the height h >= r, and the latter to h < r.
    // Thus to get the base radius a of our spherical cap-fitted droplet we check 
    // which of these holds true and read off or calculate a and r.

    real a, // spherical cap base radius 
         r; // spherical cap radius

    // First check as a safe guard if the values make sense: 
    // h should not be larger than 2 * r!
    if (h > 2.0 * dx_max_half)
    {
        r = h / 2.0;
        a = sqrt(h * (2 * r - h));

        gmx_warning(
            "When fitting a spherical cap, the height h = %f > 2 * r = %f, "
            "which is not valid for a cap. Assuming that h = 2 * r.", h, 2.0 * r
            );
    }
    // Contact angle > 90
    else if (h > dx_max_half)
    {
#ifdef DEBUG3D
        std::cerr << "theta >= 90 deg.\n"; 
#endif
        r = dx_max_half;
        a = sqrt(h * (2 * r - h));
    }
    // Contact angle < 90
    else 
    {
#ifdef DEBUG3D
        std::cerr << "theta < 90 deg.\n"; 
#endif
        a = dx_max_half;
        r = (a * a + h * h) / (2.0 * h);
    }

    const real x0 = (xmax + xmin) / 2.0;
    const real y0 = (ymax + ymin) / 2.0;
    const real z0 = zmax - r;
    const Vec3<real> center {x0, y0, z0};

#ifdef DEBUG3D
    std::cerr 
        << "a: " << a << '\n'
        << "h: " << h << '\n'
        << "R: " << r << '\n'
        << "center: " << x0 << ' ' << y0 << ' ' << z0 << '\n';
#endif

    return SphericalCap(a, h, r, center);
}

static VecIndices fine_tuning(const rvec         *x,
                              const VecIndices   &indices,
                              const SphericalCap &cap,
                              const real          boundary_width,
                              const real          dr,
                              const size_t        num)
{
    const auto& R = cap.sphere_radius;

    const real Rmax = R + boundary_width;
    const real Rmin = std::max(R - boundary_width, static_cast<real>(0.0));
    const real R2_max = Rmax * Rmax;
    const real R2_min = Rmin * Rmin;

    // Include all atoms that can be reached by the boundary layer
    // when looping through it: its search space
    const real Rmin_search_space = std::max(Rmin - dr, static_cast<real>(0.0));
    const real Rmax_search_space = Rmax + dr;
    const real R2_min_search_space = Rmin_search_space * Rmin_search_space;
    const real R2_max_search_space = Rmax_search_space * Rmax_search_space;

    const real dr2 = dr * dr;

    VecIndices droplet, boundary, boundary_neighbour_space;

    droplet.reserve(indices.size());
    boundary.reserve(droplet.size());
    boundary_neighbour_space.reserve(boundary.size());

    for (const auto& i : indices)
    {
        const real d2 = distance2(x[i], cap.center.data());

        if (d2 <= R2_min)
        {
            droplet.push_back(i);

            if (d2 >= R2_min_search_space)
            {
                boundary_neighbour_space.push_back(i);
            }
        }
        else if (d2 <= R2_max)
        {
            boundary.push_back(i);
            boundary_neighbour_space.push_back(i);
        }
        else if (d2 <= R2_max_search_space)
        {
            boundary_neighbour_space.push_back(i);
        }
    }

    // NOTE: This is O(N^2). Can it be improved, preferably without
    // doing domain decompositioning?
#pragma omp parallel for
    for (size_t i = 0; i < boundary.size(); ++i)
    {
        size_t count = num;
        const auto n = boundary.at(i);
        const auto x0 = x[n];

        for (const auto& j : boundary_neighbour_space)
        {
            const real d2 = distance2(x0, x[j]);

            if (d2 <= dr2)
            {
                --count;

                if (count == 0)
                {
// Ensure that only one thread is pushing to the vector at any point
#pragma omp critical
                    droplet.push_back(n);
                    break;
                }
            }
        }
    }

    return droplet;
}

struct DensMap {
    Vec3<double> bin_size;
    Vec2<double> origin,
                 droplet_center;
    Vec2<uint64_t> shape;
    std::vector<double> densmap;
};

static DensMap get_densmap(const rvec         *xs,
                           const VecIndices   &indices,
                           const SphericalCap &spherical_cap,
                           const t_topology   *top)
{
    const auto x0 = spherical_cap.center[XX];
    const auto y0 = spherical_cap.center[YY];
    const auto a = spherical_cap.base_radius;

    // Include indices in a small set around the center point, using a buffer
    // to include some atoms which fall outside of the base radius by a bit.
    const auto buffer = a + 3.0;
    const auto xmin = x0 - buffer;
    const auto xmax = x0 + buffer;
    const auto ymin = y0 - buffer;
    const auto ymax = y0 + buffer;

    // Include only indices in a thin slice from the minimum z point
    constexpr double dz = 0.75;
    const auto zmin = spherical_cap.center[ZZ] 
        + spherical_cap.sphere_radius 
        - spherical_cap.height;
    const auto zmax = zmin + dz;

    constexpr double bin_size = 0.1;
    const auto dx = xmax - xmin;
    const auto dy = ymax - ymin;

    const auto nx = static_cast<uint64_t>(ceil(dx / bin_size));
    const auto ny = static_cast<uint64_t>(ceil(dy / bin_size));

    const Vec2<double> origin { x0 - (0.5 * dx), y0 - (0.5 * dy) };
    const Vec2<double> center { x0, y0 };
    const Vec2<uint64_t> shape { nx, ny };

    std::vector<double> densmap(nx * ny, 0.0);

    for (const auto i : indices)
    {
        const auto z = xs[i][ZZ];

        if ((z >= zmin) && (z < zmax))
        {
            const auto x = xs[i][XX];
            const auto y = xs[i][YY];
            const auto mass = top->atoms.atom[i].m;

            if ((x >= xmin) && (x < xmax) && (y >= ymin) && (y < ymax))
            {
                const auto ix = static_cast<uint64_t>(floor((x - xmin) / bin_size));
                const auto iy = static_cast<uint64_t>(floor((y - ymin) / bin_size));
                const auto index = static_cast<size_t>((iy * nx) + ix);

                densmap.at(index) += mass;
            }
        }
    }

    return DensMap {
        {bin_size, bin_size, dz},
        origin,
        center,
        shape,
        densmap
    };
}

// In order, write the density map to an input file:
// 1. bin_size (3x double)
// 2. origin (2x double)
// 3. shape (2x uint64)
// 4. center of spherical cap (2x double)
// 5. simulation time (double)
// 6. data (NX * NY double)
static void write_densmap(const std::string &fnout,
                          const DensMap densmap,
                          const double t)
{
    auto fp = gmx_ffopen(fnout.data(), "wb");

    fwrite(densmap.bin_size.data(), sizeof(double), 3, fp);
    fwrite(densmap.origin.data(), sizeof(double), 2, fp);
    fwrite(densmap.shape.data(), sizeof(uint64_t), 2, fp);
    fwrite(densmap.droplet_center.data(), sizeof(double), 2, fp);
    fwrite(&t, sizeof(double), 1, fp);

    const auto num = static_cast<size_t>(densmap.shape[XX] * densmap.shape[YY]);
    fwrite(densmap.densmap.data(), sizeof(double), num, fp);

    gmx_ffclose(fp);
}

static std::vector<FacetInfo> get_convex_hull(const rvec       *x,
                                              const VecIndices &indices)
{

    std::vector<coordT> points;
    points.reserve(DIM * indices.size());

    for (const auto i : indices)
    {
        for (size_t e = 0; e < DIM; ++e)
        {
            points.push_back(static_cast<coordT>(x[i][e]));
        }
    }

    orgQhull::Qhull convex_hull (
        "",
        static_cast<int>(DIM),
        static_cast<int>(indices.size()),
        points.data(),
        "Qt G"
    );
    // convex_hull.outputQhull();

    std::vector<FacetInfo> facets;

    for (auto& face : convex_hull.facetList())
    {
        const auto area = face.facetArea();
        const auto n = face.hyperplane().coordinates();
        const auto normal = Normal {
            static_cast<double>(n[XX]),
            static_cast<double>(n[YY]),
            static_cast<double>(n[ZZ])
        };

        std::vector<Point> points;
        for (const auto& v : face.vertices())
        {
            points.push_back(Point {
                v.point().coordinates()[XX],
                v.point().coordinates()[YY],
                v.point().coordinates()[ZZ]
            });
        }

        facets.push_back(FacetInfo {normal, points, area});
    }

    return facets;
}

// For every facet, calculate the contact angle from its normal.
// If bZmax is true, use only facets which have any vertex point
// below zmax (in system absolute units).
static std::vector<real> calc_angle_hist(const std::vector<FacetInfo> &facets,
                                         const size_t                  nangles,
                                         const real                    amin,
                                         const real                    amax,
                                         const real                    da,
                                         const bool                    bZmax,
                                         const real                    zmax)
{
    std::vector<real> hist (nangles, 0.0);

    for (const auto& face : facets)
    {
        bool include_face = !bZmax;

        if (bZmax)
        {
            for (const auto& p : face.vertex_points)
            {
                if (p[ZZ] <= zmax)
                {
                    include_face = true;
                    break;
                }
            }
        }

        if (include_face)
        {
            const auto& n = face.normal;
            const auto angle = RAD2DEG * acos(n[ZZ] / dnorm(n.data()));

            if ((angle >= amin) && (angle <= amax))
            {
                // Use min to make the range inclusive
                const auto i = std::min(
                    static_cast<size_t>((angle - amin) / da),
                    hist.size()
                );

                hist.at(i) += face.area;
            }
        }
    }

    return hist;
}

static void add_angles_to_hist(std::vector<real>       &final_hist,
                               const std::vector<real> &current_hist)
{
    for (size_t i = 0; i < final_hist.size(); ++i)
    {
        final_hist.at(i) += current_hist.at(i);
    }
}

static std::vector<real> hist_rolling_average(const std::vector<real> &hist,
                                              const size_t             nwin)
{
    std::vector<real> result (hist.size(), 0.0);

    for (size_t i = nwin; i < hist.size() - nwin; ++i)
    {
        // Total window size is 2*n + 1, nwin is in a single direction
        for (int k = -static_cast<int>(nwin); k <= static_cast<int>(nwin); ++k)
        {
            result.at(i) += hist.at(i + k);
        }
        result.at(i) /= (2 * nwin + 1);
    }

    return result;
}

static double get_most_probable_angle(const std::vector<real> &angles,
                                      const std::vector<real> &hist)
{
    const auto it = std::max_element(hist.cbegin(), hist.cend());
    const auto imax = std::distance(hist.cbegin(), it);

    return angles.at(imax);
}

int gmx_3d_analysis(int argc, char *argv[])
{
    const char        *desc[] = {
        "[THISMODULE] computes contact angles and base radii of (3D) droplets.",
        "[PAR]",
        "The analysis follows the convex hull approach by Khalkhali et al.",
        "(J. Chem. Phys., vol. 146, pp. 114704, 2017). For each frame in the",
        "trajectory, the points making up the droplet are identified. The",
        "smallest hull which contains all points are then found using the",
        "quickhull algorithm (http://www.qhull.org/). Such a hull consists",
        "of triangular facets, where each facet has a normal. This normal",
        "is used to calculate its angle to the surface, which yields a",
        "histogram of probabilities of each angle. The most probable angle",
        "is taken as the contact angle.",
        "[PAR]",
        "Atoms which make up the droplet in each frame are identified in two",
        "steps. The first step is binning the system separately along each",
        "dimension and counting the number of atoms in each bin. The longest",
        "continuous set of bins with a number of atoms equal to or larger",
        "than a cutoff is taken as the range inside which the droplet lies.",
        "This step thus identifies all atoms inside a box, the extent of which",
        "should tightly encapsulate the droplet. The bin resolution is set",
        "with [TT]-bin[tt] and the minimum count inside each bin with ",
        "[TT]-count[tt].",
        "[PAR]",
        "The second step is removing all sparse gas atoms from the box.",
        "A spherical cap is fitted to the box dimensions (the droplet is",
        "assumed to be placed on a substrate whose normal is along the z",
        "axis). Using its radius R, all atoms inside the box are classified",
        "as: 1) Outside of the droplet if their distance r to the cap center",
        "if r > R + DR, where DR is an input approximate width of the",
        "interface. 2) Inside the droplet if r < R - DR. 3) In a boundary",
        "region if within [R - DR, R + DR]. Boundary atoms are finally",
        "classified as belonging to the droplet if they have some number n",
        "neighbours within a distance d of them. Otherwise they are removed.",
        "Doing this to boundary atoms only is done purely for computational",
        "efficiency: for every atom inside the box it would be an O(N^2)",
        "operation where N is the number of atoms in each box. As long as the",
        "droplet is roughly matching a spherical cap this identification will",
        "work. The boundary width is set using [TT]-dr[tt], the neighbouring",
        "distance with [TT]-d[tt] and the minimum number of neighbours with",
        "[TT]-nmin[tt].",
        "[PAR]",
        "When calculating the angle from the created convex hull, only facets",
        "which are at (or at least close to) the contact line should be used.",
        "By default a simple height cutoff is used. Only facets which have a",
        "vertex point within height [TT]-hmax[tt] from the convex hull bottom",
        "will be used. This behavior can be turned off using [TT]-cutz[tt].",
        "[PAR]",
        "The angle probability distribution is calculated from the remaining",
        "facets, weighed by the facet area and output averaged over all",
        "frames as a histogram to [TT]-od[tt]. The contact angle is saved",
        "per frame to [TT]-oa[tt], calculated by smoothing the individual",
        "distribution with a center-average of size (2n + 1) and taking the",
        "maximum. The window size is set using [TT]-nwin[tt]. Finally, the",
        "base radius is taken from the fitted spherical cap and saved per time",
        "to [TT]-or[tt]."
    };
    static real bin_size = 0.05,
                dr_neighbours = 0.30, boundary_width = 0.50,
                amin = 0.0, amax = 150.0, da = 1.0,
                hmax = 2.0;
    static gmx_bool bZmax = true;
    static int min_count = 20, min_neighbours = 5, nwin = 2;
    static rvec counts_dir_rvec = { 20, 20, 20 },
                rmin = { -1, -1, -1 },
                rmax = { -1, -1, -1 };

    t_pargs pa[] = {
        { "-bin", FALSE, etREAL, {&bin_size},
          "Grid size for hit-and-count binning (nm)" },
        { "-count", FALSE, etINT, {&min_count},
          "Minimum count in (1D) bins to include" },
        { "-cdir",  FALSE, etRVEC, { &counts_dir_rvec },
          "Minimum count in (1D) bins per direction" },
        { "-dr", FALSE, etREAL, {&boundary_width},
          "Width of boundary identifaction step (nm)" },
        { "-d", FALSE, etREAL, {&dr_neighbours},
          "Neighbouring search distance (nm)" },
        { "-nmin", FALSE, etINT, {&min_neighbours},
          "Minimum neighbours count" },
        { "-amin", FALSE, etREAL, {&amin},
          "Minimum angle to include (deg.)" },
        { "-amax", FALSE, etREAL, {&amax},
          "Maximum angle to include (deg.)" },
        { "-da", FALSE, etREAL, {&da},
          "Precision in angle binning (deg.)" },
        { "-cutz", FALSE, etBOOL, {&bZmax},
          "Include only facets which have a point below the maximum height" },
        { "-hmax", FALSE, etREAL, {&hmax},
          "Maximum height above bottom to include vertices from (nm)" },
        { "-nwin", FALSE, etINT, {&nwin},
          "Factor to smooth distribution with" },
        { "-rmin", FALSE, etRVEC, { &rmin },
          "Minimum limit of droplet search (-1 disables per axis)" },
        { "-rmax", FALSE, etRVEC, { &rmax },
          "Maximum limit of droplet search (-1 disables per axis)" }
    };

    FILE              *fp;
    t_trxstatus       *status;
    t_topology         top;
    int                ePBC = -1;
    rvec              *x;
    matrix             box;
    real               t, zmax = 0.0;
    char             **grpname;
    int                ngrps, anagrp, *gnx = nullptr, nindex;
    int              **ind = nullptr, *index;
    UVec               counts_dir { 0, 0, 0 };
    gmx_output_env_t  *oenv;
    t_filenm           fnm[]   = {
        { efTRX, "-f",   nullptr,       ffREAD },
        { efTPS, nullptr,   nullptr,       ffOPTRD },
        { efNDX, nullptr,   nullptr,       ffOPTRD },
        { efXVG, "-od", "distangles", ffWRITE },
        { efXVG, "-oa", "angles", ffWRITE },
        { efXVG, "-or", "radius", ffWRITE },
        { efDAT, "-dens", "densmap", ffWRITE }
    };
#define NFILE asize(fnm)
    int                npargs;

    npargs = asize(pa);

    if (!parse_common_args(&argc, argv, PCA_CAN_TIME | PCA_CAN_VIEW,
                           NFILE, fnm, npargs, pa, asize(desc), desc, 0, nullptr, &oenv))
    {
        return 0;
    }

    if (ftp2bSet(efTPS, NFILE, fnm) || !ftp2bSet(efNDX, NFILE, fnm))
    {
        read_tps_conf(ftp2fn(efTPS, NFILE, fnm), &top, &ePBC, &x, nullptr, box,
                      false);
    }

    if (opt2parg_bSet("-cdir", npargs, pa))
    {
        for (size_t e = 0; e < DIM; ++e)
        {
            counts_dir[e] = static_cast<size_t>(counts_dir_rvec[e]);
        }
    }
    else
    {
        for (size_t e = 0; e < DIM; ++e)
        {
            counts_dir[e] = min_count;
        }
    }

    Vec3<XLim> xlims;
    for (size_t e = 0; e < DIM; ++e)
    {
        if (rmin[e] >= 0.0)
        {
            xlims[e].bMin = true;
            xlims[e].min = rmin[e];
        }

        if (rmax[e] >= 0.0)
        {
            xlims[e].bMax = true;
            xlims[e].max = rmax[e];
        }
    }

    // Get the base filename for density map output, ie. without the extension.
    // We need this since we will write a separate file for every output time,
    // with the time as part of the filename.
    std::string fndensmap_begin { opt2fn("-dens", NFILE, fnm) };
    const std::string ext { ftp2ext_with_dot(efDAT) };
    const auto base_length = fndensmap_begin.size() - ext.size();
    fndensmap_begin.resize(base_length);

    // Construct the final initial name, including a separator, and the final
    // string that is appended after the time. 
    fndensmap_begin.append("_");
    const std::string fndensmap_end { "ps" + ext };

    ngrps = 1;
    fprintf(stderr, "\nSelect an analysis group\n");
    snew(gnx, ngrps);
    snew(grpname, ngrps);
    snew(ind, ngrps);
    get_index(&top.atoms, ftp2fn_null(efNDX, NFILE, fnm), ngrps, gnx, ind, grpname);
    anagrp = ngrps - 1;
    nindex = gnx[anagrp];
    index  = ind[anagrp];

    read_first_x(oenv, &status, ftp2fn(efTRX, NFILE, fnm), &t, &x, box);

    std::vector<size_t> group_indices;
    group_indices.reserve(nindex);
    for (int i = 0; i < nindex; ++i)
    {
        group_indices.push_back(index[i]);
    }

    const auto nangles = static_cast<int>((amax - amin) / da);
    GMX_RELEASE_ASSERT(
        nangles > 0, "-amin, -amax and -da are set to produce no angles"
    );

    const auto da_final = (amax - amin) / static_cast<real>(nangles);
    std::vector<real> angles;

    real a = amin;
    for (size_t i = 0; i < static_cast<size_t>(nangles); ++i)
    {
        angles.push_back(a);
        a += da_final;
    }

    std::vector<real> histogram (nangles, 0.0),
                      times, angles_per_time, radius_per_time;

    do
    {
#ifdef DEBUG3D
        std::cerr << '\n';
#endif

        const auto lim_indices = cut_system(x, group_indices, xlims);
        const auto boxed_indices = hit_and_count(
            x, lim_indices, box,
            bin_size, counts_dir
        );

        const auto fitted_cap = fit_spherical_cap(x, boxed_indices);

        const auto final_indices = fine_tuning(
            x, boxed_indices, fitted_cap,
            boundary_width, dr_neighbours, static_cast<size_t>(min_neighbours)
        );

        char buf[16];
        snprintf(buf, 16, "%09.3f", t);
        std::string fnmap_current { fndensmap_begin + buf + fndensmap_end };
        const auto densmap = get_densmap(x, final_indices, fitted_cap, &top);
        write_densmap(fnmap_current, densmap, t);

        if (bZmax)
        {
            zmax = fitted_cap.bottom + hmax;
        }

        const auto facets = get_convex_hull(x, final_indices);

        const auto current_hist = calc_angle_hist(
            facets, nangles, amin, amax, da,
            bZmax, zmax
        );
        add_angles_to_hist(histogram, current_hist);

        const auto hist_smooth = hist_rolling_average(current_hist, nwin);
        const auto best_angle = get_most_probable_angle(angles, hist_smooth);

        times.push_back(t);
        angles_per_time.push_back(best_angle);
        radius_per_time.push_back(fitted_cap.base_radius);
    }
    while (read_next_x(oenv, status, &t, x, box));
    close_trx(status);

    const auto total_weight = std::accumulate(
        histogram.cbegin(), histogram.cend(), 0.0
    );

    real pmax = 0.0;
    for (auto& p : histogram)
    {
        p *= 100.0 / total_weight;

        if (p > pmax)
        {
            pmax = p;
        }
    }

    const auto hist_smooth = hist_rolling_average(histogram, nwin);
    const auto final_best_angle = get_most_probable_angle(angles, hist_smooth);

    fprintf(stderr, "\nMeasured contact angle (deg.):\n");
    fprintf(stdout, "%-12g\n", final_best_angle);

    /* Angle distribution output */
    fp = xvgropen_type(
        opt2fn("-od", NFILE, fnm),
        "Angle distribution", "Angle (deg.)", "Probability (%)",
        exvggtXNY, oenv
    );

    xvgr_world(fp, amin, 0.0, amax, pmax * 1.25, oenv);
    xvgrLegend(fp, { "Measured", "Smoothed" }, oenv);

    for (size_t i = 0; i < histogram.size(); ++i)
    {
        fprintf(
            fp, "%12g  %12g  %12g\n",
            angles.at(i), histogram.at(i), hist_smooth.at(i)
        );
    }

    xvgrclose(fp);

    /* Radius per time output */
    fp = xvgropen_type(
        opt2fn("-or", NFILE, fnm),
        "Base radius", "t (ps)", "r (nm)",
        exvggtNONE, oenv
    );

    for (size_t i = 0; i < times.size(); ++i)
    {
        fprintf(fp, "%12g  %12g\n", times.at(i), radius_per_time.at(i));
    }

    xvgrclose(fp);

    /* Angles per time output */
    fp = xvgropen_type(
        opt2fn("-oa", NFILE, fnm),
        "Contact angle", "t (ps)", "Angle (deg.)",
        exvggtNONE, oenv
    );

    for (size_t i = 0; i < times.size(); ++i)
    {
        fprintf(fp, "%12g  %12g\n", times.at(i), angles_per_time.at(i));
    }

    xvgrclose(fp);

    do_view(oenv, opt2fn("-od", NFILE, fnm), nullptr);

    return 0;
}
