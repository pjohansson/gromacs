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

#include <algorithm> // find
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator> // end
#include <numeric> // accumulate
#include <vector>

#include "gromacs/commandline/pargs.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pbcutil/rmpbc.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/smalloc.h"

using namespace std;

#define DEBUG_CONTACTLINE

/****************************************************************************
 * This program analyzes how contact line molecules advance.                *
 * Petter Johansson, Stockholm 2017                                         *
 ****************************************************************************/

struct CLConf {
    CLConf(t_pargs pa[], const int pasize, const rvec rmin_in, const rvec rmax_in)
        :cutoff{static_cast<real>(opt2parg_real("-co", pasize, pa))},
         cutoff2{cutoff * cutoff},
         precision{static_cast<real>(opt2parg_real("-prec", pasize, pa))},
         dx{static_cast<real>(opt2parg_real("-dx", pasize, pa))},
         nmin{static_cast<int>(opt2parg_int("-nmin", pasize, pa))},
         nmax{static_cast<int>(opt2parg_int("-nmax", pasize, pa))}
    {
        copy_rvec(rmin_in, rmin);
        copy_rvec(rmax_in, rmax);
        set_search_space_limits();

        const auto stride_buf = opt2parg_int("-stride", pasize, pa);
        if (stride_buf < 1)
        {
            gmx_fatal(FARGS, "Input stride must be positive.");
        }
        stride = static_cast<unsigned int>(stride_buf);
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

    real cutoff,
         cutoff2,
         precision,
         dx;
    int nmin,
        nmax;
    unsigned int stride;
    rvec rmin,
         rmax,
         rmin_ss,
         rmax_ss;
};

struct Positions {
    Positions(const rvec *x0,
              const vector<int> input_indices)
        :indices{input_indices}
    {
        for (auto i : indices)
        {
            positions.push_back({x0[i][XX], x0[i][YY], x0[i][ZZ]});
        }
    }

    vector<int> indices;
    vector<array<real, DIM>> positions;
};

struct Interface {
    Interface(const rvec *x0,
              const vector<int> cl_indices,
              const vector<int> bottom_indices,
              const vector<int> int_indices)
        :contact_line{Positions {x0, cl_indices}},
         bottom{Positions {x0, bottom_indices}},
         interface{Positions {x0, int_indices}} {}

    Positions contact_line,
              bottom,
              interface;
};

static bool
is_inside_limits(const rvec x,
                 const rvec rmin,
                 const rvec rmax)
{
    return x[XX] >= rmin[XX] && x[XX] < rmax[XX]
        && x[YY] >= rmin[YY] && x[YY] < rmax[YY]
        && x[ZZ] >= rmin[ZZ] && x[ZZ] < rmax[ZZ];
}

static vector<int>
find_interface_indices(const rvec          *x0,
                       const int           *grpindex,
                       const int            grpsize,
                       const struct CLConf &conf,
                       const t_pbc         *pbc)
{
    // Find indices within search volume
    vector<int> search_space;
    vector<int> candidates;

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

    vector<int> interface_inds;
    rvec dx;

    for (auto i : candidates)
    {
        const auto x1 = x0[i];
        int count = 0;

        for (auto j : search_space)
        {
            if (i != j)
            {
                const auto x2 = x0[j];
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

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "Found %lu interface ", interface_inds.size());
#endif

    return interface_inds;
}

static vector<vector<int>>
slice_system_along_dir(const rvec *x0,
                       const vector<int> interface,
                       const CLConf &conf,
                       const int dir)
{
    // Identify the floor by slicing the system and finding the lowest peak
    const int num_slices = static_cast<int>(ceil((conf.rmax[dir] - conf.rmin[dir]) / conf.precision));
    const real final_slice_precision = (conf.rmax[dir] - conf.rmin[dir]) / num_slices;

    vector<vector<int>> slice_indices (num_slices);
    for (auto i : interface)
    {
        const auto x = x0[i][dir];
        const auto slice = static_cast<int>((x - conf.rmin[dir]) / final_slice_precision);
        slice_indices.at(slice).push_back(i);
    }

    return slice_indices;
}

static vector<int>
find_bottom_layer_indices(const rvec *x0,
                          const vector<int> interface,
                          const CLConf &conf)
{
    const auto slice_indices = slice_system_along_dir(x0, interface, conf, ZZ);

    // Track the maximum count, when a decrease is found the peak
    // was in the previous slice
    int prev_slice = -1;
    unsigned int max_value = 0;

    for (auto indices : slice_indices)
    {
        const auto count = indices.size();
        if (count < max_value)
        {
            break;
        }

        max_value = count;
        ++prev_slice;
    }

    return slice_indices.at(prev_slice);
}

static vector<int>
find_contact_line_indices(const rvec *x0,
                          const vector<int> bottom,
                          const CLConf &conf)
{
    // Slice the bottom along the y axis to find the minimum
    // and maximum x coordinates of the contact line
    const auto yslices = slice_system_along_dir(x0, bottom, conf, YY);
    vector<int> indices;

    for (auto slice : yslices)
    {
        auto iter = slice.cbegin();
        auto xmax = x0[*iter][XX];
        int imax = *iter;

        while (++iter != slice.cend())
        {
            const auto x = x0[*iter][XX];

            if (x > xmax)
            {
                imax = *iter;
                xmax = x;
            }
        }

        indices.push_back(imax);
    }

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "and %lu contact line atoms.\n", indices.size());
#endif

    return indices;
}

static vector<int>
find_shared_indices(const vector<int> current,
                    const vector<int> prev)
{
    vector<int> shared_indices;

    for (auto i : current)
    {
        if (find(prev.cbegin(), prev.cend(), i) != end(prev))
        {
            shared_indices.push_back(i);
        }
    }

    return shared_indices;
}

static vector<int>
contact_line_advancements(const Positions &current,
                          const Positions &previous,
                          const CLConf    &conf)
{
    vector<int> inds_advanced;

    auto cur = current.positions.cbegin();
    auto prev = previous.positions.cbegin();
    auto cur_index = current.indices.cbegin();

    while ((cur != current.positions.cend())
            && (prev != previous.positions.cend()))
    {
        const auto xcur = (*cur++)[XX];
        const auto xprev = (*prev++)[XX];

        if ((xcur - xprev) >= conf.dx)
        {
            inds_advanced.push_back(*cur_index);
        }

        ++cur_index;
    }

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "Atoms which advanced the contact line:");
    for (auto i : inds_advanced)
    {
        fprintf(stderr, " %d", i);
    }
    fprintf(stderr, "\n");
#endif

    return inds_advanced;
}

static vector<int>
at_previous_contact_line(const vector<int> indices,
                         const Interface& current,
                         const Interface& previous)
{
    // How many were at the contact line in the previous frame?
    // The first check is whether they came from there or not.
    const auto previous_contact_line = find_shared_indices(
        indices, previous.contact_line.indices);

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "Of which were at the previous contact line:");
    for (auto i : previous_contact_line)
    {
        fprintf(stderr, " %d", i);
    }
    fprintf(stderr, "\n");
#endif

    return previous_contact_line;
}

static void
analyze_advancement(const vector<size_t>& from_previous,
                    const vector<size_t>& num_advanced)
{
    const auto sum_previous = accumulate(
        from_previous.cbegin(), from_previous.cend(), 0);
    const auto sum_total = accumulate(
        num_advanced.cbegin(), num_advanced.cend(), 0);

    const auto fraction_old = static_cast<real>(sum_previous)
        / static_cast<real>(sum_total);

    fprintf(stderr, "Fraction of contact line molecules that came from the previous contact line:\n%.3f (%.3f)\n", fraction_old, 1.0 - fraction_old);
}

static void
collect_contact_lines(const char             *fn,
                      int                    *grpindex,
                      int                     grpsize,
                      struct CLConf          &conf,
                      const t_topology       *top,
                      const int               ePBC,
                      const gmx_output_env_t *oenv)
{
    int num_atoms;
    rvec *x0;
    matrix box;
    t_trxstatus *status;
    real t;

    if ((num_atoms = read_first_x(oenv, &status, fn, &t, &x0, box)) == 0)
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
    vector<size_t> num_advanced;
    vector<size_t> num_from_previous;

    // To compare against previous frames when calculating which
    // indices advanced the contact line we need to save this
    // information. A deque gives us a window of the last-to-first data.
    deque<Interface> interfaces;

#ifdef DEBUG_CONTACTLINE
    constexpr size_t MAXLEN = 80;
    auto debug_filename = new char[MAXLEN];
    auto debug_title = new char[MAXLEN];
#endif

    do
    {
        gmx_rmpbc(gpbc, num_atoms, box, x0);
        const auto interface = find_interface_indices(
            x0, grpindex, grpsize, conf, pbc);
        const auto bottom = find_bottom_layer_indices(x0, interface, conf);
        const auto contact_line = find_contact_line_indices(x0, bottom, conf);

        Interface current {x0, contact_line, bottom, interface};
        interfaces.push_back(current);

        if (interfaces.size() > conf.stride)
        {
            const auto& previous = interfaces.front();
            const auto inds_advanced = contact_line_advancements(
                current.contact_line, previous.contact_line, conf);
            const auto from_previous = at_previous_contact_line(
                inds_advanced, current, previous);

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
        snprintf(debug_filename, MAXLEN, "contact_line_%.1fps.gro", t);
        snprintf(debug_title, MAXLEN, "contact_line");
        write_sto_conf_indexed(debug_filename, debug_title,
                               &top->atoms, x0, NULL, ePBC, box,
                               contact_line.size(),
                               contact_line.data());

        snprintf(debug_filename, MAXLEN, "interface_%.1fps.gro", t);
        snprintf(debug_title, MAXLEN, "interface");
        write_sto_conf_indexed(debug_filename, debug_title,
                               &top->atoms, x0, NULL, ePBC, box,
                               interface.size(),
                               interface.data());
#endif
    }
    while (read_next_x(oenv, status, &t, x0, box));
    gmx_rmpbc_done(gpbc);

#ifdef DEBUG_CONTACTLINE
    delete[] debug_filename;
    delete[] debug_title;
#endif

    close_trj(status);
    fprintf(stderr, "\nRead %d frames from trajectory.\n", 0);
    sfree(x0);
    delete pbc;

    analyze_advancement(num_from_previous, num_advanced);
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
               stride = 1;
    static real cutoff = 1.0,
                precision = 0.3,
                dx = 0.3;

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
          "Precision of slices along y and z (nm)." },
        { "-dx", FALSE, etREAL, { &dx },
          "Minimum distance for contact line advancement along x (nm)." },
        { "-stride", FALSE, etINT, { &stride },
          "Stride between contact line comparisons." },
    };

    const char *bugs[] = {
    };

    t_filenm fnm[] = {
        { efTRX, "-f", NULL,  ffREAD },
        { efNDX, NULL, NULL,  ffOPTRD },
        { efTPR, NULL, NULL,  ffREAD },
        { efXVG, "-o", "base", ffWRITE },
    };

#define NFILE asize(fnm)

    gmx_output_env_t *oenv;
    if (!parse_common_args(&argc, argv, PCA_CAN_TIME,
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

    fprintf(stderr, "\nSelect group to collect traces for:\n");
    get_index(&top->atoms, ftp2fn_null(efNDX, NFILE, fnm),
              1, grpsizes, index, grpnames);

    struct CLConf conf {pa, asize(pa), rmin, rmax};

    collect_contact_lines(
        ftp2fn(efTRX, NFILE, fnm),
        *index, *grpsizes, conf,
        top, ePBC, oenv
    );

    sfree(index);
    sfree(grpnames);
    sfree(grpsizes);

    return 0;
}
