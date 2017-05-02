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

#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
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
         nmin{static_cast<int>(opt2parg_int("-nmin", pasize, pa))},
         nmax{static_cast<int>(opt2parg_int("-nmax", pasize, pa))},
         stride{static_cast<int>(opt2parg_int("-stride", pasize, pa))}
    {
        copy_rvec(rmin_in, rmin);
        copy_rvec(rmax_in, rmax);
        set_search_space_limits();

        if (stride < 1)
        {
            gmx_fatal(FARGS, "Input stride must be positive.");
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

    real cutoff,
         cutoff2,
         precision;
    int nmin,
        nmax,
        stride;
    rvec rmin,
         rmax,
         rmin_ss,
         rmax_ss;
};

struct ContactLine {
    ContactLine(const rvec *x0, const vector<int> input_indices)
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

static bool
is_inside_limits(const rvec x,
                 const rvec rmin,
                 const rvec rmax)
{
    return x[XX] >= rmin[XX] && x[XX] <= rmax[XX]
        && x[YY] >= rmin[YY] && x[YY] <= rmax[YY]
        && x[ZZ] >= rmin[ZZ] && x[ZZ] <= rmax[ZZ];
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
find_bottom_layer(const rvec *x0,
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

    auto bottom = slice_indices.at(prev_slice);

    return bottom;
}

static vector<int>
find_contact_line_indices(const rvec *x0,
                          const vector<int> interface,
                          const CLConf &conf)
{
    // Identify the bottom layer, then slice it along the y axis
    // to find the minimum and maximum x coordinates of the contact line
    const auto bottom = find_bottom_layer(x0, interface, conf);
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

static vector<ContactLine>
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

    vector<ContactLine> contact_lines;

#ifdef DEBUG_CONTACTLINE
    int i = 0;
#endif

    do
    {
        gmx_rmpbc(gpbc, num_atoms, box, x0);
        const auto interface = find_interface_indices(x0, grpindex, grpsize,
                                                      conf, pbc);
        auto contact_line_inds = find_contact_line_indices(x0, interface, conf);
        ContactLine contact_line {x0, contact_line_inds};
        contact_lines.push_back(contact_line);

#ifdef DEBUG_CONTACTLINE
        if (i++ == 4)
        {
            string outfile { "test.gro" };
            string title { "interface" };
            write_sto_conf_indexed(outfile.data(), title.data(),
                                   &top->atoms, x0, NULL, ePBC, box,
                                   contact_line.indices.size(),
                                   contact_line.indices.data());
            break;
        }
#endif
    }
    while (read_next_x(oenv, status, &t, x0, box));
    gmx_rmpbc_done(gpbc);

    close_trj(status);
    fprintf(stderr, "\nRead %d frames from trajectory.\n", 0);
    sfree(x0);

    return contact_lines;
}

static vector<int>
find_new_indices(const vector<int> current,
                 const vector<int> prev)
{
    vector<int> new_indices;

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "Current:");
    for (auto i : current)
    {
        fprintf(stderr, " %d", i);
    }
    fprintf(stderr, "\nPrevious:");
    for (auto i : prev)
    {
        fprintf(stderr, " %d", i);
    }
#endif

    for (auto i : current)
    {
        bool is_in_both = false;

        for (auto j : prev)
        {
            if (i == j)
            {
                is_in_both = true;
                break;
            }
        }

        if (!is_in_both)
        {
            new_indices.push_back(i);
        }
    }
#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "\nNew:");
    for (auto i : new_indices)
    {
        fprintf(stderr, " %d", i);
    }
    fprintf(stderr, "\n");
#endif

    return new_indices;
}

static void
analyze_contact_lines(const vector<ContactLine> contact_lines,
                      const CLConf              conf)
{
    if (contact_lines.size() < (static_cast<unsigned int>(conf.stride) + 1))
    {
        gmx_fatal(FARGS, "The used stride is larger than the read contact lines.");
    }

    auto current = contact_lines.cbegin();
    auto prev = contact_lines.cbegin();

    for (auto i = 0; i < conf.stride; ++i)
    {
        ++current;
    }

    while (current != contact_lines.cend())
    {
        find_new_indices((*current).indices, (*prev).indices);
        ++current;
        ++prev;
    }
    fprintf(stderr, "\n");
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
                precision = 0.3;

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

    const auto contact_lines = collect_contact_lines(
        ftp2fn(efTRX, NFILE, fnm),
        *index, *grpsizes, conf,
        top, ePBC, oenv
    );
    analyze_contact_lines(contact_lines, conf);

    sfree(index);
    sfree(grpnames);
    sfree(grpsizes);

    return 0;
}
