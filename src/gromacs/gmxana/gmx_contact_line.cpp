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
 * This program backtraces molecules from their final positions.            *
 * Petter Johansson, Stockholm 2016                                         *
 ****************************************************************************/

struct RLims {
    RLims(t_pargs pa[], const int pasize, const rvec rmin_in, const rvec rmax_in)
        :cutoff{static_cast<real>(opt2parg_real("-co", pasize, pa))},
         cutoff2{cutoff * cutoff},
         nmin{static_cast<int>(opt2parg_int("-nmin", pasize, pa))},
         nmax{static_cast<int>(opt2parg_int("-nmax", pasize, pa))}
        {
            copy_rvec(rmin_in, rmin);
            copy_rvec(rmax_in, rmax);
            set_search_space_limits();
        }

    void set_search_space_limits() {
        const rvec add_ss = {cutoff, cutoff, cutoff};
        rvec_sub(rmin, add_ss, rmin_ss);
        rvec_add(rmax, add_ss, rmax_ss);
    }

    real cutoff,
         cutoff2;
    int nmin,
        nmax;
    rvec rmin,
         rmax,
         rmin_ss,
         rmax_ss;
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
find_interface_indices(const rvec         *x0,
                       const int          *grpindex,
                       const int           grpsize,
                       const struct RLims &rlim,
                       const t_pbc        *pbc)
{
    // Find indices within search volume
    vector<int> search_space;
    vector<int> indices;

    for (int i = 0; i < grpsize; ++i)
    {
        const auto n = grpindex[i];
        const auto x1 = x0[n];

        if (is_inside_limits(x1, rlim.rmin_ss, rlim.rmax_ss))
        {
            search_space.push_back(n);
            if (is_inside_limits(x1, rlim.rmin, rlim.rmax))
            {
                indices.push_back(n);
            }
        }
    }

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "Kept %lu (%lu) indices within the search volume. ",
            indices.size(), search_space.size());
#endif

    vector<int> interface_inds;
    rvec dx;

    // Outer loop over all atoms which should be tested
    for (auto i : indices)
    {
        const auto x1 = x0[i];
        int count = 0;

        // Inner loop over all atoms to test against
        for (auto j : search_space)
        {
            if (i != j)
            {
                const auto x2 = x0[j];
                pbc_dx(pbc, x1, x2, dx);

                if (norm2(dx) <= rlim.cutoff2)
                {
                    ++count;
                }
            }
        }

        if (count >= rlim.nmin && count <= rlim.nmax)
        {
            interface_inds.push_back(i);
        }
    }

#ifdef DEBUG_CONTACTLINE
    fprintf(stderr, "Found %lu interface atoms.\n", interface_inds.size());
#endif

    return interface_inds;
}

static void
collect_indices(const char             *fn,
                int                    *grpindex,
                int                     grpsize,
                struct RLims           &rlim,
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

    // Allow the use of -1.0 to not set specific limits along an axis
    rlim.rmax[XX] = (rlim.rmax[XX] <= 0.0) ? box[XX][XX] : rlim.rmax[XX];
    rlim.rmax[YY] = (rlim.rmax[YY] <= 0.0) ? box[YY][YY] : rlim.rmax[YY];
    rlim.rmax[ZZ] = (rlim.rmax[ZZ] <= 0.0) ? box[ZZ][ZZ] : rlim.rmax[ZZ];
    rlim.set_search_space_limits();

    auto gpbc = gmx_rmpbc_init(&top->idef, ePBC, top->atoms.nr);
    auto pbc = new t_pbc;
    set_pbc(pbc, ePBC, box);
    do
    {
        gmx_rmpbc(gpbc, num_atoms, box, x0);
        auto indices = find_interface_indices(x0, grpindex, grpsize, rlim, pbc);

// Debug: write gro file
#ifdef DEBUG_CONTACTLINE
        string outfile { "test.gro" };
        string title { "interface" };
        write_sto_conf_indexed(outfile.data(),
                               title.data(),
                               &top->atoms,
                               x0,
                               NULL,
                               ePBC,
                               box,
                               indices.size(),
                               indices.data());

                              break;
#endif
    }
    while (read_next_x(oenv, status, &t, x0, box));
    gmx_rmpbc_done(gpbc);

    close_trj(status);
    fprintf(stderr, "\nRead %d frames from trajectory.\n",
            0);

    sfree(x0);

    return;
}

int
gmx_contact_line(int argc, char *argv[])
{
    const char        *desc[] = {
        "[THISMODULE] identifies molecules at the contact line.",
        "",
    };

    static rvec rmin = { 0.0,  0.0,  0.0},
                rmax = {-1.0, -1.0, -1.0};
    static int nmin = 20,
               nmax = 100;
    static real cutoff = 1.0;

    t_pargs            pa[] = {
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
    };

    const char        *bugs[] = {
    };

    t_filenm           fnm[] = {
        { efTRX, "-f", NULL,  ffREAD },
        { efNDX, NULL, NULL,  ffOPTRD },
        { efTPR, NULL, NULL,  ffREAD },
        { efXVG, "-o", "base", ffWRITE },
    };

#define NFILE asize(fnm)

    gmx_output_env_t  *oenv;
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

    struct RLims rlim {pa, asize(pa), rmin, rmax};

    collect_indices(ftp2fn(efTRX, NFILE, fnm),
                   *index, *grpsizes, rlim,
                    top, ePBC, oenv);

    sfree(index);
    sfree(grpnames);
    sfree(grpsizes);

    return 0;
}
