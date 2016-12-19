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

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <array>
#include <vector>
#include <iostream>

#include "gromacs/commandline/pargs.h"
#include "gromacs/commandline/viewit.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/gmxana/gstat.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pbcutil/rmpbc.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/smalloc.h"

using namespace std;

using RVec = array<real, DIM>;
using Frames = vector<vector<RVec>>;

struct RLim {
    bool rmin_set,
         rmax_set;
    rvec rmin,
         rmax;
};

Frames collect_positions(const char             *fn,
                         int              *grpindex,
                         int               grpsize,
                         t_topology             *top,
                         vector<real>           &times,
                         struct RLim     &rlim,
                         const int               ePBC,
                         const gmx_output_env_t *oenv)
{
    int          num_atoms;
    rvec        *x0;
    matrix       box;
    t_trxstatus *status;
    real         t;

    if (rlim.rmin_set || rlim.rmax_set)
    {
        fprintf(stderr, "\nLimits on coordinates to include were chosen.\n");
        fprintf(stderr, "Checking final frame of trajectory for indices:\n");
    }

    if ((num_atoms = read_first_x(oenv, &status, fn, &t, &x0, box)) == 0)
    {
        gmx_fatal(FARGS, "Could not read coordinates from statusfile\n");
    }

    if (rlim.rmin_set || rlim.rmax_set)
    {
        do {}
        while (read_next_x(oenv, status, &t, x0, box));

        int new_grpsize = 0;
        int *index_buf;
        snew(index_buf, grpsize);

        // Allow the use of -1.0 to not set specific limits along an axis
        rlim.rmax[XX] = (rlim.rmax[XX] <= 0.0) ? box[XX][XX] : rlim.rmax[XX];
        rlim.rmax[YY] = (rlim.rmax[YY] <= 0.0) ? box[YY][YY] : rlim.rmax[YY];
        rlim.rmax[ZZ] = (rlim.rmax[ZZ] <= 0.0) ? box[ZZ][ZZ] : rlim.rmax[ZZ];

        for (int i = 0; i < grpsize; ++i)
        {
            auto j = grpindex[i];
            if (x0[j][XX] >= rlim.rmin[XX] && x0[j][XX] <= rlim.rmax[XX]
                && x0[j][YY] >= rlim.rmin[YY] && x0[j][YY] <= rlim.rmax[YY]
                && x0[j][ZZ] >= rlim.rmin[ZZ] && x0[j][ZZ] <= rlim.rmax[ZZ])
            {
                index_buf[new_grpsize] = j;
                ++new_grpsize;
            }
        }

        sfree(grpindex);
        grpindex = index_buf;
        grpsize = new_grpsize;

        fprintf(stderr, "Done. Kept %d indices within:\n", grpsize);
        fprintf(stderr, "  X: [%8.3f, %8.3f]\n", rlim.rmin[XX], rlim.rmax[XX]);
        fprintf(stderr, "  Y: [%8.3f, %8.3f]\n", rlim.rmin[YY], rlim.rmax[YY]);
        fprintf(stderr, "  Z: [%8.3f, %8.3f]\n\n", rlim.rmin[ZZ], rlim.rmax[ZZ]);

        // Rewind doesn't work properly, instead close and reopen
        close_trj(status);
        read_first_x(oenv, &status, fn, &t, &x0, box);
    }

    Frames xs_frames;
    auto gpbc = gmx_rmpbc_init(&top->idef, ePBC, top->atoms.nr);
    do
    {
        gmx_rmpbc(gpbc, num_atoms, box, x0);

        vector<RVec> xs (grpsize);
        for (int i = 0; i < grpsize; ++i)
        {
            for (int j = 0; j < DIM; ++j)
            {
                xs[i][j] = x0[grpindex[i]][j];
            }
        }

        xs_frames.push_back(xs);
        times.push_back(t);
    }
    while (read_next_x(oenv, status, &t, x0, box));
    gmx_rmpbc_done(gpbc);

    close_trj(status);
    fprintf(stderr, "\nRead %d frames from trajectory.\n",
            static_cast<int>(xs_frames.size()));

    sfree(x0);

    return xs_frames;
}

Frames calc_relative_positions(const Frames &xs_frames)
{
    auto final_positions = xs_frames.back();
    Frames xs_relative;

    for (auto xs : xs_frames)
    {
        const int num_atoms = xs.size();
        vector<RVec> xs_buf (num_atoms, {0.0, 0.0, 0.0});

        for (int i = 0; i < num_atoms; ++i)
        {
            for (int j = 0; j < DIM; ++j)
            {
                xs_buf[i][j] = final_positions[i][j] - xs[i][j];
            }
        }

        xs_relative.push_back(xs_buf);
    }

    return xs_relative;
}

void save_traces(const Frames           &xs_frames,
                 const vector<real>     &times,
                 const char             *filename,
                 const gmx_output_env_t *oenv)
{
    const char *title = "Trace data per time";
    const char *xlabel = "Time";
    const char *ylabel = "Positions (X0 Y0 Z0 X1 Y1 Z1 ... XN YN ZN for atom N)";

    auto file = xvgropen(filename, title, xlabel, ylabel, oenv);

    //xvgr_legend(den, nr_grps, (const char**)grpname, oenv);

    auto t = times.cbegin();
    for (auto xs : xs_frames)
    {
        fprintf(file, "%12.3f", *t++);
        for (auto atom : xs)
        {
            fprintf(file, "  %8.3f  %8.3f  %8.3f", atom[XX], atom[YY], atom[ZZ]);
        }
        fprintf(file, "\n");
    }

    xvgrclose(file);
}

int gmx_density(int argc, char *argv[])
{
    const char        *desc[] = {
        "[THISMODULE] computes partial densities across the box, using an index file.[PAR]",
        "For the total density of NPT simulations, use [gmx-energy] instead.",
        "[PAR]",
        "Hello!",
        "",
    };

    static rvec rmin = {0.0, 0.0, 0.0},
                rmax = {0.0, 0.0, 0.0};

    t_pargs pa[] = {
        { "-rmin", FALSE, etRVEC, { rmin },
          "Minimum coordinate values for final atom positions to include." },
        { "-rmax", FALSE, etRVEC, { rmax },
          "Maximum coordinate values for final atom positions to include." },
    };
    const char *bugs[] = {
    };

    t_filenm fnm[] = {
        { efTRX, "-f", NULL,  ffREAD },
        { efNDX, NULL, NULL,  ffOPTRD },
        { efTPR, NULL, NULL,  ffREAD },
        { efXVG, "-o", "density", ffWRITE },
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
    auto top = read_top(ftp2fn(efTPR, NFILE, fnm), &ePBC); /* read topology file */

    char **grpnames;  /* groupnames              */
    int   *grpsizes;  /* sizes of groups         */
    int  **index;     /* indices for all groups  */

    snew(index, 1);
    snew(grpnames, 1);
    snew(grpsizes, 1);


    fprintf(stderr, "\nSelect group to collect traces for:\n");
    get_index(&top->atoms, ftp2fn_null(efNDX, NFILE, fnm),
              1, grpsizes, index, grpnames);

    struct RLim rlim;
    if (opt2parg_bSet("-rmin", asize(pa), pa))
    {
        rlim.rmin_set = true;
        for (int i = 0; i < DIM; ++i)
        {
            rlim.rmin[i] = rmin[i];
        }
    }
    else
    {
        rlim.rmin_set = false;
    }
    if (opt2parg_bSet("-rmax", asize(pa), pa))
    {
        rlim.rmax_set = true;
        for (int i = 0; i < DIM; ++i)
        {
            rlim.rmax[i] = rmax[i];
        }
    }
    else
    {
        rlim.rmax_set = false;
    }
    vector<real> times;
    const auto xs_frames = collect_positions(ftp2fn(efTRX, NFILE, fnm),
                                             *index, *grpsizes, top,
                                             times, rlim, ePBC, oenv);
    const auto xs_relative = calc_relative_positions(xs_frames);

    save_traces(xs_relative, times, opt2fn("-o", NFILE, fnm), oenv);

    return 0;
}
