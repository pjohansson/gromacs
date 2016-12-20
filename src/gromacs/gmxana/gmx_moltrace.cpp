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
#include <vector>

#include "gromacs/commandline/pargs.h"
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

/****************************************************************************
 * This program backtraces molecules from their final positions.            *
 * Petter Johansson, Stockholm 2016                                         *
 ****************************************************************************/

using FrameX = vector<array<real, DIM>>; // Easiest to use proper arrays for
                                         // positions. Each frame (one per time)
                                         // needs a vector of these.

struct FrameData {
    vector<FrameX> frames;
    vector<real>   times;
};

struct RLims {
    RLims(t_pargs pa[], const int pasize, const rvec rmin_in, const rvec rmax_in)
         :rmin_set{static_cast<bool>(opt2parg_bSet("-rmin", pasize, pa))},
          rmax_set{static_cast<bool>(opt2parg_bSet("-rmax", pasize, pa))}
         {
             copy_rvec(rmin_in, rmin);
             copy_rvec(rmax_in, rmax);
         }

    bool rmin_set,
         rmax_set;
    rvec rmin,
         rmax;
};

static
FrameData collect_positions(const char             *fn,
                            int                    *grpindex,
                            int                     grpsize,
                            struct RLims           &rlim,
                            const t_topology       *top,
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
        fprintf(stderr, "Reading final frame of trajectory to limit indices:\n");
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

    vector<FrameX> frames;
    vector<real> times;

    auto gpbc = gmx_rmpbc_init(&top->idef, ePBC, top->atoms.nr);
    do
    {
        gmx_rmpbc(gpbc, num_atoms, box, x0);

        FrameX xs (grpsize);
        for (int i = 0; i < grpsize; ++i)
        {
            for (int j = 0; j < DIM; ++j)
            {
                xs[i][j] = x0[grpindex[i]][j];
            }
        }

        frames.push_back(xs);
        times.push_back(t);
    }
    while (read_next_x(oenv, status, &t, x0, box));
    gmx_rmpbc_done(gpbc);

    close_trj(status);
    fprintf(stderr, "\nRead %d frames from trajectory.\n",
            static_cast<int>(frames.size()));

    sfree(x0);

    return FrameData { frames, times };
}

static
FrameData calc_relative_to_final(const FrameData &fdata)
{
    auto final_positions = fdata.frames.back();
    vector<FrameX> frames_relative;

    for (auto xs : fdata.frames)
    {
        const int num_atoms = xs.size();
        auto xs_buf = xs;

        for (int i = 0; i < num_atoms; ++i)
        {
            // The direction here is: final position -> current (in backtrace).
            // Thus we subtract the final positions.
            for (int j = 0; j < DIM; ++j)
            {
                xs_buf[i][j] -= final_positions[i][j];
            }
        }

        frames_relative.push_back(xs_buf);
    }

    vector<real> times_relative = fdata.times;
    const auto tend = fdata.times.back();

    for (auto &t : times_relative)
    {
        t -= tend;
    }

    return FrameData { frames_relative, times_relative };
}

static
void save_traces(const FrameData        &fdata,
                 const char             *filename,
                 const gmx_output_env_t *oenv)
{
    const char *title  = "Trace data per time from final positions";
    const char *xlabel = "Time (ps)";
    const char *ylabel = "Positions (X0 Y0 Z0 X1 Y1 Z1 ... XN YN ZN for atom N)";

    auto file = xvgropen(filename, title, xlabel, ylabel, oenv);
    auto t_it = fdata.times.cbegin();
    auto frame_it = fdata.frames.cbegin();

    while (frame_it != (fdata.frames.cend()-1))
    {
        fprintf(file, "%12.3f", *t_it);
        for (auto atom : *frame_it)
        {
            fprintf(file, "  %8.3f  %8.3f  %8.3f", atom[XX], atom[YY], atom[ZZ]);
        }
        fprintf(file, "\n");

        ++t_it;
        ++frame_it;
    }

    xvgrclose(file);
}

int gmx_moltrace(int argc, char *argv[])
{
    const char        *desc[] = {
        "[THISMODULE] backtraces molecules from their final position of a ",
        "trajectory and calculates the difference from that position for ",
        "every frame. In effect this will show from where a selected ",
        "molecule arrived.",
        "[PAR]",
        "The selection of molecules can be done through an index group ",
        "and/or through submitting minimum and maximum coordinates for ",
        "the final particle positions to include. This is done by supplying ",
        "[TT]-rmin[tt] and [TT]-rmax[tt] vectors. For [TT]-rmax[tt] a value ",
        "of -1.0 corresponds to using the full system length along that ",
        "dimension. The time of the final frame is the final frame read ",
        "by the trajectory: this is controlled through the flags [TT]-b[tt], ",
        "[TT]-e[tt] and [TT]-dt[tt] described below.",
        "",
    };

    static rvec rmin = { 0.0,  0.0,  0.0},
                rmax = {-1.0, -1.0, -1.0};

    t_pargs            pa[] = {
        { "-rmin", FALSE, etRVEC, { rmin },
          "Minimum coordinate values for final atom positions to include." },
        { "-rmax", FALSE, etRVEC, { rmax },
          "Maximum coordinate values for final atom positions to include." },
    };

    const char        *bugs[] = {
    };

    t_filenm           fnm[] = {
        { efTRX, "-f", NULL,  ffREAD },
        { efNDX, NULL, NULL,  ffOPTRD },
        { efTPR, NULL, NULL,  ffREAD },
        { efXVG, "-o", "trace", ffWRITE },
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

    const auto fdata = collect_positions(ftp2fn(efTRX, NFILE, fnm),
                                         *index, *grpsizes, rlim,
                                         top, ePBC, oenv);
    const auto fdata_relative = calc_relative_to_final(fdata);

    save_traces(fdata_relative, opt2fn("-o", NFILE, fnm), oenv);

    sfree(index);
    sfree(grpnames);
    sfree(grpsizes);

    return 0;
}
