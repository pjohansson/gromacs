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

#include <cmath>
#include <cstring>

#include <algorithm>
#include <iostream>

#include "gromacs/commandline/pargs.h"
#include "gromacs/commandline/viewit.h"
#include "gromacs/correlationfunctions/autocorr.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/enxio.h"
#include "gromacs/fileio/matio.h"
#include "gromacs/fileio/tpxio.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/linearalgebra/nrjac.h"
#include "gromacs/listed-forces/bonded.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vecdump.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pbcutil/rmpbc.h"
#include "gromacs/statistics/statistics.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/topology.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/binaryinformation.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/smalloc.h"

#define DEBUG

#ifdef DEBUG
#define EPRINTLN(x) std::cerr << (#x) << " = " << (x) << '\n';
#else
#define EPRINTLN(X);
#endif

// Grid is ordered as: grid[iz][ir]
template <typename T>
using Grid = std::vector<std::vector<T>>;

// Contains: pairs of [z, r] vector components
using ComponentGrid = Grid<std::array<double, 2>>;


// Calculate the radial components (vz, vr) of the dipole moment
// in a grid of input spacing. vr is positive along the increasing
// radial axis.
static ComponentGrid calc_radial(const t_topology       *top,
                                 int                     ePBC,
                                 const char             *fn,
                                 const int               axis,
                                 const rvec              origin,
                                 const bool              bUseOrigin,
                                 const real              spacing,
                                 int                    *gnx,
                                 int                    *grpindex[],
                                 const gmx_output_env_t *oenv)
{
    const auto atom = top->atoms.atom;

    t_pbc          pbc;
    t_trxframe    frame;
    t_trxstatus   *status;

    if (!read_first_frame(oenv, &status, fn, &frame, TRX_NEED_X | TRX_NEED_V))
    {
        gmx_fatal(FARGS, "Could not read trajectory");
    }

    if (!frame.bV)
    {
        gmx_fatal(FARGS, "No velocities in trajectory");
    }

    // const int natom = read_first_x(oenv, &status, fn, &t, &x, box);
    const auto gpbc = gmx_rmpbc_init(&top->idef, ePBC, frame.natoms);

    int e1, e2;
    switch (axis)
    {
        case XX:
            e1 = YY;
            e2 = ZZ;
            break;
        case YY:
            e1 = XX;
            e2 = ZZ;
            break;
        case ZZ:
            e1 = XX;
            e2 = YY;
            break;
    }

    /* Initialize 2D grid for radial distribution */
    const auto box = frame.box;
    const real rmax =
        box[e1][e1] > box[e2][e2] ? 0.5 * box[e2][e2] : 0.5 * box[e1][e1];
    const real hmax = box[axis][axis];

    const auto nr = static_cast<size_t>(ceil(rmax / spacing));
    const auto nh = static_cast<size_t>(ceil(hmax / spacing));

    /* Grid is ordered as: grid[iz][ir]
       Contains: pairs of [z, r] vector components
       counts of additions are stored separately for averaging */
    ComponentGrid grid(nh, std::vector<std::array<double, 2>> (nr, {0.0, 0.0}));
    Grid<double> mass(nh, std::vector<double> (nr, 0.0));

    /* Start while loop over frames */
    gmx_bool bCont;
    int      timecheck = 0;
    rvec     dx, r0;

    if (bUseOrigin)
    {
        r0[XX] = origin[XX];
        r0[YY] = origin[YY];
        r0[ZZ] = origin[ZZ];
    }

    const auto bHaveMass = static_cast<bool>(top->atoms.haveMass);

    do
    {
        auto& x = frame.x;
        const auto& v = frame.v;

        set_pbc(&pbc, ePBC, box);
        gmx_rmpbc(gpbc, frame.natoms, box, x); // Makes molecules whole

        if (!bUseOrigin)
        {
            r0[XX] = 0.5 * box[XX][XX];
            r0[YY] = 0.5 * box[YY][YY];
            r0[ZZ] = 0.5 * box[ZZ][ZZ];
        }

        for (int n = 0; n < gnx[0]; n++)
        {
            const auto i = grpindex[0][n];
            const auto& x0 = x[i];
            pbc_dx(&pbc, r0, x0, dx);

            const real r = sqrt(std::pow(dx[e1], 2) + std::pow(dx[e2], 2));
            const real h = x0[axis];

            if ((r < rmax) && (h < hmax))
            {
                const auto& v0 = v[i];
                const auto m = bHaveMass ? atom[i].m : 1.0;

                // Project the velocity onto the radial axis to get the radial component
                // The negation is due to dx pointing *towards* the center axis while
                // we want vr to be in the opposite direction
                const real rcomponent = -(v0[e1] * dx[e1] + v0[e2] * dx[e2]) / r;

                const auto ir = static_cast<size_t>(floor(r / spacing));
                const auto ih = static_cast<size_t>(floor(h / spacing));

                grid[ih][ir][0] += m * v0[axis];
                grid[ih][ir][1] += m * rcomponent;
                mass[ih][ir] += static_cast<double>(m);
            }
        }

        bCont = read_next_frame(oenv, status, &frame);
        timecheck = check_times(frame.time);
    }
    while (bCont && (timecheck == 0));

    for (size_t ih = 0; ih < nh; ++ih)
    {
        for (size_t ir = 0; ir < nr; ++ir)
        {
            const auto m = mass[ih][ir];

            if (m > 0.0)
            {
                auto& bin = grid[ih][ir];
                bin[0] /= m;
                bin[1] /= m;
            }
        }
    }

    gmx_rmpbc_done(gpbc);
    close_trx(status);

    return grid;
}

static void print_grid_components(const ComponentGrid &grid,
                                  const char          *fn,
                                  const real           spacing)
{
    FILE *fp = gmx_ffopen(fn, "w");

    fprintf(fp, "# %6s %8s %12s %12s\n", "r (nm)", "z (nm)", "vr (nm/ps)", "vz (nm/ps)");

    real h = 0.5 * spacing;

    for (const auto column : grid)
    {
        real r = 0.5 * spacing;

        for (const auto bin : column)
        {
            fprintf(fp, "%8.3f %8.3f %12.6f %12.6f\n", r, h, bin[1], bin[0]);
            r += spacing;
        }

        h += spacing;
    }

    gmx_ffclose(fp);

    return;
}

int gmx_radial(int argc, char *argv[])
{
    const char       *desc[] = {
        "[THISMODULE] computes the radial density distribution of velocity ",
        "in a simulation.",
        "[PAR]",
        "The distribution is calculated on a 2D grid along the radius and height from",
        "a center axis. The position of this axis is by default the center of the",
        "simulation box. It can be given manually using the [TT]-origin[tt] argument.",
        "The grid spacing is controlled using the [TT]-spacing[tt] argument.",
        "[PAR]",
        "The velocity components are written to the [TT]-o[tt] filename in (nm/ps).",
        "The output is a mass average of the flow, if masses exist in the topology."
    };
    const char       *axtitle    = "Z";
    real              spacing    = 0.1;
    rvec              origin     = {-1.0, -1.0, -1.0};
    gmx_output_env_t *oenv;
    t_pargs           pa[] = {
        { "-axis",     FALSE, etSTR, {&axtitle},
          "Set cylinder axis along X, Y or Z" },
        { "-origin",   FALSE, etRVEC, {&origin},
          "Origin of cylinder, only radial component is used" },
        { "-spacing",  FALSE, etREAL, {&spacing},
          "Spacing of radial distribution function (nm)" }
    };
    int              *gnx;
    int             **grpindex;
    char            **grpname = nullptr;
    t_filenm          fnm[] = {
        { efTRX, "-f", nullptr,              ffREAD },
        { efTPR, nullptr, nullptr,           ffREAD },
        { efNDX, nullptr, nullptr,           ffOPTRD },
        { efDAT, "-o",   "radial",    ffWRITE }
    };
#define NFILE asize(fnm)
    t_topology       *top;
    int               ePBC;
    int               natoms;
    matrix            box;

    const int npargs = asize(pa);
    if (!parse_common_args(&argc, argv, PCA_CAN_TIME,
                           NFILE, fnm, npargs, pa, asize(desc), desc, 0,
                           nullptr, &oenv))
    {
        return 0;
    }

    int axis;
    switch (axtitle[0])
    {
        case 'x':
        case 'X':
            axis = XX;
            break;
        case 'y':
        case 'Y':
            axis = YY;
            break;
        case 'z':
        case 'Z':
            axis = ZZ;
            break;
        default:
            gmx_fatal(FARGS, "Invalid axis '%s': must be X, Y or Z", axtitle);
    }

    const auto bUseOrigin = static_cast<bool>(opt2parg_bSet("-origin", npargs, pa));

    snew(top, 1);
    t_inputrec irInstance;
    t_inputrec *ir = &irInstance;
    ePBC = read_tpx_top(ftp2fn(efTPR, NFILE, fnm), ir, box,
                        &natoms, nullptr, nullptr, top);
    
    snew(gnx, 1);
    snew(grpname, 1);
    snew(grpindex, 1);
    get_index(&top->atoms, ftp2fn_null(efNDX, NFILE, fnm),
              1, gnx, grpindex, grpname);

    const auto velocity_components = calc_radial(
        top, ePBC, ftp2fn(efTRX, NFILE, fnm), axis, origin, bUseOrigin, spacing,
        gnx, grpindex, oenv);

    print_grid_components(
        velocity_components,
        opt2fn("-o", NFILE, fnm),
        spacing
    );

    return 0;
}
