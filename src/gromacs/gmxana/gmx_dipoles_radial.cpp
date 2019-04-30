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
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pbcutil/rmpbc.h"
#include "gromacs/statistics/statistics.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/topology.h"
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

#define e2d(x) ENM2DEBYE*(x)
#define EANG2CM  E_CHARGE*1.0e-10       /* e Angstrom to Coulomb meter */
#define CM2D  SPEED_OF_LIGHT*1.0e+24    /* Coulomb meter to Debye */

// Grid is ordered as: grid[iz][ir]
template <typename T>
using Grid = std::vector<std::vector<T>>;

// Contains: pairs of [z, r] vector components
using ComponentGrid = Grid<std::array<double, 2>>;

static void neutralize_mols(int n, int *index, const t_block *mols, t_atom *atom)
{
    double mtot, qtot;
    int    ncharged, m, a0, a1, a;

    ncharged = 0;
    for (m = 0; m < n; m++)
    {
        a0   = mols->index[index[m]];
        a1   = mols->index[index[m]+1];
        mtot = 0;
        qtot = 0;
        for (a = a0; a < a1; a++)
        {
            mtot += atom[a].m;
            qtot += atom[a].q;
        }
        /* This check is only for the count print */
        if (std::abs(qtot) > 0.01)
        {
            ncharged++;
        }
        if (mtot > 0)
        {
            /* Remove the net charge at the center of mass */
            for (a = a0; a < a1; a++)
            {
                atom[a].q -= qtot*atom[a].m/mtot;
            }
        }
    }

    if (ncharged)
    {
        printf("There are %d charged molecules in the selection,\n"
               "will subtract their charge at their center of mass\n", ncharged);
    }
}

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

static void mol_dip(int i0, int i1, rvec x[], const t_atom atom[], rvec& dipole)
{
    clear_rvec(dipole);

    for (int i = i0; (i < i1); i++)
    {
        const real q  = e2d(atom[i].q);

        for (int j = 0; (j < DIM); j++)
        {
            dipole[j] += q * x[i][j];
        }
    }
}

static void mol_center_of_mass(int k0, int k1, rvec x[], const t_atom atom[], rvec &com)
{
    clear_rvec(com);
    real masstot = 0.0;

    for (int k = k0; (k < k1); k++)
    {
        real mass = atom[k].m;
        masstot  += mass;

        for (int i = 0; (i < DIM); i++)
        {
            com[i] += mass * x[k][i];
        }
    }

    svmul(1.0 / masstot, com, com);
}

// Calculate the radial dipole angle using the dipole moment components.
// Returns values on the interval [-pi, pi].
static Grid<real> do_angle_calc(const ComponentGrid &dipole_grid)
{
    const auto nz = dipole_grid.size();
    const auto nr = dipole_grid[0].size();

    Grid<real> angle_grid (nz, std::vector<real>(nr, 0.0));

    auto angle_column = angle_grid.begin();

    for (const auto dipole_column : dipole_grid)
    {
        auto angle = (*angle_column).begin();
        for (const auto dipole : dipole_column)
        {
            const auto z = dipole[0];
            const auto r = dipole[1];

            const auto value = std::atan2(z, r);
            *angle++ = value;
        }

        ++angle_column;
    }

    return angle_grid;
}

// Calculate the radial components (vz, vr) of the dipole moment
// in a grid of input spacing. vr is positive along the increasing
// radial axis.
static ComponentGrid do_dip(const t_topology *top,
                            int               ePBC,
                            const char       *fn,
                            const int         axis,
                            const rvec        origin,
                            const bool        bUseOrigin,
                            const real        spacing,
                            int              *gnx,
                            int              *molindex[],
                            const gmx_output_env_t *oenv)
{
    const auto atom = top->atoms.atom;
    const auto mols = &(top->mols);

    real           t;
    rvec          *x;
    matrix         box;
    t_pbc          pbc;
    t_trxstatus   *status;
    const int natom = read_first_x(oenv, &status, fn, &t, &x, box);
    const auto gpbc = gmx_rmpbc_init(&top->idef, ePBC, natom);

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
    const real rmax =
        box[e1][e1] > box[e2][e2] ? 0.5 * box[e2][e2] : 0.5 * box[e1][e1];
    const real hmax = box[axis][axis];

    const auto nr = static_cast<size_t>(ceil(rmax / spacing));
    const auto nh = static_cast<size_t>(ceil(hmax / spacing));

    /* Grid is ordered as: grid[iz][ir]
       Contains: pairs of [z, r] vector components
       counts of additions are stored separately for averaging */
    ComponentGrid grid(nh, std::vector<std::array<double, 2>> (nr, {0.0, 0.0}));
    Grid<size_t> counts(nh, std::vector<size_t> (nr, 0));

    /* Start while loop over frames */
    gmx_bool bCont;
    int      timecheck = 0;
    rvec     dipole, dx, r0, xcom;

    if (bUseOrigin) 
    {
        r0[XX] = origin[XX];
        r0[YY] = origin[YY];
        r0[ZZ] = origin[ZZ];
    }

    do
    {
        set_pbc(&pbc, ePBC, box);
        gmx_rmpbc(gpbc, natom, box, x); // Makes molecules whole

        if (!bUseOrigin)
        {
            r0[XX] = 0.5 * box[XX][XX];
            r0[YY] = 0.5 * box[YY][YY];
            r0[ZZ] = 0.5 * box[ZZ][ZZ];
        }

        for (int i = 0; i < gnx[0]; i++)
        {
            const int i0 = mols->index[molindex[0][i]];
            const int i1 = mols->index[molindex[0][i]+1];

            // Get the center coordinate and distance from it to the origin
            mol_center_of_mass(i0, i1, x, atom, xcom);
            pbc_dx(&pbc, r0, xcom, dx);
            const real r = sqrt(std::pow(dx[e1], 2) + std::pow(dx[e2], 2));
            const real h = xcom[axis];

            if ((r < rmax) && (h < hmax))
            {
                mol_dip(i0, i1, x, atom, dipole);

                // Project the dipole onto the radial axis to get the radial component
                // The negation is due to dx pointing *towards* the center axis while
                // we want vr to be in the opposite direction
                const real rcomponent = -(dipole[e1] * dx[e1] + dipole[e2] * dx[e2]) / r;

                const auto ir = static_cast<size_t>(floor(r / spacing));
                const auto ih = static_cast<size_t>(floor(h / spacing));

                grid[ih][ir][0] += dipole[axis];
                grid[ih][ir][1] += rcomponent;
                counts[ih][ir] += 1;
            }
        }

        bCont = read_next_x(oenv, status, &t, x, box);
        timecheck = check_times(t);
    }
    while (bCont && (timecheck == 0));

    for (size_t ih = 0; ih < nh; ++ih)
    {
        for (size_t ir = 0; ir < nr; ++ir)
        {
            const auto count = counts[ih][ir];

            if (count > 0)
            {
                auto& bin = grid[ih][ir];
                bin[0] /= count;
                bin[1] /= count;
            }
        }
    }

    gmx_rmpbc_done(gpbc);
    close_trx(status);

    return grid;
}

template <typename T>
static Grid<T> downsample_grid_singles(const Grid<T> &grid, size_t n)
{
    if ((n <= 1) || grid.empty())
    {
        return grid;
    }
    
    const auto ni = static_cast<size_t>(grid.size() / n);
    const auto nj = static_cast<size_t>(grid[0].size() / n);
    const auto nsq = std::pow(n, 2);

    Grid<T> final(ni, std::vector<T> (nj, 0.0));

    for (size_t i = 0; i < n * ni; ++i)
    {
        for (size_t j = 0; j < n * nj; ++j)
        {
            const auto i1 = static_cast<size_t>(i / n);
            const auto j1 = static_cast<size_t>(j / n);

            final[i1][j1] += grid[i][j];
        }
    }

    for (auto& row : final)
    {
        for (auto& v : row)
        {
            v /= static_cast<double>(nsq);
        }
    }

    return final;
}

static ComponentGrid downsample_grid_components(const ComponentGrid &grid, size_t n)
{
    if ((n <= 1) || grid.empty())
    {
        return grid;
    }
    
    const auto ni = static_cast<size_t>(grid.size() / n);
    const auto nj = static_cast<size_t>(grid[0].size() / n);
    const auto nsq = std::pow(n, 2);

    ComponentGrid final(ni, std::vector<std::array<double, 2>> (nj, {0.0, 0.0}));

    for (size_t i = 0; i < n * ni; ++i)
    {
        for (size_t j = 0; j < n * nj; ++j)
        {
            const auto i1 = static_cast<size_t>(i / n);
            const auto j1 = static_cast<size_t>(j / n);

            final[i1][j1][0] += grid[i][j][0];
            final[i1][j1][1] += grid[i][j][1];
        }
    }

    for (auto& row : final)
    {
        for (auto& bin : row)
        {
            bin[0] /= static_cast<double>(nsq);
            bin[1] /= static_cast<double>(nsq);
        }
    }

    return final;
}

template <typename T>
static void print_grid_singles(const Grid<T> &grid,
                               const char    *fn,
                               const real     spacing)
{
    FILE *fp = gmx_ffopen(fn, "w");

    fprintf(fp, "# %6s %8s %12s\n", "r (nm)", "z (nm)", "angle (rad.)");

    real h = 0.5 * spacing;

    for (const auto column : grid)
    {
        real r = 0.5 * spacing;

        for (const auto value : column)
        {
            fprintf(fp, "%8.3f %8.3f %12.6f\n", r, h, value);
            r += spacing;
        }

        h += spacing;
    }

    gmx_ffclose(fp);


    return;
}

static void print_grid_components(const ComponentGrid &grid,
                                  const char          *fn,
                                  const real           spacing)
{
    FILE *fp = gmx_ffopen(fn, "w");

    fprintf(fp, "# %6s %8s %12s %12s\n", "r (nm)", "z (nm)", "vr (nm)", "vz (nm)");

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

int gmx_dipoles_radial(int argc, char *argv[])
{
    const char       *desc[] = {
        "[THISMODULE] computes the radial density distribution of dipole orientation",
        "in a simulation.",
        "[PAR]",
        "The distribution is calculated on a 2D grid along the radius and height from",
        "a center axis. The position of this axis is by default the center of the",
        "simulation box. It can be given manually using the [TT]-origin[tt] argument.",
        "The grid spacing is controlled using the [TT]-spacing[tt] argument.",
        "[PAR]",
        "The dipole angle is written to the [TT]-oa[tt] filename as radians on the",
        "interval [-pi, pi], where 0 points away from the center axis. The average dipole",
        "components are written to the [TT]-oc[tt] filename."
    };
    const char       *axtitle    = "Z";
    real              spacing    = 0.1;
    rvec              origin     = {-1.0, -1.0, -1.0};
    int               downsample_angles = 1,
                      downsample_comps = 1;
    gmx_output_env_t *oenv;
    t_pargs           pa[] = {
        { "-axis",     FALSE, etSTR, {&axtitle},
          "Set cylinder axis along X, Y or Z" },
        { "-origin",   FALSE, etRVEC, {&origin},
          "Origin of cylinder, only radial component is used" },
        { "-spacing",  FALSE, etREAL, {&spacing},
          "Spacing of radial distribution function (nm)" },
        { "-na", FALSE, etINT, {&downsample_angles},
          "Factor to downsample output [TT]-oa[tt] grid with" },
        { "-nc", FALSE, etINT, {&downsample_comps},
          "Factor to downsample output [TT]-oc[tt] grid with" }
    };
    int              *gnx;
    int             **grpindex;
    char            **grpname = nullptr;
    t_filenm          fnm[] = {
        { efTRX, "-f", nullptr,           ffREAD },
        { efTPR, nullptr, nullptr,           ffREAD },
        { efNDX, nullptr, nullptr,           ffOPTRD },
        { efDAT, "-oa",   "dipole_angle",       ffWRITE },
        { efDAT, "-oc",   "dipole_comps",       ffWRITE }
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

    if (downsample_angles < 1)
    {
        gmx_fatal(FARGS, "Downsample factor -na must be at least 1");
    }

    if (downsample_comps < 1)
    {
        gmx_fatal(FARGS, "Downsample factor -nc must be at least 1");
    }

    const auto bUseOrigin = static_cast<bool>(opt2parg_bSet("-origin", npargs, pa));

    snew(top, 1);
    ePBC = read_tpx_top(ftp2fn(efTPR, NFILE, fnm), nullptr, box,
                        &natoms, nullptr, nullptr, top);

    snew(gnx, 1);
    snew(grpname, 1);
    snew(grpindex, 1);
    get_index(&top->atoms, ftp2fn_null(efNDX, NFILE, fnm),
              1, gnx, grpindex, grpname);

    dipole_atom2molindex(&gnx[0], grpindex[0], &(top->mols));
    neutralize_mols(gnx[0], grpindex[0], &(top->mols), top->atoms.atom);

    const auto dipole_radial_components_grid = do_dip(
        top, ePBC, ftp2fn(efTRX, NFILE, fnm), axis, origin, bUseOrigin, spacing,
        gnx, grpindex, oenv);
    const auto angle_grid = do_angle_calc(dipole_radial_components_grid);

    const auto final_angles = 
        downsample_grid_singles(angle_grid, downsample_angles);
    const auto final_components_grid = 
        downsample_grid_components(dipole_radial_components_grid, downsample_comps);

    print_grid_singles(
        final_angles, 
        opt2fn("-oa", NFILE, fnm), 
        static_cast<real>(downsample_angles) * spacing
    );
    print_grid_components(
        final_components_grid, 
        opt2fn("-oc", NFILE, fnm), 
        static_cast<real>(downsample_comps) * spacing
    );

    return 0;
}
