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

// Calculate the radial components (z, r) of the dipole moment
// in a grid of input spacing.
static ComponentGrid do_dip(const t_topology *top,
                            int               ePBC,
                            const char       *fn,
                            const int         axis,
                            const rvec        origin,
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

    const auto nr = static_cast<int>(ceil(rmax / spacing));
    const auto nh = static_cast<int>(ceil(hmax / spacing));

    /* Grid is ordered as: grid[iz][ir]
       Contains: pairs of [z, r] vector components */
    ComponentGrid grid(nh, std::vector<std::array<double, 2>> (nr, {0.0, 0.0}));

    /* Start while loop over frames */
    gmx_bool bCont;
    int      timecheck = 0;
    rvec     dipole, dx, xcom;

    do
    {
        set_pbc(&pbc, ePBC, box);
        gmx_rmpbc(gpbc, natom, box, x); // Makes molecules whole

        for (int i = 0; i < gnx[0]; i++)
        {
            const int i0 = mols->index[molindex[0][i]];
            const int i1 = mols->index[molindex[0][i]+1];

            // Get the center coordinate and distance from it to the origin
            mol_center_of_mass(i0, i1, x, atom, xcom);
            pbc_dx(&pbc, xcom, origin, dx);
            const real r = sqrt(std::pow(dx[e1], 2) + std::pow(dx[e2], 2));
            const real h = xcom[axis];

            if ((r < rmax) && (h < hmax))
            {
                mol_dip(i0, i1, x, atom, dipole);

                const auto ir = static_cast<int>(floor(r / spacing));
                const auto ih = static_cast<int>(floor(h / spacing));

                const real rcomponent =
                    sqrt(std::pow(dipole[e1], 2) + std::pow(dipole[e2], 2));
                grid[ih][ir][0] += dipole[axis];
                grid[ih][ir][1] += rcomponent;
            }
        }

        bCont = read_next_x(oenv, status, &t, x, box);
        timecheck = check_times(t);
    }
    while (bCont && (timecheck == 0));

    gmx_rmpbc_done(gpbc);
    close_trx(status);

    return grid;
}

static void print_grid(const Grid<real> &grid,
                       const char*       fn,
                       const real        spacing)
{
    constexpr int MAXLEN = 256;
    FILE *fp = gmx_ffopen(fn, "w");

    fprintf(fp, "# %6s %8s %12s\n", "h (nm)", "r (nm)", "angle (rad.)");

    real h = 0.5 * spacing;

    for (const auto column : grid)
    {
        real r = 0.5 * spacing;

        for (const auto value : column)
        {
            r += spacing;
            fprintf(fp, "%8.3f %8.3f %12.6f\n", h, r, value);
        }

        h += spacing;
    }

    gmx_ffclose(fp);


    return;
}

int gmx_dipoles_radial(int argc, char *argv[])
{
    const char       *desc[] = {
        "[THISMODULE] computes the total dipole plus fluctuations of a simulation",
        "system. From this you can compute e.g. the dielectric constant for",
        "low-dielectric media.",
        "For molecules with a net charge, the net charge is subtracted at",
        "center of mass of the molecule.[PAR]",
        "The file [TT]Mtot.xvg[tt] contains the total dipole moment of a frame, the",
        "components as well as the norm of the vector.",
        "The file [TT]aver.xvg[tt] contains [CHEVRON][MAG][GRK]mu[grk][mag]^2[chevron] and [MAG][CHEVRON][GRK]mu[grk][chevron][mag]^2 during the",
        "simulation.",
        "The file [TT]dipdist.xvg[tt] contains the distribution of dipole moments during",
        "the simulation",
        "The value of [TT]-mumax[tt] is used as the highest value in the distribution graph.[PAR]",
        "Furthermore, the dipole autocorrelation function will be computed when",
        "option [TT]-corr[tt] is used. The output file name is given with the [TT]-c[tt]",
        "option.",
        "The correlation functions can be averaged over all molecules",
        "([TT]mol[tt]), plotted per molecule separately ([TT]molsep[tt])",
        "or it can be computed over the total dipole moment of the simulation box",
        "([TT]total[tt]).[PAR]",
        "Option [TT]-g[tt] produces a plot of the distance dependent Kirkwood",
        "G-factor, as well as the average cosine of the angle between the dipoles",
        "as a function of the distance. The plot also includes gOO and hOO",
        "according to Nymand & Linse, J. Chem. Phys. 112 (2000) pp 6386-6395. In the same plot, ",
        "we also include the energy per scale computed by taking the inner product of",
        "the dipoles divided by the distance to the third power.[PAR]",
        "[PAR]",
        "EXAMPLES[PAR]",
        "[TT]gmx dipoles -corr mol -P 1 -o dip_sqr -mu 2.273 -mumax 5.0[tt][PAR]",
        "This will calculate the autocorrelation function of the molecular",
        "dipoles using a first order Legendre polynomial of the angle of the",
        "dipole vector and itself a time t later. For this calculation 1001",
        "frames will be used. Further, the dielectric constant will be calculated",
        "using an [TT]-epsilonRF[tt] of infinity (default), temperature of 300 K (default) and",
        "an average dipole moment of the molecule of 2.273 (SPC). For the",
        "distribution function a maximum of 5.0 will be used."
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
          "Spacing of radial distribution function" }
    };
    int              *gnx;
    int             **grpindex;
    char            **grpname = nullptr;
    t_filenm          fnm[] = {
        { efTRX, "-f", nullptr,           ffREAD },
        { efTPR, nullptr, nullptr,           ffREAD },
        { efNDX, nullptr, nullptr,           ffOPTRD },
        { efDAT, "-o",   "dipole_angle",       ffWRITE }
    };
#define NFILE asize(fnm)
    t_topology       *top;
    int               ePBC;
    int               natoms;
    matrix            box;

    const int npargs = asize(pa);
    if (!parse_common_args(&argc, argv, PCA_CAN_TIME | PCA_CAN_VIEW,
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
            gmx_fatal(FARGS, "invalid axis '%s': must be X, Y or Z", axtitle);
    }

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
        top, ePBC, ftp2fn(efTRX, NFILE, fnm), axis, origin, spacing,
        gnx, grpindex, oenv);

    const auto angle_grid = do_angle_calc(dipole_radial_components_grid);

    print_grid(angle_grid, ftp2fn(efDAT, NFILE, fnm), spacing);

    return 0;
}
