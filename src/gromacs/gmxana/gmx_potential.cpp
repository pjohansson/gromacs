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

#include <cctype>
#include <cmath>
#include <cstring>
#include <numeric>

#include "gromacs/commandline/pargs.h"
#include "gromacs/commandline/viewit.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/gmxana/princ.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h"
#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/rmpbc.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/smalloc.h"

#define EPS0 8.85419E-12
#define ELC 1.60219E-19

/****************************************************************************/
/* This program calculates the electrostatic potential across the box by    */
/* determining the charge density in slices of the box and integrating these*/
/* twice.                                                                   */
/* Peter Tieleman, April 1995                                               */
/* It now also calculates electrostatic potential in spherical micelles,    */
/* using \frac{1}{r}\frac{d^2r\Psi}{r^2} = - \frac{\rho}{\epsilon_0}        */
/* This probably sucks but it seems to work.                                */
/****************************************************************************/

static int ce = 0, cb = 0;

/* this routine integrates the array data and returns the resulting array */
/* routine uses simple trapezoid rule                                     */
static void p_integrate(double *result, double data[], int ndata, double slWidth)
{
    int    i, slice;
    double sum;

    if (ndata <= 2)
    {
        fprintf(stderr, "Warning: nr of slices very small. This will result"
                "in nonsense.\n");
    }

    fprintf(stderr, "Integrating from slice %d to slice %d\n", cb, ndata-ce);

    for (slice = cb; slice < (ndata-ce); slice++)
    {
        sum = 0;
        for (i = cb; i < slice; i++)
        {
            sum += slWidth * (data[i] + 0.5 * (data[i+1] - data[i]));
        }
        result[slice] = sum;
    }
    return;
}

static void calc_potential(const char *fn, int **index, int gnx[],
                           double ***slPotential, double ***slCharge,
                           double ***slField, int *nslices,
                           const t_topology *top, int ePBC,
                           int axis, int nr_grps, double *slWidth,
                           double fudge_z, gmx_bool bSpherical, gmx_bool bCorrect,
                           const gmx_output_env_t *oenv)
{
    rvec        *x0;     /* coordinates without pbc */
    matrix       box;    /* box (3x3) */
    int          natoms; /* nr. atoms in trj */
    t_trxstatus *status;
    int          i, n,   /* loop indices */
                 teller    = 0,
                 ax1       = 0, ax2 = 0,
                 nr_frames = 0, /* number of frames */
                 slice;         /* current slice */
    double       slVolume;      /* volume of slice for spherical averaging */
    double       qsum, nn;
    real         t;
    double       z;
    rvec         xcm;
    gmx_rmpbc_t  gpbc = nullptr;

    switch (axis)
    {
        case 0:
            ax1 = 1; ax2 = 2;
            break;
        case 1:
            ax1 = 0; ax2 = 2;
            break;
        case 2:
            ax1 = 0; ax2 = 1;
            break;
        default:
            gmx_fatal(FARGS, "Invalid axes. Terminating\n");
    }

    if ((natoms = read_first_x(oenv, &status, fn, &t, &x0, box)) == 0)
    {
        gmx_fatal(FARGS, "Could not read coordinates from statusfile\n");
    }

    if (!*nslices)
    {
        *nslices = static_cast<int>(box[axis][axis] * 10.0); /* default value */

    }
    fprintf(stderr, "\nDividing the box in %d slices\n", *nslices);

    snew(*slField, nr_grps);
    snew(*slCharge, nr_grps);
    snew(*slPotential, nr_grps);

    for (i = 0; i < nr_grps; i++)
    {
        snew((*slField)[i], *nslices);
        snew((*slCharge)[i], *nslices);
        snew((*slPotential)[i], *nslices);
    }


    gpbc = gmx_rmpbc_init(&top->idef, ePBC, natoms);

    /*********** Start processing trajectory ***********/
    do
    {
        *slWidth = box[axis][axis]/(*nslices);
        teller++;
        gmx_rmpbc(gpbc, natoms, box, x0);

        /* calculate position of center of mass based on group 1 */
        calc_xcm(x0, gnx[0], index[0], top->atoms.atom, xcm, FALSE);
        svmul(-1, xcm, xcm);

        for (n = 0; n < nr_grps; n++)
        {
            /* Check whether we actually have all positions of the requested index
             * group in the trajectory file */
            if (gnx[n] > natoms)
            {
                gmx_fatal(FARGS, "You selected a group with %d atoms, but only %d atoms\n"
                          "were found in the trajectory.\n", gnx[n], natoms);
            }
            for (i = 0; i < gnx[n]; i++) /* loop over all atoms in index file */
            {
                if (bSpherical)
                {
                    rvec_add(x0[index[n][i]], xcm, x0[index[n][i]]);
                    /* only distance from origin counts, not sign */
                    slice = static_cast<int>(norm(x0[index[n][i]])/(*slWidth));

                    /* this is a nice check for spherical groups but not for
                       all water in a cubic box since a lot will fall outside
                       the sphere
                       if (slice > (*nslices))
                       {
                       fprintf(stderr,"Warning: slice = %d\n",slice);
                       }
                     */
                    (*slCharge)[n][slice] += top->atoms.atom[index[n][i]].q;
                }
                else
                {
                    z = x0[index[n][i]][axis];
                    z = z + fudge_z;
                    if (z < 0)
                    {
                        z += box[axis][axis];
                    }
                    if (z > box[axis][axis])
                    {
                        z -= box[axis][axis];
                    }
                    /* determine which slice atom is in */
                    slice                  = static_cast<int>((z / (*slWidth)));
                    (*slCharge)[n][slice] += top->atoms.atom[index[n][i]].q;
                }
            }
        }
        nr_frames++;
    }
    while (read_next_x(oenv, status, &t, x0, box));

    gmx_rmpbc_done(gpbc);

    /*********** done with status file **********/
    close_trx(status);

    /* slCharge now contains the total charge per slice, summed over all
       frames. Now divide by nr_frames and integrate twice
     */


    if (bSpherical)
    {
        fprintf(stderr, "\n\nRead %d frames from trajectory. Calculating potential"
                "in spherical coordinates\n", nr_frames);
    }
    else
    {
        fprintf(stderr, "\n\nRead %d frames from trajectory. Calculating potential\n",
                nr_frames);
    }

    for (n = 0; n < nr_grps; n++)
    {
        for (i = 0; i < *nslices; i++)
        {
            if (bSpherical)
            {
                /* charge per volume is now the summed charge, divided by the nr
                   of frames and by the volume of the slice it's in, 4pi r^2 dr
                 */
                slVolume = 4*M_PI * gmx::square(i) * gmx::square(*slWidth) * *slWidth;
                if (slVolume == 0)
                {
                    (*slCharge)[n][i] = 0;
                }
                else
                {
                    (*slCharge)[n][i] = (*slCharge)[n][i] / (nr_frames * slVolume);
                }
            }
            else
            {
                /* get charge per volume */
                (*slCharge)[n][i] = (*slCharge)[n][i] * (*nslices) /
                    (nr_frames * box[axis][axis] * box[ax1][ax1] * box[ax2][ax2]);
            }
        }
        /* Now we have charge densities */
    }

    if (bCorrect && !bSpherical)
    {
        for (n = 0; n < nr_grps; n++)
        {
            nn   = 0;
            qsum = 0;
            for (i = 0; i < *nslices; i++)
            {
                if (std::abs((*slCharge)[n][i]) >= GMX_DOUBLE_MIN)
                {
                    nn++;
                    qsum += (*slCharge)[n][i];
                }
            }
            qsum /= nn;
            for (i = 0; i < *nslices; i++)
            {
                if (std::abs((*slCharge)[n][i]) >= GMX_DOUBLE_MIN)
                {
                    (*slCharge)[n][i] -= qsum;
                }
            }
        }
    }

    for (n = 0; n < nr_grps; n++)
    {
        /* integrate twice to get field and potential */
        p_integrate((*slField)[n], (*slCharge)[n], *nslices, *slWidth);
    }


    if (bCorrect && !bSpherical)
    {
        for (n = 0; n < nr_grps; n++)
        {
            nn   = 0;
            qsum = 0;
            for (i = 0; i < *nslices; i++)
            {
                if (std::abs((*slCharge)[n][i]) >= GMX_DOUBLE_MIN)
                {
                    nn++;
                    qsum += (*slField)[n][i];
                }
            }
            qsum /= nn;
            for (i = 0; i < *nslices; i++)
            {
                if (std::abs((*slCharge)[n][i]) >= GMX_DOUBLE_MIN)
                {
                    (*slField)[n][i] -= qsum;
                }
            }
        }
    }

    for (n = 0; n < nr_grps; n++)
    {
        p_integrate((*slPotential)[n], (*slField)[n], *nslices, *slWidth);
    }

    /* Now correct for eps0 and in spherical case for r*/
    for (n = 0; n < nr_grps; n++)
    {
        for (i = 0; i < *nslices; i++)
        {
            if (bSpherical)
            {
                (*slPotential)[n][i] = ELC * (*slPotential)[n][i] * -1.0E9 /
                    (EPS0 * i * (*slWidth));
                (*slField)[n][i] = ELC * (*slField)[n][i] * 1E18 /
                    (EPS0 * i * (*slWidth));
            }
            else
            {
                (*slPotential)[n][i] = ELC * (*slPotential)[n][i] * -1.0E9 / EPS0;
                (*slField)[n][i]     = ELC * (*slField)[n][i] * 1E18 / EPS0;
            }
        }
    }

    sfree(x0); /* free memory used by coordinate array */
}

static void plot_potential(double *potential[], double *charge[], double *field[],
                           const char *afile, const char *bfile, const char *cfile,
                           int nslices, int nr_grps, const char *grpname[], double slWidth,
                           const gmx_output_env_t *oenv)
{
    FILE       *pot,     /* xvgr file with potential */
    *cha,                /* xvgr file with charges   */
    *fie;                /* xvgr files with fields   */
    char       buf[256]; /* for xvgr title */
    int        slice, n;

    sprintf(buf, "Electrostatic Potential");
    pot = xvgropen(afile, buf, "Box (nm)", "Potential (V)", oenv);
    xvgr_legend(pot, nr_grps, grpname, oenv);

    sprintf(buf, "Charge Distribution");
    cha = xvgropen(bfile, buf, "Box (nm)", "Charge density (q/nm\\S3\\N)", oenv);
    xvgr_legend(cha, nr_grps, grpname, oenv);

    sprintf(buf, "Electric Field");
    fie = xvgropen(cfile, buf, "Box (nm)", "Field (V/nm)", oenv);
    xvgr_legend(fie, nr_grps, grpname, oenv);

    for (slice = cb; slice < (nslices - ce); slice++)
    {
        fprintf(pot, "%20.16g  ", slice * slWidth);
        fprintf(cha, "%20.16g  ", slice * slWidth);
        fprintf(fie, "%20.16g  ", slice * slWidth);
        for (n = 0; n < nr_grps; n++)
        {
            fprintf(pot, "   %20.16g", potential[n][slice]);
            fprintf(fie, "   %20.16g", field[n][slice]/1e9); /* convert to V/nm */
            fprintf(cha, "   %20.16g", charge[n][slice]);
        }
        fprintf(pot, "\n");
        fprintf(cha, "\n");
        fprintf(fie, "\n");
    }

    xvgrclose(pot);
    xvgrclose(cha);
    xvgrclose(fie);
}

static void calc_potential_diff(const char                *fntraj,
                                const t_topology          *top,
                                std::vector<real>         &diffs,
                                std::vector<real>         &times,
                                const std::vector<size_t> &indices,
                                const rvec                 r0,
                                const rvec                 r1,
                                const gmx_output_env_t    *oenv)
{
    rvec        *x;
    real         t;
    matrix       box;
    t_trxstatus *status;

    read_first_x(oenv, &status, fntraj, &t, &x, box);

    constexpr real to_si_factor = (E_CHARGE/NANO) / (4.0*M_PI*EPSILON0_SI);

    do
    {
        double V0 = 0.0, V1 = 0.0;

        for (const auto i : indices)
        {
            const auto d0 = sqrt(distance2(r0, x[i]));
            const auto d1 = sqrt(distance2(r1, x[i]));
            const auto q = top->atoms.atom[i].q;

            V0 += q / d0;
            V1 += q / d1;
        }

        diffs.push_back(static_cast<real>((V1 - V0)) * to_si_factor);
        times.push_back(t);
    }
    while (read_next_x(oenv, status, &t, x, box));
}

static real calc_mean(const std::vector<real> &values)
{
    return std::accumulate(values.cbegin(), values.cend(), 0.0)
        / static_cast<real>(values.size());
}

static real calc_stderr(const real               mean,
                        const std::vector<real> &values)
{
    double diffsq = 0.0;

    for (const auto v : values)
    {
        diffsq += std::pow(static_cast<double>(v - mean), 2.0);
    }

    const auto N = values.size();

    return sqrt(diffsq / static_cast<real>(N - 1) / static_cast<real>(N));
}

int gmx_potential(int argc, char *argv[])
{
    const char        *desc[] = {
        "[THISMODULE] computes the electrostatical potential across the box. The potential is",
        "calculated by first summing the charges per slice and then integrating",
        "twice of this charge distribution. Periodic boundaries are not taken",
        "into account. Reference of potential is taken to be the left side of",
        "the box. It is also possible to calculate the potential in spherical",
        "coordinates as function of r by calculating a charge distribution in",
        "spherical slices and twice integrating them. epsilon_r is taken as 1,",
        "but 2 is more appropriate in many cases.",
        "[PAR]",
        "Alternatively the potential difference between two points can be",
        "calculated with the command [TT]-diff[tt]. Supply the points with",
        "[TT]-r0[tt] and [TT]-r1[tt]. The difference DV = V(r1) - V(r0) is",
        "returned."
    };
    gmx_output_env_t  *oenv;
    static int         axis       = 2;       /* normal to memb. default z  */
    static const char *axtitle    = "Z";
    static int         nslices    = 10;      /* nr of slices defined       */
    static int         ngrps      = 1;
    static gmx_bool    bSpherical = FALSE;   /* default is bilayer types   */
    static real        fudge_z    = 0;       /* translate coordinates      */
    static gmx_bool    bCorrect   = 0, bDiff = FALSE;
    static rvec        r0 = { -1.0, -1.0, -1.0 },
                       r1 = { -1.0, -1.0, -1.0 };
    t_pargs            pa []      = {
        { "-diff", FALSE, etBOOL, { &bDiff },
          "Calculate the potential difference between -r0 and -r1" },
        { "-r0",  FALSE, etRVEC, { &r0 },
          "Calculate potential difference from this point to -r1" },
        { "-r1",  FALSE, etRVEC, { &r1 },
          "Calculate potential difference from -r0 to this point" },
        { "-d",   FALSE, etSTR, {&axtitle},
          "Take the normal on the membrane in direction X, Y or Z." },
        { "-sl",  FALSE, etINT, {&nslices},
          "Calculate potential as function of boxlength, dividing the box"
          " in this number of slices." },
        { "-cb",  FALSE, etINT, {&cb},
          "Discard this number of  first slices of box for integration" },
        { "-ce",  FALSE, etINT, {&ce},
          "Discard this number of last slices of box for integration" },
        { "-tz",  FALSE, etREAL, {&fudge_z},
          "Translate all coordinates by this distance in the direction of the box" },
        { "-spherical", FALSE, etBOOL, {&bSpherical},
          "Calculate spherical thingie" },
        { "-ng",       FALSE, etINT, {&ngrps},
          "Number of groups to consider" },
        { "-correct",  FALSE, etBOOL, {&bCorrect},
          "Assume net zero charge of groups to improve accuracy" }
    };
    const char        *bugs[] = {
        "Discarding slices for integration should not be necessary."
    };

    double           **potential,              /* potential per slice        */
    **charge,                                  /* total charge per slice     */
    **field,                                   /* field per slice            */
                       slWidth;                /* width of one slice         */
    char      **grpname;                       /* groupnames                 */
    int        *ngx;                           /* sizes of groups            */
    t_topology *top;                           /* topology        */
    int         ePBC;
    int       **index;                         /* indices for all groups     */
    t_filenm    fnm[] = {                      /* files for g_order       */
        { efTRX, "-f", nullptr,  ffREAD },     /* trajectory file             */
        { efNDX, nullptr, nullptr,  ffREAD },  /* index file          */
        { efTPR, nullptr, nullptr,  ffREAD },  /* topology file               */
        { efXVG, "-o", "potential", ffWRITE }, /* xvgr output file    */
        { efXVG, "-oc", "charge", ffWRITE },   /* xvgr output file    */
        { efXVG, "-of", "field", ffWRITE },    /* xvgr output file    */
    };

#define NFILE asize(fnm)

    if (!parse_common_args(&argc, argv, PCA_CAN_VIEW | PCA_CAN_TIME,
                           NFILE, fnm, asize(pa), pa, asize(desc), desc, asize(bugs), bugs,
                           &oenv))
    {
        return 0;
    }

    /* Calculate axis */
    axis = toupper(axtitle[0]) - 'X';

    top = read_top(ftp2fn(efTPR, NFILE, fnm), &ePBC); /* read topology file */

    snew(grpname, ngrps);
    snew(index, ngrps);
    snew(ngx, ngrps);


    if (bDiff)
    {
        get_index(&top->atoms, ftp2fn_null(efNDX, NFILE, fnm), 1, ngx, index, grpname);
        const auto natoms = ngx[0];
        const auto inds = index[0];

        std::vector<size_t> indices;
        indices.reserve(natoms);

        for (size_t i = 0; i < static_cast<size_t>(natoms); ++i)
        {
            // Only charged atoms matter for the calculation
            if (top->atoms.atom[inds[i]].q != 0.0)
            {
                indices.push_back(inds[i]);
            }
        }

        std::vector<real> diffs, times;

        calc_potential_diff(
            ftp2fn(efTRX, NFILE, fnm), top,
            diffs, times,
            indices, r0, r1,
            oenv
        );

        const auto mean = calc_mean(diffs);
        const auto std = calc_stderr(mean, diffs);

        FILE *fp = xvgropen_type(
            opt2fn("-o", NFILE, fnm),
            "Potential difference", "t (ps)", "\\DeltaV (V)",
            exvggtNONE, oenv
        );

        for (size_t i = 0; i < times.size(); ++i)
        {
            fprintf(fp, "%12g  %12g\n", times.at(i), diffs.at(i));
        }

        xvgrclose(fp);

        fprintf(stderr, "Mean difference: %12g +/- %-6g V\n", mean, std);
        do_view(oenv, opt2fn("-o", NFILE, fnm), nullptr);
    }
    else
    {
        rd_index(ftp2fn(efNDX, NFILE, fnm), ngrps, ngx, index, grpname);
        calc_potential(ftp2fn(efTRX, NFILE, fnm), index, ngx,
                       &potential, &charge, &field,
                       &nslices, top, ePBC, axis, ngrps, &slWidth, fudge_z,
                       bSpherical, bCorrect, oenv);

        plot_potential(potential, charge, field, opt2fn("-o", NFILE, fnm),
                       opt2fn("-oc", NFILE, fnm), opt2fn("-of", NFILE, fnm),
                       nslices, ngrps, (const char**)grpname, slWidth, oenv);

        do_view(oenv, opt2fn("-o", NFILE, fnm), nullptr);  /* view xvgr file */
        do_view(oenv, opt2fn("-oc", NFILE, fnm), nullptr); /* view xvgr file */
        do_view(oenv, opt2fn("-of", NFILE, fnm), nullptr); /* view xvgr file */
    }

    return 0;
}
