/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2017,2018,2019, by the GROMACS development team, led by
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
#include <vector>

#include "gromacs/commandline/pargs.h"
#include "gromacs/commandline/viewit.h"
#include "gromacs/fileio/confio.h"
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
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/smalloc.h"

/****************************************************************************/
/* This program calculates the velocity profile across the box.             */
/* Modified from gmx_density.cpp                                            */
/* Petter Johansson, Pau 2020                                               */
/****************************************************************************/

/* used for sorting the list */
static void center_coords(t_atoms* atoms, const int* index_center, int ncenter, matrix box, rvec x0[])
{
    int  i, k, m;
    real tmass, mm;
    rvec com, shift, box_center;

    tmass = 0;
    clear_rvec(com);
    for (k = 0; (k < ncenter); k++)
    {
        i = index_center[k];
        if (i >= atoms->nr)
        {
            gmx_fatal(FARGS, "Index %d refers to atom %d, which is larger than natoms (%d).", k + 1,
                      i + 1, atoms->nr);
        }
        mm = atoms->atom[i].m;
        tmass += mm;
        for (m = 0; (m < DIM); m++)
        {
            com[m] += mm * x0[i][m];
        }
    }
    for (m = 0; (m < DIM); m++)
    {
        com[m] /= tmass;
    }
    calc_box_center(ecenterDEF, box, box_center);
    rvec_sub(com, box_center, shift);

    /* Important - while the center was calculated based on a group, we should move all atoms */
    for (i = 0; (i < atoms->nr); i++)
    {
        rvec_dec(x0[i], shift);
    }
}

struct VelocityProfile {
    double slice_width;
    std::vector<std::vector<double>> profile_per_group;
};

static VelocityProfile
calc_profile(const char*             fn,
             int**                   index,
             const int               gnx[],
             const size_t            num_slices,
             t_topology*             top,
            //  const int               ePBC,
             const PbcType          *ePBC,
             const int               axis,
             const int               velaxis,
             const int               num_groups,
             const gmx_bool          bCenter,
             const int*              index_center,
             const int               ncenter,
             const gmx_bool          bRelative,
             const gmx_output_env_t* oenv,
             const char**            dens_opt)
{
    if (axis < 0 || axis >= DIM)
    {
        gmx_fatal(FARGS, "Invalid axes. Terminating\n");
    }

    if (velaxis < 0 || velaxis >= DIM)
    {
        gmx_fatal(FARGS, "Invalid velocity axes. Terminating\n");
    }

    /* Read first frame of the trajectory and set-up PBC*/
    t_trxframe   fr;
    t_trxstatus *status = nullptr;
    gmx_rmpbc_t  gpbc = nullptr;

    if (!read_first_frame(oenv, &status, fn, &fr, TRX_NEED_X | TRX_NEED_V))
    {
        gmx_fatal(FARGS, "Could not read coordinates from statusfile\n");
    }

    gpbc = gmx_rmpbc_init(&top->idef, *ePBC, top->atoms.nr);

    /* Set-up containers for collecting velocity profile data and weighting */
    std::vector<std::vector<double>> profile_per_group (
        num_groups, std::vector<double> (num_slices, 0.0));
    std::vector<std::vector<double>> sum_weight_per_group = profile_per_group;

    std::vector<double> atom_weights (top->atoms.nr, 1.0);
    if (dens_opt[0][0] == 'm')
    {
        for (size_t i = 0; i < top->atoms.nr; ++i)
        {
            atom_weights.at(i) = top->atoms.atom[i].m;
        }
    }

    /*********** Start processing trajectory ***********/
    size_t num_frames = 0;
    double sum_box_height = 0.0;

    do
    {
        gmx_rmpbc(gpbc, fr.natoms, fr.box, fr.x);

        /* Translate atoms so the com of the center-group is in the
         * box geometrical center.
         */
        if (bCenter)
        {
            center_coords(&top->atoms, index_center, ncenter, fr.box, fr.x);
        }

        const auto height = fr.box[axis][axis];
        sum_box_height += height;

        const real relative_box_height = bRelative ? 1.0 : height;
        const real slice_width = relative_box_height / static_cast<real>(num_slices);

        const real invvol = static_cast<real>(num_slices) / (fr.box[XX][XX] * fr.box[YY][YY] * fr.box[ZZ][ZZ]);

        for (size_t n = 0; n < num_groups; ++n)
        {
            for (size_t i = 0; i < gnx[n]; ++i)
            {
                const size_t atom_index = static_cast<size_t>(index[n][i]);
                real z = fr.x[atom_index][axis];

                while (z < 0)
                {
                    z += height;
                }
                while (z > height)
                {
                    z -= height;
                }

                if (bRelative)
                {
                    z = z / height;
                }

                /* Determine which slice atom is in */
                size_t slice = bCenter ?
                    static_cast<size_t>(std::floor((z - (relative_box_height / 2.0)) / slice_width) + static_cast<real>(num_slices) / 2.0) 
                    : static_cast<size_t>(std::floor(z / slice_width));

                /* Slice should already be [0, num_slices), but we just make
                 * sure we are not hit by IEEE rounding errors since we do
                 * math operations after applying PBC above.
                 */
                if (slice >= num_slices)
                {
                    slice -= num_slices;
                }

                const auto velocity = fr.v[atom_index][velaxis];

                profile_per_group.at(n).at(slice) += velocity * atom_weights[atom_index];
                sum_weight_per_group.at(n).at(slice) += atom_weights[atom_index];
            }
        }

        ++num_frames;

    } while (read_next_frame(oenv, status, &fr));

    gmx_rmpbc_done(gpbc);
    close_trx(status);

    fprintf(stderr, "\nRead %d frames from trajectory. Calculating velocity profile ... ", num_frames);

    const auto average_box_height = sum_box_height / static_cast<real>(num_frames);
    const auto average_slice_width = average_box_height / static_cast<real>(num_slices);

    for (size_t n = 0; n < num_groups; n++)
    {
        for (size_t i = 0; i < num_slices; i++)
        {
            const auto weight = sum_weight_per_group.at(n).at(i);

            if (weight != 0.0)
            {
                profile_per_group.at(n).at(i) /= weight;
            }
        }
    }

    fprintf(stderr, "Done.\n");

    return VelocityProfile {
        average_slice_width,
        profile_per_group
    };
}

static void plot_profile(const VelocityProfile&  velocity_profile,
                         const char*             fnout,
                         const int               num_slices,
                         const int               num_groups,
                         char*                   grpname[],
                         const char**            dens_opt,
                         const gmx_bool          bCenter,
                         const gmx_bool          bRelative,
                         const gmx_bool          bSymmetrize,
                         const gmx_output_env_t* oenv)
{
    const char* title = "Velocity profile";
    const char* xlabel = nullptr;
    const char* ylabel = nullptr;

    if (bCenter)
    {
        xlabel = bRelative ? "Average relative position from center (nm)"
                           : "Relative position from center (nm)";
    }
    else
    {
        xlabel = bRelative ? "Average coordinate (nm)" : "Coordinate (nm)";
    }

    switch (dens_opt[0][0])
    {
        case 'm': ylabel = "Mass averaged flow (nm ps\\S-1\\N)"; break;
        case 'n': ylabel = "Flow (nm ps\\S-1\\N)"; break;
    }

    FILE* fp = xvgropen(fnout, title, xlabel, ylabel, oenv);

    xvgr_legend(fp, num_groups, grpname, oenv);

    for (size_t slice = 0; slice < num_slices; ++slice)
    {
        const real z = bCenter ? 
            (static_cast<real>(slice) - (static_cast<real>(num_slices) / 2.0) + 0.5) * velocity_profile.slice_width
            : (static_cast<real>(slice) + 0.5) * velocity_profile.slice_width;
        fprintf(fp, "%12g  ", z);

        for (size_t n = 0; (n < num_groups); n++)
        {
            auto value = velocity_profile.profile_per_group.at(n).at(slice);

            if (bSymmetrize)
            {
                value += velocity_profile.profile_per_group.at(n).at(num_slices - slice - 1);
                value /= 2.0;
            }

            fprintf(fp, "   %12g", value);
        }

        fprintf(fp, "\n");
    }

    xvgrclose(fp);
}

int gmx_velocity_profile(int argc, char* argv[])
{
    const char* desc[] = {
        "[THISMODULE] computes the velocity across the box, using an index file.[PAR]",
        "[PAR]",

        "Option [TT]-center[tt] performs the histogram binning relative to the center",
        "of an arbitrary group, in absolute box coordinates. If you are calculating",
        "profiles along the Z axis box dimension bZ, output would be from -bZ/2 to",
        "bZ/2 if you center based on the entire system.",
        "Note that this behaviour has changed in GROMACS 5.0; earlier versions",
        "merely performed a static binning in (0,bZ) and shifted the output. Now",
        "we compute the center for each frame and bin in (-bZ/2,bZ/2).",
        "[PAR]",

        "Option [TT]-symm[tt] symmetrizes the output around the center. This will",
        "automatically turn on [TT]-center[tt] too.",

        "Option [TT]-relative[tt] performs the binning in relative instead of absolute",
        "box coordinates, and scales the final output with the average box dimension",
        "along the output axis. This can be used in combination with [TT]-center[tt].",
        "[PAR]",
    };

    gmx_output_env_t*  oenv;
    static const char* dens_opt[]  = { nullptr, "mass", "number", nullptr };
    static int         axis        = 2; /* normal to memb. default z  */
    static int         velaxis     = 0; /* parallel to memb. default x  */
    static const char* axtitle     = "Z";
    static const char* velaxtitle  = "X";
    static int         num_slices  = 100; /* nr of slices defined       */
    static int         ngrps       = 1;  /* nr. of groups              */
    static gmx_bool    bCenter     = false;
    static gmx_bool    bRelative   = false;
    static gmx_bool    bSymmetrize = false;

    t_pargs pa[] = {
        { "-axis", false, etSTR, { &axtitle },
          "Positional axis along which to measure: X, Y or Z." },
        { "-d", false, etSTR, { &velaxtitle },
          "Which axis to measure the velocity for: X, Y or Z." },
        { "-sl", false, etINT, { &num_slices }, "Divide the box in this number of slices." },
        { "-dens", false, etENUM, { dens_opt }, "Density" },
        { "-ng", false, etINT, { &ngrps }, "Number of groups of which to compute densities." },
        { "-center", false, etBOOL, { &bCenter },
          "Perform the binning relative to the center of the (changing) box. Useful for bilayers." },
        { "-relative", false, etBOOL, { &bRelative },
          "Use relative coordinates for changing boxes and scale output by average dimensions." },
        { "-symm", false, etBOOL, { &bSymmetrize },
          "Symmetrize the density along the axis, with respect to the center. Useful for bilayers." }
    };

    const char* bugs[] = { "Has not been tested for varying box sizes." };

    char*       grpname_center; /* centering group name     */
    char**      grpname;        /* groupnames                 */
    int         ncenter;        /* size of centering group    */
    int*        ngx;            /* sizes of groups            */
    t_topology* top;            /* topology               */
    // int         ePBC;
    int*        index_center; /* index for centering group  */
    int**       index;        /* indices for all groups     */

    t_filenm fnm[] = {
        { efTRN, "-f", nullptr, ffREAD },
        { efNDX, nullptr, nullptr, ffOPTRD },
        { efTPR, nullptr, nullptr, ffREAD },
        { efXVG, "-o", "velocity-profile", ffWRITE },
    };

#define NFILE asize(fnm)

    if (!parse_common_args(&argc, argv, PCA_CAN_VIEW | PCA_CAN_TIME, NFILE, fnm, asize(pa), pa,
                           asize(desc), desc, asize(bugs), bugs, &oenv))
    {
        return 0;
    }

    GMX_RELEASE_ASSERT(dens_opt[0] != nullptr, "Option setting inconsistency; dens_opt[0] is NULL");

    if (bSymmetrize && !bCenter)
    {
        fprintf(stderr, "Can not symmetrize without centering. Turning on -center\n");
        bCenter = true;
    }

    /* Calculate axis */
    axis = toupper(axtitle[0]) - 'X';
    velaxis = toupper(velaxtitle[0]) - 'X';

    PbcType *ePBC;
    top = read_top(ftp2fn(efTPR, NFILE, fnm), ePBC); /* read topology file */
    snew(grpname, ngrps);
    snew(index, ngrps);
    snew(ngx, ngrps);

    if (bCenter)
    {
        fprintf(stderr,
                "\nNote: that the center of mass is calculated inside the box without applying\n"
                "any special periodicity. If necessary, it is your responsibility to first use\n"
                "trjconv to make sure atoms in this group are placed in the right periodicity.\n\n"
                "Select the group to center density profiles around:\n");
        get_index(&top->atoms, ftp2fn_null(efNDX, NFILE, fnm), 1, &ncenter, &index_center, &grpname_center);
    }
    else
    {
        ncenter      = 0;
        index_center = nullptr;
    }

    fprintf(stderr, "\nSelect %d group%s to calculate the velocity profile for:\n", ngrps, (ngrps > 1) ? "s" : "");
    get_index(&top->atoms, ftp2fn_null(efNDX, NFILE, fnm), ngrps, ngx, index, grpname);

    const auto velocity_profile = calc_profile(
        ftp2fn(efTRN, NFILE, fnm), index, ngx, static_cast<size_t>(num_slices), top, ePBC, axis, velaxis,
        ngrps, bCenter, index_center, ncenter, bRelative, oenv, dens_opt);

    plot_profile(velocity_profile, opt2fn("-o", NFILE, fnm), num_slices, ngrps, grpname, dens_opt,
                 bCenter, bRelative, bSymmetrize, oenv);

    do_view(oenv, opt2fn("-o", NFILE, fnm), "-nxy"); /* view xvgr file */

    return 0;
}
