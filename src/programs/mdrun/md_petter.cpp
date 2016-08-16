#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gromacs/fileio/filenm.h"
#include "gromacs/legacyheaders/sim_util.h"
#include "gromacs/legacyheaders/typedefs.h"
#include "gromacs/legacyheaders/types/commrec.h"
#include "gromacs/legacyheaders/types/inputrec.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/units.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/smalloc.h"

#include "md_petter.h"


t_flow_container *
get_flow_container(const t_commrec  *cr,
                   const int         nfile,
                   const t_filenm    fnm[],
                   const t_inputrec *ir,
                   const t_state    *state)
{
    int     var,
            step_collect = ir->userint1,
            step_output  = ir->userint2,
            num_bins[ni] = {
                ir->userint3,
                ir->userint4
            };

    float   bin_size[ni] = {
                state->box[XX][XX]/num_bins[xi],
                state->box[ZZ][ZZ]/num_bins[zi]
            },
            inv_bin[ni] = {
                1/bin_size[xi],
                1/bin_size[zi]
            };

    t_flow_container *flowcr;

    // Control userargs, although this should be done during pre-processing
    if (num_bins[xi] <= 0 || num_bins[zi] <= 0)
    {
        gmx_fatal(FARGS,
                  "Number of bins in x (userint3, %d) and z (userint4, %d) "
                  "for flow data calculation and output must be larger than 0.",
                  num_bins[xi], num_bins[zi]);
    }

    if (step_collect <= 0 || step_output <= 0)
    {
        gmx_fatal(FARGS,
                  "Number of steps that elapse between collection (userint1, %d) "
                  "and output (userint2, %d) of flow data must be larger than 0.",
                  step_collect, step_output);
    }
    else if (step_collect > step_output)
    {
        gmx_fatal(FARGS,
                  "Number of steps elapsing between output (userint2, %d) "
                  "must be larger than steps between collection (userint1, %d).",
                  step_output, step_collect, step_output);
    }
    else if (step_output % step_collect != 0)
    {
        int new_step_output = round(step_output/step_collect)*step_collect;
        gmx_warning("Steps for outputting flow data (userint2, %d) not "
                    "multiple of steps for collecting (userint1, %d). "
                    "Setting number of steps that elapse between output to %d.",
                    step_output, step_collect, new_step_output);
        step_output = new_step_output;
    }

    // Allocate memory for flow data
    snew(flowcr, sizeof(t_flow_container));

    // Print output information to user
    if (MASTER(cr))
    {
        fprintf(stderr,
                "\nData for flow field maps will be collected every %g ps "
                "(%d steps).\n", step_collect*ir->delta_t, step_collect);
        fprintf(stderr,
                "It will be averaged and output to data maps every %g ps "
                "(%d steps).\n", step_output*ir->delta_t, step_output);
        fprintf(stderr,
                "The system has been divided into %d x %d bins "
                "of size %g x %g nm^2 \nin x and z.\n",
                num_bins[xi], num_bins[zi], bin_size[xi], bin_size[zi]);
        fprintf(stderr, "Have a nice day.\n\n");
    }

    // Set options to allocated struct
    for (var = 0; var < ni; var++)
    {
        flowcr->num_bins[var] = num_bins[var];
        flowcr->bin_size[var] = bin_size[var];
        flowcr->inv_bin[var]  = inv_bin[var];
    }

    flowcr->step_collect = step_collect;
    flowcr->step_output  = step_output;
    flowcr->step_ratio   = step_output/step_collect;

    // Allocate memory for data collection
    snew(flowcr->data, num_bins[xi]*num_bins[zi]*NumVar);

    // Get name base of output datamaps
    strcpy(flowcr->fnbase, opt2fn("-flow", nfile, fnm));
    flowcr->fnbase[strlen(flowcr->fnbase) - strlen(ftp2ext(efDAT)) - 1] = '\0';

    return flowcr;
}


static void
collect_flow_data(t_flow_container   *flowcr,
                  const t_commrec    *cr,
                  const t_mdatoms    *mdatoms,
                  const t_state      *state,
                  const gmx_groups_t *groups)
{
    int     i, j,
            bin,                  // Corresponding bin index in 1D *data array
            grid_position[ni];    // Current bin position in 2D grid

    double *fdata = flowcr->data; // Shorthand to the data array

    for (i = 0; i < mdatoms->homenr; i++)
    {
        if (DOMAINDECOMP(cr))
        {
            j = cr->dd->gatindex[i];
        }
        else
        {
            j = i;
        }

        // Check for match to input group
        if (ggrpnr(groups, egcUser1, j) == 0)
        {
            // Calculate atom position in data grid
            grid_position[xi] = ((int) (state->x[i][XX]*flowcr->inv_bin[xi]
                        + flowcr->num_bins[xi] - 1))
                    % flowcr->num_bins[xi];
            grid_position[zi] = ((int) (state->x[i][ZZ]*flowcr->inv_bin[zi]
                        + flowcr->num_bins[zi] - 1))
                    % flowcr->num_bins[zi];
            bin = (flowcr->num_bins[xi]*grid_position[zi] + grid_position[xi])*NumVar;

            // Add atom data to bin at index
            fdata[bin + NumAtoms] += 1;
            fdata[bin + Temp]     += mdatoms->massT[i]*norm2(state->v[i]);
            fdata[bin + Mass]     += mdatoms->massT[i];
            fdata[bin + UU]       += mdatoms->massT[i]*state->v[i][XX];
            fdata[bin + VV]       += mdatoms->massT[i]*state->v[i][ZZ];
        }
    }
}


static void
output_flow_data(const t_flow_container *flowcr,
                 const t_commrec        *cr,
                 const gmx_int64_t       step)
{
    FILE    *fp;

    char    fnout[STRLEN];             // Final file name of output map

    int     i, j,
            bin,                       // Bin position in 1d data array
            num_out;                   // Number of data map to output

    float   write_array[ni + NumVar];  // Array for output data

    double *fdata = flowcr->data;      // Shorthand to the data array

    // Reduce data from MPI processing elements
    // Raise warning if MPI_IN_PLACE does not run on platform
    if (PAR(cr))
    {
#if defined(MPI_IN_PLACE_EXISTS)
        /* Master collects data from all PE's and prints */
        MPI_Reduce(MASTER(cr) ? MPI_IN_PLACE : flowcr->data,
                MASTER(cr) ? flowcr->data : NULL,
                flowcr->num_bins[xi]*flowcr->num_bins[zi]*NumVar,
                MPI_DOUBLE, MPI_SUM, MASTERRANK(cr),
                cr->mpi_comm_mygroup);
#else
#warning "MPI_IN_PLACE not available on platform"
        MPI_Reduce(MASTER(cr) ? MPI_IN_PLACE : flowcr->data,
                MASTER(cr) ? flowcr->data : NULL,
                flowcr->num_bins[xi]*flowcr->num_bins[zi]*NumVar,
                MPI_DOUBLE, MPI_SUM, MASTERRANK(cr),
                cr->mpi_comm_mygroup);
#endif
    }

    if (MASTER(cr))
    {
        // Construct output file name
        num_out = (int) (step/flowcr->step_output);
        sprintf(fnout, "%s_%05d.%s", flowcr->fnbase, num_out, ftp2ext(efDAT));
        fp = gmx_ffopen(fnout, "wb");

        /* Calculate and output to data maps:
         *   Positions of bin centers in X and Z
         *   Number of atoms, sample average
         *   Temperature
         *   Mass, sample average
         *   Flow U and V
         *
         * Data is collected in write_array before outputting
         */
        for (i = 0; i < flowcr->num_bins[xi]; i++)
        {
            write_array[xi] = (i + 0.5)*flowcr->bin_size[xi];

            for (j = 0; j < flowcr->num_bins[zi]; j++)
            {
                write_array[zi] = (j + 0.5)*flowcr->bin_size[zi];
                bin = (i + flowcr->num_bins[xi]*j)*NumVar;

                // Sample average mass and number density
                write_array[ni + NumAtoms] = fdata[bin + NumAtoms]/flowcr->step_ratio;
                write_array[ni + Mass]     = fdata[bin + Mass]/flowcr->step_ratio;

                /* Sample average temperature */
                if (fdata[bin + NumAtoms] > 0.0)
                {
                    write_array[ni + Temp] = fdata[bin + Temp]/(2*BOLTZ*fdata[bin + NumAtoms]);
                }
                else
                {
                    write_array[ni + Temp] = 0.0;
                }

                /* Average U and V over accumulated mass in bins */
                if (fdata[bin + Mass] > 0.0)
                {
                    write_array[ni + UU] = fdata[bin + UU]/fdata[bin + Mass];
                    write_array[ni + VV] = fdata[bin + VV]/fdata[bin + Mass];
                }
                else
                {
                    write_array[ni + UU] = 0.0;
                    write_array[ni + VV] = 0.0;
                }

                fwrite(&write_array, sizeof(float), ni + NumVar, fp);
            }
        }

        gmx_ffclose(fp);
    }
}


static void
reset_flow_data(t_flow_container *flowcr)
{
    sfree(flowcr->data);
    snew(flowcr->data, flowcr->num_bins[xi]*flowcr->num_bins[zi]*NumVar);
}


void
flow_collect_or_output(t_flow_container   *flowcr,
                       const gmx_int64_t   step,
                       const t_commrec    *cr,
                       const t_inputrec   *ir,
                       const t_mdatoms    *mdatoms,
                       const t_state      *state,
                       const gmx_groups_t *groups)
{
    if (do_per_step(step, flowcr->step_collect))
    {
        collect_flow_data(flowcr, cr, mdatoms, state, groups);

        if (do_per_step(step, flowcr->step_output) && step != ir->init_step)
        {
            output_flow_data(flowcr, cr, step);
            reset_flow_data(flowcr);
        }
    }
}

