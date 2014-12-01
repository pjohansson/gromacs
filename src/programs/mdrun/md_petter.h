#ifndef MD_PETTER
#define MD_PETTER

typedef enum {
    xi,
    zi,
    ni
} Axes;

typedef enum {
    CollectFlowXX,
    CollectFlowYY,
    CollectTemp,
    CollectN,
    CollectMass,
    CollectNumvar
} CollectIndex;

typedef enum {
    OutputXX,
    OutputYY,
    OutputN,
    OutputTemp,
    OutputMass,
    OutputFlowXX,
    OutputFlowYY,
    OutputLength
} OutputIndex;

typedef enum {
    FlowUU,
    FlowVV,
    Temp,
    Num,
    Mass,
    PosXX,
    PosZZ
} Index;
static const int NumCollect = 5;
static const int OutStart = 2;
static const int NumOutput = 7;

typedef struct flowdata {
    char    fnbase[STRLEN];

    int     step_collect,
            step_output,
            step_ratio,
            num_bins[ni];   // Number of bins in X and Z

    float   bin_size[ni],    // Bin sizes in X and Z
            inv_bin_size[ni]; // Inverted bin sizes for grid calculations

    double  *data;          // A 2D grid is represented by this 1D array
} t_flowdata;

// Prepare data for flow field calculations
t_flowdata* prepare_flow_field_data(t_commrec *cr, int nfile,
        const t_filenm fnm[], t_inputrec *ir, t_state *state)
{
    int     var,
            step_collect = ir->userint1,
            step_output = ir->userint2,
            num_bins[ni] =
            {
                ir->userint3,
                ir->userint4
            };

    float   bin_size[ni] =
            {
                state->box[XX][XX]/num_bins[xi],
                state->box[ZZ][ZZ]/num_bins[zi]
            },
            inv_bin_size[ni] =
            {
                1/bin_size[xi],
                1/bin_size[zi]
            };

    // Control userargs
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
    t_flowdata *flow_data = (t_flowdata*) malloc(sizeof(t_flowdata));

    // Print output information to user
    if (MASTER(cr))
    {
        fprintf(stderr, "\nData for flow field maps will be collected every %g ps "
                "(%d steps). It will be\naveraged and output to data "
                " maps every %g ps (%d steps).\n",
                step_collect*ir->delta_t, step_collect,
                step_output*ir->delta_t, step_output);

        fprintf(stderr, "The system has been divided into %d x %d bins "
                "of sizes %g x %g nm^2 \nin x and z.\n",
               num_bins[xi], num_bins[zi], bin_size[xi], bin_size[zi]);
        fprintf(stderr, "Have a nice day.\n\n");
    }

    // Set options to allocated struct
    for (var = xi; var <= zi; var++)
    {
        flow_data->num_bins[var]     = num_bins[var];
        flow_data->bin_size[var]     = bin_size[var];
        flow_data->inv_bin_size[var]  = inv_bin_size[var];
    }

    flow_data->step_collect = step_collect;
    flow_data->step_output = step_output;
    flow_data->step_ratio = step_output/step_collect;

    // Allocate memory for data collection
    snew(flow_data->data, num_bins[xi]*num_bins[zi]*CollectNumvar);

    // Get name base of output datamaps
    strcpy(flow_data->fnbase, opt2fn("-flow", nfile, fnm));
    flow_data->fnbase[strlen(flow_data->fnbase)
        - strlen(ftp2ext(efDAT)) - 1] = '\0';

    return flow_data;
}

// Collect flow field data to the grid
void collect_flow_data(t_flowdata *flow_data, t_commrec *cr,
                       t_mdatoms *mdatoms, t_state *state,
                       gmx_groups_t *groups)
{
    int     i, j,
            array_ind,                  // Corresponding position in 1D *data
            bin_position[ni];        // Current bin position in 2D system

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
            bin_position[xi] = ((int) (state->x[i][xi]*flow_data->inv_bin_size[xi]
                        + flow_data->num_bins[xi] - 1))
                    % flow_data->num_bins[xi];
            bin_position[zi] = ((int) (state->x[i][ZZ]*flow_data->inv_bin_size[zi]
                        + flow_data->num_bins[zi] - 1))
                    % flow_data->num_bins[zi];
            array_ind = (flow_data->num_bins[xi]*bin_position[zi]
                    + bin_position[xi])*CollectNumvar;

            // Add atom data to collection
            flow_data->data[array_ind + CollectFlowXX]
                += mdatoms->massT[i]*state->v[i][xi];
            flow_data->data[array_ind + CollectFlowYY]
                += mdatoms->massT[i]*state->v[i][ZZ];
            flow_data->data[array_ind + CollectTemp]
                += mdatoms->massT[i]*norm2(state->v[i]);
            flow_data->data[array_ind + CollectN]
                += 1;
            flow_data->data[array_ind + CollectMass]
                += mdatoms->massT[i];
        }
    }
}

// Finalise and output the data to specified maps
void output_flow_data(t_flowdata *flow_data, t_commrec *cr, gmx_int64_t step)
{
    FILE    *fp;

    char    fnout[STRLEN];          // Final file name of output map

    int     i, j,
            array_ind,              // Bin position in 1d data array
            num_out;             // Number of data map to output

    float   bin_center[ni],       // Center position of bin
            bin_data[OutputLength];      // Array for output data

    // Reduce data from MPI processing elements
    // Raise warning if MPI_IN_PLACE does not run on platform
    if (PAR(cr))
    {
#if defined(MPI_IN_PLACE_EXISTS)
        /* Master collects data from all PE's and prints */
        MPI_Reduce(MASTER(cr) ? MPI_IN_PLACE : flow_data->data,
                MASTER(cr) ? flow_data->data : NULL,
                flow_data->num_bins[xi]*flow_data->num_bins[zi]*CollectNumvar,
                MPI_DOUBLE, MPI_SUM, MASTERRANK(cr),
                cr->mpi_comm_mygroup);
#else
#warning "MPI_IN_PLACE not available on platform"
        MPI_Reduce(MASTER(cr) ? MPI_IN_PLACE : flow_data->data,
                MASTER(cr) ? flow_data->data : NULL,
                flow_data->num_bins[xi]*flow_data->num_bins[zi]*CollectNumvar,
                MPI_DOUBLE, MPI_SUM, MASTERRANK(cr),
                cr->mpi_comm_mygroup);
#endif
    }

    if (MASTER(cr))
    {
        num_out = (int) (step/flow_data->step_output);

        sprintf(fnout, "%s_%05d.%s",
                flow_data->fnbase, num_out, ftp2ext(efDAT));
        fp = gmx_ffopen(fnout, "wb");

        /* Calculate and output to data maps:
         *   Positions of bin centers in X and Y
         *   Accumulated number of atoms
         *   Temperature
         *   Mass, average over steps
         *   Flow U and V in X and Y respectively
         */
        for (i = 0; i < flow_data->num_bins[xi]; i++)
        {
            bin_center[xi] = (i + 0.5)*flow_data->bin_size[xi];

            for (j = 0; j < flow_data->num_bins[zi]; j++)
            {
                bin_center[zi] = (j + 0.5)*flow_data->bin_size[zi];
                array_ind = (flow_data->num_bins[xi]*j + i)*CollectNumvar;

                /* Average U and V over accumulated
                 * mass in bins
                 */
                if (flow_data->data[array_ind + CollectMass] > 0.0)
                {
                    flow_data->data[array_ind + CollectFlowXX]
                        /= flow_data->data[array_ind + CollectMass];
                    flow_data->data[array_ind + CollectFlowYY]
                        /= flow_data->data[array_ind + CollectMass];
                }
                else
                {
                    flow_data->data[array_ind + CollectFlowXX] = 0.0;
                    flow_data->data[array_ind + CollectFlowYY] = 0.0;
                }

                /* T calculated here, output with density */
                if (flow_data->data[array_ind + CollectN] > 0)
                {
                    flow_data->data[array_ind + CollectTemp]
                        /= (2*BOLTZ*flow_data->data[array_ind + CollectN]);
                }
                else
                {
                    flow_data->data[array_ind + CollectTemp] = 0.0;
                }

                // Prepare output data array
                bin_data[OutputXX] = bin_center[xi];
                bin_data[OutputYY] = bin_center[zi];
                bin_data[OutputN] = flow_data->data[array_ind + CollectN]
                    /(flow_data->step_ratio);
                bin_data[OutputTemp] = flow_data->data[array_ind + CollectTemp];
                bin_data[OutputMass] = flow_data->data[array_ind + CollectMass]
                    /(flow_data->step_ratio);
                bin_data[OutputFlowXX] = flow_data->data[array_ind + CollectFlowXX];
                bin_data[OutputFlowYY] = flow_data->data[array_ind + CollectFlowYY];

                // Output to fp
                fwrite(&bin_data, sizeof(bin_data[0]), OutputLength,
                        fp);
            }
        }

        fclose(fp);
    }

    /* Reset calculated quantities on all nodes after output */
    for (
            i = 0;
            i < flow_data->num_bins[xi]*flow_data->num_bins[zi]*CollectNumvar;
            i++
            )
    {
        flow_data->data[i] = 0;
    }
}

// Collect and output flow field data at specified steps
void check_flow_data_out(gmx_int64_t step, t_flowdata *flow_data, t_commrec *cr,
        t_mdatoms *mdatoms, t_state *state, gmx_groups_t *groups)
{
    if (do_per_step(step, flow_data->step_collect))
    {
        collect_flow_data(flow_data, cr, mdatoms, state, groups);

        if (do_per_step(step, flow_data->step_output) && step > 0)
        {
            output_flow_data(flow_data, cr, step);
        }
    }
}

#endif
