#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "typedefs.h"
#include "gromacs/utility/smalloc.h"
#include "sysstuff.h"
#include "vec.h"
#include "vcm.h"
#include "mdebin.h"
#include "nrnb.h"
#include "calcmu.h"
#include "index.h"
#include "vsite.h"
#include "update.h"
#include "ns.h"
#include "mdrun.h"
#include "md_support.h"
#include "md_logging.h"
#include "network.h"
#include "xvgr.h"
#include "physics.h"
#include "names.h"
#include "force.h"
#include "disre.h"
#include "orires.h"
#include "pme.h"
#include "mdatoms.h"
#include "repl_ex.h"
#include "deform.h"
#include "qmmm.h"
#include "domdec.h"
#include "domdec_network.h"
#include "gromacs/gmxlib/topsort.h"
#include "coulomb.h"
#include "constr.h"
#include "shellfc.h"
#include "gromacs/gmxpreprocess/compute_io.h"
#include "checkpoint.h"
#include "mtop_util.h"
#include "sighandler.h"
#include "txtdump.h"
#include "gromacs/utility/cstringutil.h"
#include "pme_loadbal.h"
#include "bondf.h"
#include "membed.h"
#include "types/nlistheuristics.h"
#include "types/iteratedconstraints.h"
#include "nbnxn_cuda_data_mgmt.h"

#include "gromacs/utility/gmxmpi.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/trajectory_writing.h"
#include "gromacs/fileio/trnio.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xtcio.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/pulling/pull.h"
#include "gromacs/swap/swapcoords.h"
#include "gromacs/imd/imd.h"

#ifndef MD_PETTER
#define MD_PETTER

enum {
    NumBinDim = 2
};

typedef enum {
    Collect,
    Output,
    Ratio,
    NumStepVars
} DataStepIndex;

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

typedef struct flowFieldData t_flowFieldData;

struct flowFieldData {
    gmx_bool bData;                     // Flag for if -flow is set

    char    fnDataBase[STRLEN];

    int     dataStep[NumStepVars],      // Step multiples for collection and output
            numBins[NumBinDim];         // Number of bins in XX and YY

    float   binSize[NumBinDim],         // Bin sizes in XX and YY
            invBinSize[NumBinDim];      // Inverted bin sizes for grid calculations

    double  *dataGrid;                  // A 2D grid is represented by this 1D array
};

// Prepare data for flow field calculations
void prepare_flow_field_data(t_flowFieldData *flowFieldData, t_commrec *cr,
                             int nfile, const t_filenm fnm[], t_inputrec *ir,
                             t_state *state)
{
    int     var,
            dataStep[NumStepVars] =
            {
                ir->userint1,
                ir->userint2
            },
            numBins[NumBinDim] =
            {
                ir->userint3,
                ir->userint4
            };

    float   binSize[NumBinDim] =
            {
                state->box[XX][XX]/numBins[XX],
                state->box[ZZ][ZZ]/numBins[YY]
            },
            invBinSize[NumBinDim] =
            {
                1/binSize[XX],
                1/binSize[YY]
            };

    // Control userargs
    if (numBins[XX] <= 0 || numBins[YY] <= 0)
    {
        gmx_fatal(FARGS, "Number of bins in x (userint3) and z "
                "(userint4) required for flow calculations are "
                "not set in the preprocessor.");
    }

    if (dataStep[Collect] <= 0 || dataStep[Output] <= 0)
    {
        gmx_fatal(FARGS, "Steps for which multiples of data for flow "
                "field calculations will be collected (userint1) and "
                "output (userint2) on are not set in the preprocessor.");
    }
    else if (dataStep[Output] % dataStep[Collect] != 0)
    {
        gmx_fatal(FARGS, "Steps for which multiples of data for flow "
                "field calculations are output at (userint2) is required "
                "to be a multiple of steps to collect data at (userint1).");
    }
    else
    {
        dataStep[Ratio] = dataStep[Output]/dataStep[Collect];
    }

    // Set flag
    flowFieldData->bData = TRUE;

    // Print output information to user
    if (MASTER(cr))
    {
        fprintf(stderr, "\nData for flow field maps will be collected every %g ps "
                "(%d steps). It will be\naveraged and output to data "
                " maps every %g ps (%d steps).\n",
                dataStep[Collect]*ir->delta_t, dataStep[Collect],
                dataStep[Output]*ir->delta_t, dataStep[Output]);

        fprintf(stderr, "The system has been divided into %d x %d bins "
                "of sizes %g x %g nm^2 \nin x and z respectively.\n",
               numBins[XX], numBins[YY], binSize[XX], binSize[YY]);
        fprintf(stderr, "Have a nice day.\n\n");
    }

    // Set options to allocated struct
    for (var = XX; var <= YY; var++)
    {
        flowFieldData->dataStep[var]    = dataStep[var];
        flowFieldData->numBins[var]     = numBins[var];
        flowFieldData->binSize[var]     = binSize[var];
        flowFieldData->invBinSize[var]  = invBinSize[var];
    }

    flowFieldData->dataStep[Ratio] = dataStep[Ratio];

    // Allocate memory for data collection
    snew(flowFieldData->dataGrid, numBins[XX]*numBins[YY]*CollectNumvar);

    // Get name base of output datamaps
    strcpy(flowFieldData->fnDataBase, opt2fn("-flow", nfile, fnm));
    flowFieldData->fnDataBase[strlen(flowFieldData->fnDataBase)
        - strlen(ftp2ext(efDAT)) - 1] = '\0';
}

// Collect flow field data to the grid
void collect_flow_data(t_flowFieldData *flowFieldData, t_commrec *cr,
                       t_mdatoms *mdatoms, t_state *state,
                       gmx_groups_t *groups)
{
    int     i, j,
            bin1DPos,                   // Corresponding position in 1D *dataGrid
            bin2DPos[NumBinDim];        // Current bin position in 2D system

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
            // Calculate atom position in dataGrid
            bin2DPos[XX] = ((int) (state->x[i][XX]*flowFieldData->invBinSize[XX]
                        + flowFieldData->numBins[XX] - 1))
                    % flowFieldData->numBins[XX];
            bin2DPos[YY] = ((int) (state->x[i][ZZ]*flowFieldData->invBinSize[YY]
                        + flowFieldData->numBins[YY] - 1))
                    % flowFieldData->numBins[YY];
            bin1DPos = (flowFieldData->numBins[XX]*bin2DPos[YY]
                    + bin2DPos[XX])*CollectNumvar;

            // Add atom data to collection
            flowFieldData->dataGrid[bin1DPos + CollectFlowXX]
                += mdatoms->massT[i]*state->v[i][XX];
            flowFieldData->dataGrid[bin1DPos + CollectFlowYY]
                += mdatoms->massT[i]*state->v[i][ZZ];
            flowFieldData->dataGrid[bin1DPos + CollectTemp]
                += mdatoms->massT[i]*norm2(state->v[i]);
            flowFieldData->dataGrid[bin1DPos + CollectN]
                += 1;
            flowFieldData->dataGrid[bin1DPos + CollectMass]
                += mdatoms->massT[i];
        }
    }
}

// Finalise and output the data to specified maps
void output_flow_data(t_flowFieldData *flowFieldData, t_commrec *cr, gmx_int64_t step)
{
    FILE    *fpData;

    char    fnDataMap[STRLEN];          // Final file name of output map

    int     i, j,
            bin1DPos,                   // Corresponding osition in 1D *dataGrid
            numDataMap;                 // Number of data map to output

    float   binCenter[NumBinDim],       // Center position of bin
            outData[OutputLength];      // Array for output data

    // Reduce data from MPI processing elements
    // Raise warning if MPI_IN_PLACE does not run on platform
    if (PAR(cr))
    {
#if defined(MPI_IN_PLACE_EXISTS)
        /* Master collects data from all PE's and prints */
        MPI_Reduce(MASTER(cr) ? MPI_IN_PLACE : flowFieldData->dataGrid,
                MASTER(cr) ? flowFieldData->dataGrid : NULL,
                flowFieldData->numBins[XX]*flowFieldData->numBins[YY]*CollectNumvar,
                MPI_DOUBLE, MPI_SUM, MASTERRANK(cr),
                cr->mpi_comm_mygroup);
#else
#warning "MPI_IN_PLACE not available on platform"
        MPI_Reduce(MASTER(cr) ? MPI_IN_PLACE : flowFieldData->dataGrid,
                MASTER(cr) ? flowFieldData->dataGrid : NULL,
                flowFieldData->numBins[XX]*flowFieldData->numBins[YY]*CollectNumvar,
                MPI_DOUBLE, MPI_SUM, MASTERRANK(cr),
                cr->mpi_comm_mygroup);
#endif
    }

    if (MASTER(cr))
    {
        numDataMap = (int) (step/flowFieldData->dataStep[Output]);

        sprintf(fnDataMap, "%s_%05d.%s",
                flowFieldData->fnDataBase, numDataMap, ftp2ext(efDAT));
        fpData = gmx_ffopen(fnDataMap, "wb");

        /* Calculate and output to data maps:
         *   Positions of bin centers in X and Y
         *   Accumulated number of atoms
         *   Temperature
         *   Mass, average over steps
         *   Flow U and V in X and Y respectively
         */
        for (i = 0; i < flowFieldData->numBins[XX]; i++)
        {
            binCenter[XX] = (i + 0.5)*flowFieldData->binSize[XX];

            for (j = 0; j < flowFieldData->numBins[YY]; j++)
            {
                binCenter[YY] = (j + 0.5)*flowFieldData->binSize[YY];
                bin1DPos = (flowFieldData->numBins[XX]*j + i)*CollectNumvar;

                /* Average U and V over accumulated
                 * mass in bins
                 */
                if (flowFieldData->dataGrid[bin1DPos + CollectMass] > 0.0)
                {
                    flowFieldData->dataGrid[bin1DPos + CollectFlowXX]
                        /= flowFieldData->dataGrid[bin1DPos + CollectMass];
                    flowFieldData->dataGrid[bin1DPos + CollectFlowYY]
                        /= flowFieldData->dataGrid[bin1DPos + CollectMass];
                }
                else
                {
                    flowFieldData->dataGrid[bin1DPos + CollectFlowXX] = 0.0;
                    flowFieldData->dataGrid[bin1DPos + CollectFlowYY] = 0.0;
                }

                /* T calculated here, output with density */
                if (flowFieldData->dataGrid[bin1DPos + CollectN] > 0)
                {
                    flowFieldData->dataGrid[bin1DPos + CollectTemp]
                        /= (2*BOLTZ*flowFieldData->dataGrid[bin1DPos + CollectN]);
                }
                else
                {
                    flowFieldData->dataGrid[bin1DPos + CollectTemp] = 0.0;
                }

                // Prepare output data array
                outData[OutputXX] = binCenter[XX];
                outData[OutputYY] = binCenter[YY];
                outData[OutputN] = flowFieldData->dataGrid[bin1DPos + CollectN]
                    /(flowFieldData->dataStep[Ratio]);
                outData[OutputTemp] = flowFieldData->dataGrid[bin1DPos + CollectTemp];
                outData[OutputMass] = flowFieldData->dataGrid[bin1DPos + CollectMass]
                    /(flowFieldData->dataStep[Ratio]);
                outData[OutputFlowXX] = flowFieldData->dataGrid[bin1DPos + CollectFlowXX];
                outData[OutputFlowYY] = flowFieldData->dataGrid[bin1DPos + CollectFlowYY];

                // Output to fpData
                fwrite(&outData, sizeof(outData[OutputXX]), OutputLength,
                        fpData);
            }
        }

        fclose(fpData);
    }

    /* Reset calculated quantities on all nodes after output */
    for (i = 0;
            i < flowFieldData->numBins[XX]*flowFieldData->numBins[YY]*CollectNumvar;
            i++)
    {
        flowFieldData->dataGrid[i] = 0;
    }
}

#endif
