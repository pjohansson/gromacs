#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gromacs/commandline/filenm.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/mdlib/sim_util.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/units.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/smalloc.h"

#include "md_petter.h"


FlowData
init_flow_container(const int         nfile,
                    const t_filenm    fnm[],
                    const t_inputrec *ir,
                    const t_state    *state)
{
    const auto step_collect = static_cast<uint64_t>(ir->userint1);
    auto step_output = static_cast<uint64_t>(ir->userint2);

    const auto nx = static_cast<size_t>(ir->userint3);
    const auto nz = static_cast<size_t>(ir->userint4);

    const auto dx = static_cast<double>(state->box[XX][XX]) / static_cast<double>(nx);
    const auto dz = static_cast<double>(state->box[ZZ][ZZ]) / static_cast<double>(nz);

    // Control userargs, although this should be done during pre-processing
    if (nx <= 0 || nz <= 0)
    {
        gmx_fatal(FARGS,
                  "Number of bins along x (userint3 = %lu) and z (userint4 = %lu) "
                  "for flow data calculation and output must be larger than 0.",
                  nx, nz);
    }

    if (step_collect <= 0 || step_output <= 0)
    {
        gmx_fatal(FARGS,
                  "Number of steps that elapse between collection (userint1 = %lu) "
                  "and output (userint2 = %lu) of flow data must be larger than 0.",
                  step_collect, step_output);
    }
    else if (step_collect > step_output)
    {
        gmx_fatal(FARGS,
                  "Number of steps elapsing between output (userint2 = %lu) "
                  "must be larger than steps between collection (userint1 = %lu).",
                  step_output, step_collect);
    }
    else if (step_output % step_collect != 0)
    {
        const auto new_step_output = static_cast<uint64_t>(
            round(step_output / step_collect) * step_collect
        );

        gmx_warning("Steps for outputting flow data (userint2 = %lu) not "
                    "multiple of steps for collecting (userint1 = %lu). "
                    "Setting number of steps that elapse between output to %lu.",
                    step_output, step_collect, new_step_output);

        step_output = new_step_output;
    }

    // Get name base of output datamaps by stripping the extension and dot (.)
    std::string fnbase = opt2fn("-flow", nfile, fnm);

    const int ext_length = static_cast<int>(strlen(ftp2ext(efDAT)));
    const int base_length = static_cast<int>(fnbase.size()) - ext_length - 1;

    if (base_length > 0) 
    {
        fnbase.resize(static_cast<size_t>(base_length));
    }

    return FlowData(fnbase, {nx , nz}, {dx, dz}, step_collect, step_output);
}


void 
print_flow_collection_information(const FlowData &flowcr, const double dt) 
{
    fprintf(stderr,
            "\nData for flow field maps will be collected every %g ps "
            "(%lu steps).\n", 
            flowcr.step_collect * dt, flowcr.step_collect);

    fprintf(stderr,
            "It will be averaged and output to data maps every %g ps "
            "(%lu steps).\n", 
            flowcr.step_output * dt, flowcr.step_output);

    fprintf(stderr,
            "The system has been divided into %lu x %lu bins "
            "of size %g x %g nm^2 \nin x and z.\n",
            flowcr.nx(), flowcr.nz(), flowcr.dx(), flowcr.dz());

    fprintf(stderr, "Have a nice day.\n\n");
}


static void
collect_flow_data(FlowData           &flowcr,
                  const t_commrec    *cr,
                  const t_mdatoms    *mdatoms,
                  const t_state      *state,
                  const gmx_groups_t *groups)
{
    for (size_t i = 0; i < static_cast<size_t>(mdatoms->homenr); ++i)
    {
        // Check for match to the input group using the global atom index,
        // since groups contain these indices instead of MPI rank local indices
        const auto index_global = DOMAINDECOMP(cr) ? cr->dd->globalAtomIndices[i] : static_cast<int>(i);

        if (getGroupType(groups, egcUser1, index_global) == 0)
        {
            const auto ix = flowcr.get_xbin(state->x[i][XX]);
            const auto iz = flowcr.get_zbin(state->x[i][ZZ]);

            const auto bin = flowcr.get_1d_index(ix, iz);
            const auto mass = mdatoms->massT[i];

            flowcr.data[bin + static_cast<size_t>(FlowVariable::NumAtoms)] += 1.0;
            flowcr.data[bin + static_cast<size_t>(FlowVariable::Temp)    ] += mass * norm2(state->v[i]);
            flowcr.data[bin + static_cast<size_t>(FlowVariable::Mass)    ] += mass;
            flowcr.data[bin + static_cast<size_t>(FlowVariable::U)       ] += mass * state->v[i][XX];
            flowcr.data[bin + static_cast<size_t>(FlowVariable::V)       ] += mass * state->v[i][ZZ];
        }
    }
}

enum class FlowOutput {
    X,
    Z,
    NumAtoms,
    Temp,
    Mass,
    U,
    V,
    NumVariables
};
constexpr size_t NUM_OUTPUT_VARIABLES = static_cast<size_t>(FlowOutput::NumVariables);

static void
output_flow_data(FlowData               &flowcr,
                 const t_commrec        *cr,
                 const uint64_t          current_step)
{
    // Reduce data from MPI processing elements
    // Raise warning if MPI_IN_PLACE does not run on platform
    if (PAR(cr))
    {
#if defined(MPI_IN_PLACE_EXISTS)
        /* Master collects data from all PE's and prints */
        MPI_Reduce(MASTER(cr) ? MPI_IN_PLACE : flowcr.data.data(),
                MASTER(cr) ? flowcr.data.data() : NULL,
                flowcr.data.size(),
                MPI_DOUBLE, MPI_SUM, MASTERRANK(cr),
                cr->mpi_comm_mygroup);
#else
#warning "MPI_IN_PLACE not available on platform"
        MPI_Reduce(MASTER(cr) ? MPI_IN_PLACE : flowcr.data.data(),
                MASTER(cr) ? flowcr.data.data() : NULL,
                flowcr.data.size(),
                MPI_DOUBLE, MPI_SUM, MASTERRANK(cr),
                cr->mpi_comm_mygroup);
#endif
    }

    if (MASTER(cr))
    {
        // Construct output file name
        char fnout[STRLEN];
        const auto file_index = static_cast<uint16_t>(current_step / flowcr.step_output);
        snprintf(fnout, STRLEN, "%s_%05u.%s", flowcr.fnbase.c_str(), file_index, ftp2ext(efDAT));

        FILE* fp = gmx_ffopen(fnout, "wb");
        std::array<float, NUM_OUTPUT_VARIABLES> output_buffer;

        for (size_t ix = 0; ix < flowcr.nx(); ++ix)
        {
            output_buffer[static_cast<size_t>(FlowOutput::X)] = flowcr.get_x(ix);

            for (size_t iz = 0; iz < flowcr.nz(); ++iz)
            {
                output_buffer[static_cast<size_t>(FlowOutput::Z)] = flowcr.get_z(iz);

                const auto bin = flowcr.get_1d_index(ix, iz);

                const auto num_atoms = flowcr.data[bin + static_cast<size_t>(FlowVariable::NumAtoms)];
                const auto mass      = flowcr.data[bin + static_cast<size_t>(FlowVariable::Mass)    ];

                // The temperature and flow is averaged by the sampled number 
                // of atoms and mass in each bin. 
                const auto temperature = num_atoms > 0.0 
                    ? flowcr.data[bin + static_cast<size_t>(FlowVariable::Temp)] / (2.0 * BOLTZ * num_atoms) 
                    : 0.0;

                const auto flow_x = mass > 0.0 ? flowcr.data[bin + static_cast<size_t>(FlowVariable::U)] / mass : 0.0;
                const auto flow_z = mass > 0.0 ? flowcr.data[bin + static_cast<size_t>(FlowVariable::V)] / mass : 0.0;

                // In contrast to above, the mass and number of atoms has to be divided by 
                // the number of samples taken to get their average.
                const auto num_samples = static_cast<float>(flowcr.step_ratio);
                const auto avg_num_atoms = num_atoms / num_samples;
                const auto avg_mass = mass / num_samples;

                output_buffer[static_cast<size_t>(FlowOutput::NumAtoms)] = static_cast<float>(avg_num_atoms);
                output_buffer[static_cast<size_t>(FlowOutput::Mass)    ] = static_cast<float>(avg_mass);
                output_buffer[static_cast<size_t>(FlowOutput::Temp)    ] = static_cast<float>(temperature);
                output_buffer[static_cast<size_t>(FlowOutput::U)       ] = static_cast<float>(flow_x);
                output_buffer[static_cast<size_t>(FlowOutput::V)       ] = static_cast<float>(flow_z);

                fwrite(output_buffer.data(), sizeof(float), NUM_OUTPUT_VARIABLES, fp);
            }
        }

        gmx_ffclose(fp);
    }
}


void
flow_collect_or_output(FlowData           &flowcr,
                       const uint64_t      current_step,
                       const t_commrec    *cr,
                       const t_inputrec   *ir,
                       const t_mdatoms    *mdatoms,
                       const t_state      *state,
                       const gmx_groups_t *groups)
{
    collect_flow_data(flowcr, cr, mdatoms, state, groups);

    if (do_per_step(current_step, flowcr.step_output) 
        && (static_cast<int64_t>(current_step) != ir->init_step))
    {
        output_flow_data(flowcr, cr, current_step);
        flowcr.reset_data();
    }
}

