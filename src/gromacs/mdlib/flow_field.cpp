#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>

#include "gromacs/commandline/filenm.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/mdlib/stat.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/units.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/smalloc.h"

#include "flow_field.h"

/*! \brief Get the number of groups in User1

    This is slightly complicated by how Gromacs adds a "rest" group 
    to the array of names if the other groups do not add up to all 
    atoms in the system. Thus, we detect if the final group is called 
    exactly "rest" and if so do not count it as one of the groups. */
static size_t get_num_groups(const SimulationGroups *groups)
{
    size_t num_groups = 0;

    for (const auto global_group_index 
         : groups->groups[SimulationAtomGroupType::User1])
    {
        const auto name = groups->groupNames[global_group_index];

        if (strncmp(*name, "rest", 8) == 0)
        {
            break;
        }

        num_groups++;
    }

    return num_groups;
}


FlowData
init_flow_container(const int               nfile,
                    const t_filenm          fnm[],
                    const t_inputrec       *ir,
                    const SimulationGroups *groups,
                    const t_state          *state)
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

    // If more than one group is selected for output, collect them to do separate 
    // collection for each individual group (as well as them all combined)
    //
    // Get the number of selected groups, subtract 1 because "rest" is always present
    // const size_t num_groups = groups->grps[egcUser1].nr - 1;
    const size_t num_groups = get_num_groups(groups);

    std::vector<std::string> group_names;

    if (num_groups > 1)
    {
        for (size_t i = 0; i < num_groups; ++i)
        {
            const auto global_group_index 
                = groups->groups[SimulationAtomGroupType::User1].at(i);

            const char *name = *groups->groupNames[global_group_index];
            // const size_t index_name = groups->grps[egcUser1].nm_ind[i];
            // const char *name = *groups->grpname[index_name];

            group_names.push_back(std::string(name));
        }
    }

    return FlowData(fnbase, group_names, nx , nz, dx, dz, step_collect, step_output);
}


void 
print_flow_collection_information(const FlowData &flowcr, const double dt) 
{
    fprintf(stderr, "\n\n************************************\n");
    fprintf(stderr, "* FLOW DATA COLLECTION INFORMATION *\n");
    fprintf(stderr, "************************************\n\n");

    fprintf(stderr,
            "Data for flow field maps will be collected every %g ps "
            "(%lu steps).\n\n", 
            flowcr.step_collect * dt, flowcr.step_collect);

    fprintf(stderr,
            "It will be averaged and output to data maps every %g ps "
            "(%lu steps).\n\n", 
            flowcr.step_output * dt, flowcr.step_output);

    fprintf(stderr,
            "The system has been divided into %lu x %lu bins "
            "of size %g x %g nm^2 \nin x and z.\n\n",
            flowcr.nx(), flowcr.nz(), flowcr.dx(), flowcr.dz());
    
    fprintf(stderr,
            "Writing full flow data to files with base '%s_00001.dat' (...).\n\n", flowcr.fnbase.c_str());
    
    if (!flowcr.group_data.empty()) 
    {
        fprintf(stderr, 
                "Multiple groups selected for flow output. Will collect individual flow\n"
                "data for each group individually in addition to the combined field:\n\n");
        
        for (const auto& group : flowcr.group_data)
        {
            fprintf(stderr, 
                    "  %s -> '%s_00001.dat' (...)\n", group.name.c_str(), group.fnbase.c_str());
        }

        fprintf(stderr, "\n");
    }

    fprintf(stderr, "Have a nice day.\n\n");

    fprintf(stderr, "****************************************\n");
    fprintf(stderr, "* END FLOW DATA COLLECTION INFORMATION *\n");
    fprintf(stderr, "****************************************\n\n");
}


static void 
add_flow_to_bin(std::vector<double> &data, 
                const size_t         atom,
                const size_t         bin,
                const real           mass,
                const t_state       *state)
{
    data[bin + static_cast<size_t>(FlowVariable::NumAtoms)] += 1.0;
    data[bin + static_cast<size_t>(FlowVariable::Temp)    ] += mass * norm2(state->v[atom]);
    data[bin + static_cast<size_t>(FlowVariable::Mass)    ] += mass;
    data[bin + static_cast<size_t>(FlowVariable::U)       ] += mass * state->v[atom][XX];
    data[bin + static_cast<size_t>(FlowVariable::V)       ] += mass * state->v[atom][ZZ];
}


static void
collect_flow_data(FlowData           &flowcr,
                  const t_commrec    *cr,
                  const t_mdatoms    *mdatoms,
                  const t_state      *state,
                  const SimulationGroups *groups)
{
    const int num_groups = flowcr.group_data.empty() ? 1 : flowcr.group_data.size();

    for (size_t i = 0; i < static_cast<size_t>(mdatoms->homenr); ++i)
    {
        // Check for match to the input group using the global atom index,
        // since groups contain these indices instead of MPI rank local indices
        const auto index_global = DOMAINDECOMP(cr) ? cr->dd->globalAtomIndices[i] : static_cast<int>(i);

        const auto index_group = getGroupType(*groups, SimulationAtomGroupType::User1, index_global);

        if (index_group < num_groups)
        {
            const auto ix = flowcr.get_xbin(state->x[i][XX]);
            const auto iz = flowcr.get_zbin(state->x[i][ZZ]);

            const auto bin = flowcr.get_1d_index(ix, iz);
            const auto mass = mdatoms->massT[i];

            add_flow_to_bin(flowcr.data, i, bin, mass, state);

            if (!flowcr.group_data.empty() && index_group < static_cast<int>(flowcr.group_data.size()))
            {
                add_flow_to_bin(flowcr.group_data.at(index_group).data, i, bin, mass, state);
            }
        }
    }
}


struct FlowBinData {
    float mass, 
          temp, 
          num_atoms, 
          u, 
          v;
};


struct FlowDataOutput {
    std::vector<uint64_t> ix, iy;
    std::vector<float> mass, num_atoms, temp, us, vs;

    FlowDataOutput(const size_t num_elements) 
    {
        ix.reserve(num_elements);
        iy.reserve(num_elements);
        mass.reserve(num_elements);
        num_atoms.reserve(num_elements);
        temp.reserve(num_elements);
        us.reserve(num_elements);
        vs.reserve(num_elements);
    }
};


static FlowBinData 
calc_values_in_bin(const std::vector<double> &data,
                   const size_t               bin,
                   const uint64_t             samples_per_output)
{
    const auto num_atoms = data[bin + static_cast<size_t>(FlowVariable::NumAtoms)];
    const auto mass      = data[bin + static_cast<size_t>(FlowVariable::Mass)    ];

    // The temperature and flow is averaged by the sampled number 
    // of atoms and mass in each bin. 
    const auto temperature = num_atoms > 0.0 
        ? data[bin + static_cast<size_t>(FlowVariable::Temp)] / (2.0 * BOLTZ * num_atoms) 
        : 0.0;

    const auto flow_x = mass > 0.0 ? data[bin + static_cast<size_t>(FlowVariable::U)] / mass : 0.0;
    const auto flow_z = mass > 0.0 ? data[bin + static_cast<size_t>(FlowVariable::V)] / mass : 0.0;

    // In contrast to above, the mass and number of atoms has to be divided by 
    // the number of samples taken to get their average.
    const auto num_samples = static_cast<float>(samples_per_output);
    const auto avg_num_atoms = num_atoms / num_samples;
    const auto avg_mass = mass / num_samples;

    FlowBinData bin_data;

    bin_data.num_atoms = static_cast<float>(avg_num_atoms);
    bin_data.mass = static_cast<float>(avg_mass);
    bin_data.temp = static_cast<float>(temperature);
    bin_data.u    = static_cast<float>(flow_x);
    bin_data.v    = static_cast<float>(flow_z);

    return bin_data;
}


static void
add_bin_if_non_empty(FlowDataOutput    &data,
                     const size_t       ix,
                     const size_t       iy,
                     const FlowBinData &bin_data)
{
    if (bin_data.mass > 0.0) 
    {
        data.ix.push_back(static_cast<uint64_t>(ix));
        data.iy.push_back(static_cast<uint64_t>(iy));

        data.mass.push_back(bin_data.mass);
        data.num_atoms.push_back(bin_data.num_atoms);
        data.temp.push_back(bin_data.temp);
        data.us.push_back(bin_data.u);
        data.vs.push_back(bin_data.v);
    }
}


static void 
write_header(FILE        *fp,
             const size_t nx,
             const size_t ny,
             const double dx,
             const double dy,
             const size_t num_values)
{
    std::ostringstream buf;

    buf << "FORMAT " << FLOW_FILE_HEADER_NAME << '\n';
    buf << "ORIGIN 0.0 0.0\n";
    buf << "SHAPE " << nx << ' ' << ny << '\n';
    buf << "SPACING " << dx << ' ' << dy << '\n';
    buf << "NUMDATA " << num_values << '\n';
    buf << "FIELDS IX IY N T M U V\n";
    buf << "COMMENT Grid is regular but only non-empty bins are output\n";
    buf << "COMMENT There are 'NUMDATA' non-empty bins and that many values are stored for each field\n";
    buf << "COMMENT 'FIELDS' is the different fields for each bin:\n";
    buf << "COMMENT 'IX' and 'IY' are bin indices along x and y respectively\n";
    buf << "COMMENT 'N' is the average number of atoms\n";
    buf << "COMMENT 'M' is the average mass\n";
    buf << "COMMENT 'T' is the temperature\n";
    buf << "COMMENT 'U' and 'V' is the mass flow along x and y respectively\n";
    buf << "COMMENT Data is stored as 'NUMDATA' counts for each field in 'FIELDS', in order\n";
    buf << "COMMENT 'IX' and 'IY' are 64-bit unsigned integers\n";
    buf << "COMMENT Other fields are 32-bit floating point numbers\n";
    buf << "COMMENT Data begins after '\\0' character\n";
    buf << "COMMENT Example: with 'NUMDATA' = 4 and 'FIELDS' = 'IX IY N T', "
                << "the data following the '\\0' marker is 4 + 4 64-bit integers "
                << "and then 4 + 4 32-bit floating point numbers\n";
    buf << '\0';

    const std::string header_str { buf.str() };

    fwrite(header_str.c_str(), sizeof(char), header_str.size(), fp);
}


static void 
write_flow_data(const std::string    &fnbase, 
                const size_t          num_file, 
                const FlowDataOutput &data,
                const size_t          nx,
                const size_t          ny,
                const double          dx,
                const double          dy)
{
    char fn[STRLEN];
    snprintf(fn, STRLEN, "%s_%05lu.%s", fnbase.c_str(), num_file, ftp2ext(efDAT));

    FILE *fp = gmx_ffopen(fn, "wb");

    const size_t num_elements = data.ix.size();
    write_header(fp, nx, ny, dx, dy, num_elements);

    fwrite(data.ix.data(),        sizeof(uint64_t), num_elements, fp);
    fwrite(data.iy.data(),        sizeof(uint64_t), num_elements, fp);
    fwrite(data.num_atoms.data(), sizeof(float),    num_elements, fp);
    fwrite(data.temp.data(),      sizeof(float),    num_elements, fp);
    fwrite(data.mass.data(),      sizeof(float),    num_elements, fp);
    fwrite(data.us.data(),        sizeof(float),    num_elements, fp);
    fwrite(data.vs.data(),        sizeof(float),    num_elements, fp);

    gmx_ffclose(fp);
}


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
        
        for (auto& group_data : flowcr.group_data)
        {
            MPI_Reduce(MASTER(cr) ? MPI_IN_PLACE : group_data.data.data(),
                    MASTER(cr) ? group_data.data.data() : NULL,
                    group_data.data.size(),
                    MPI_DOUBLE, MPI_SUM, MASTERRANK(cr),
                    cr->mpi_comm_mygroup);

        }
#else
#warning "MPI_IN_PLACE not available on platform"
        MPI_Reduce(MASTER(cr) ? MPI_IN_PLACE : flowcr.data.data(),
                MASTER(cr) ? flowcr.data.data() : NULL,
                flowcr.data.size(),
                MPI_DOUBLE, MPI_SUM, MASTERRANK(cr),
                cr->mpi_comm_mygroup);
        for (auto& group_data : flowcr.group_data)
        {
            MPI_Reduce(MASTER(cr) ? MPI_IN_PLACE : group_data.data.data(),
                    MASTER(cr) ? group_data.data.data() : NULL,
                    group_data.data.size(),
                    MPI_DOUBLE, MPI_SUM, MASTERRANK(cr),
                    cr->mpi_comm_mygroup);
        }
#endif
    }

    if (MASTER(cr))
    {
        const auto num_elements = flowcr.nx() * flowcr.nz();
        FlowDataOutput system_bin_data(num_elements);
        std::vector<FlowDataOutput> separate_group_bin_data;

        for (size_t i = 0; i < flowcr.group_data.size(); ++i)
        {
            separate_group_bin_data.push_back(FlowDataOutput(num_elements));
        }

        for (size_t ix = 0; ix < flowcr.nx(); ++ix)
        {
            for (size_t iz = 0; iz < flowcr.nz(); ++iz)
            {
                const auto bin = flowcr.get_1d_index(ix, iz);
                const auto bin_data = calc_values_in_bin(flowcr.data, bin, flowcr.step_ratio);
                add_bin_if_non_empty(system_bin_data, ix, iz, bin_data);

                for (size_t i = 0; i < flowcr.group_data.size(); ++i)
                {
                    const auto& data = flowcr.group_data.at(i).data;
                    auto& group_data = separate_group_bin_data.at(i);

                    const auto group_bin_data = calc_values_in_bin(data, bin, flowcr.step_ratio);
                    add_bin_if_non_empty(group_data, ix, iz, group_bin_data);
                }
            }
        }

        const auto file_index = static_cast<uint16_t>(current_step / flowcr.step_output);

        write_flow_data(
            flowcr.fnbase, file_index, system_bin_data, 
            flowcr.nx(), flowcr.nz(), flowcr.dx(), flowcr.dz()
        );

        for (size_t i = 0; i < flowcr.group_data.size(); ++i)
        {
            const auto& fnbase = flowcr.group_data.at(i).fnbase;
            const auto& group_data = separate_group_bin_data.at(i);

            write_flow_data(
                fnbase, file_index, group_data,
                flowcr.nx(), flowcr.nz(), flowcr.dx(), flowcr.dz()
            );
        }
    }
}


void
flow_collect_or_output(FlowData               &flowcr,
                       const uint64_t          current_step,
                       const t_commrec        *cr,
                       const t_inputrec       *ir,
                       const t_mdatoms        *mdatoms,
                       const t_state          *state,
                       const SimulationGroups *groups)
{
    collect_flow_data(flowcr, cr, mdatoms, state, groups);

    if (do_per_step(current_step, flowcr.step_output) 
        && (static_cast<int64_t>(current_step) != ir->init_step))
    {
        output_flow_data(flowcr, cr, current_step);
        flowcr.reset_data();
    }
}

