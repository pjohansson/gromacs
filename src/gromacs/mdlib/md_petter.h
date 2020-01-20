#ifndef MD_PETTER
#define MD_PETTER

#include <array>
#include <string>
#include <vector>

#include "gromacs/mdtypes/state.h"

// We are using a grid along X and Z so we use a separate enum 
// to not confuse our indexing with regular XX, YY and ZZ
enum class GridAxes {
    X,
    Z,
    NumAxes
};
constexpr size_t NUM_AXES = static_cast<size_t>(GridAxes::NumAxes);

// Indices for different data in array
enum class FlowVariable {
    NumAtoms,
    Temp,
    Mass,
    U,      // Mass flow along X
    V,      //             and Z
    NumVariables
};
constexpr size_t NUM_FLOW_VARIABLES = static_cast<size_t>(FlowVariable::NumVariables);

class FlowData {
public:
    bool bDoFlowCollection = false;

    std::string fnbase;

    std::vector<double> data;   // A 2D grid is represented by this 1D array

    uint64_t step_collect = 0, 
             step_output = 0,
             step_ratio = 0;

    FlowData() {}

    FlowData(const std::string fnbase,
             const std::array<size_t, NUM_AXES> num_bins,
             const std::array<double, NUM_AXES> bin_size,
             const uint64_t step_collect,
             const uint64_t step_output)
    :bDoFlowCollection { true },
     fnbase { fnbase },
     data(nx() * nz() * NUM_FLOW_VARIABLES, 0.0),
     step_collect { step_collect },
     step_output { step_output },
     step_ratio { static_cast<uint64_t>(step_output / step_collect) },
     num_bins { num_bins },
     bin_size { bin_size },
     inv_bin_size { 1.0 / dx(), 1.0 / dz() } {}

    double dx() const { return bin_size[static_cast<size_t>(GridAxes::X)]; }
    double dz() const { return bin_size[static_cast<size_t>(GridAxes::Z)]; }

    double inv_dx() const { return inv_bin_size[static_cast<size_t>(GridAxes::X)]; }
    double inv_dz() const { return inv_bin_size[static_cast<size_t>(GridAxes::Z)]; }

    size_t nx() const { return num_bins[static_cast<size_t>(GridAxes::X)]; }
    size_t nz() const { return num_bins[static_cast<size_t>(GridAxes::Z)]; }

    size_t get_1d_index(const size_t ix, const size_t iz) const 
    { 
        return (iz * nx() + ix) * NUM_FLOW_VARIABLES;
    }

    size_t get_xbin(const real x) const { return get_bin_from_position(x, nx(), inv_dx()); }
    size_t get_zbin(const real z) const { return get_bin_from_position(z, nz(), inv_dz()); }

    float get_x(const size_t ix) const { return get_position(ix, dx()); }
    float get_z(const size_t iz) const { return get_position(iz, dz()); }

    void reset_data() { data.assign(data.size(), 0.0); }

private:
    std::array<size_t, NUM_AXES> num_bins;
    std::array<double, NUM_AXES> bin_size,
                                 inv_bin_size;

    size_t get_bin_from_position(const real x, const size_t num_bins, const real inv_bin) const
    {
        auto index = static_cast<int>(floor(x * inv_bin)) % static_cast<int>(num_bins);

        while (index < 0)
        {
            index += num_bins;
        }

        return index;
    }

    float get_position(const size_t index, const float bin_size) const
    {
        return (static_cast<float>(index) + 0.5) * bin_size;
    }
};

// Prepare and return a container for flow field data
FlowData
init_flow_container(const int         nfile,
                    const t_filenm    fnm[],
                    const t_inputrec *ir,
                    const t_state    *state);

// Write information about the flow field collection
void 
print_flow_collection_information(const FlowData &flowcr, const double dt);

// If at a collection or output step, perform actions
void
flow_collect_or_output(FlowData           &flowcr,
                       const uint64_t      step,
                       const t_commrec    *cr,
                       const t_inputrec   *ir,
                       const t_mdatoms    *mdatoms,
                       const t_state      *state,
                       const gmx_groups_t *groups);

#endif
