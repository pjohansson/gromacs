#include "gromacs/mdrun/md_shear_coupling.h"

#include "gromacs/fileio/oenv.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/math/units.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/futil.h"

#include <array>
#include <cstring>
#include <vector>

/*-----------------------------------------------------------------------------*
 * SHEAR VELOCITY COUPLING                                                     *
 * -----------------------                                                     *
 * This is an implementation of a method to set a target velocity in           *
 * different areas of a liquid system, in order to create a shear flow.        *
 * Instead of pulling with an external force or position restraints,           *
 * we exchange the velocities of the atom with the highest velocity            *
 * opposite to the targeted flow direction in one of the areas, with           *
 * the velocity of the corresponding atom in the other area.                   *
 *                                                                             *
 * For instance, if we desire a flow along X that is positive at the           *
 * top of the simulation box (along Z), and negative at the bottom,            *
 * we take the atom in the top area with the highest *negative* velocity       *
 * along X, and exchange its velocity with the atom in the bottom area         *
 * with the highest *positive* velocity along X. Thus the top area             *
 * gets a boost in its flow towards the target velocity, and respectively      *
 * for the bottom area.                                                        *
 *                                                                             *
 * The exchange is only done if the current velocities in the exchange         *
 * areas are below the target velocities. If done frequently enough            *
 * (before viscous friction eliminates the local flow), a shear will           *
 * be created with no external contributions to the energy. This works         *
 * really well for simple Lennard-Jones liquids. Probably not as well          *
 * for complex molecules, but I have yet to check.                             *
 *                                                                             *
 * Kinetic energy is conserved by scaling the exchanged velocities with        *
 * masses of the atoms. Thus the total energy of the system is not affected    *
 * by creating this shear.                                                     *
 *                                                                             *
 * I call this "shear coupling" because there is an exchange which couples     *
 * the two areas of the system, but I am not sure that this is strictly        *
 * correct terminology.                                                        *
 *                                                                             *
 *                                                                             *
 * With this method I introduce some new MDP options:                          *
 *                                                                             *
 *   shear-coupling       = yes/no, to turn on the coupling                    *
 *   shear-axis           = x/y/z, the axis along which the areas              *
 *                          are positioned                                     *
 *   shear-direction      = x/y/z, the axis along which the flow is directed   *
 *   shear-strategy       = Edges/Edge-Center, which sets up the two           *
 *                          exchange areas:                                    *
 *                            Edges: area0 and area1 are at the top and        *
 *                              bottom edges along the axis                    *
 *                            Edge-Center: area0 is split between the bottom   *
 *                              and top edges and area1 is in the center       *
 *   shear-tcoupl         = time, perform the coupling at multiples of this    *
 *   shear-area-size      = size, extent of area0 and area1 along the axis     *
 *   shear-zadj           = size, move areas towards the center of the box     *
 *                          by this amount (useful with walls)                 *
 *   shear-ref-velocity   = vel, target velocity reference for areas:          *
 *                            area0: -shear-ref-velocity                       *
 *                            area1: +shear-ref-velocity                       *
 *   shear-grps           = 1 or 2 groups, if 1 the group of atoms to exchange *
 *                          in both areas, if 2 respectively the atoms of      *
 *                          area0 and area1. Note that this is an alias        *
 *                          for user2-grps.                                    *
 *                                                                             *
 * Since a have touched the MDP options, there are a lot of small changes      *
 * scattered throughout the Gromacs source code beyond this file. They can     *
 * be found by grep'ing for "PETTER" or comparing the source to the original   *
 * Gromacs distribution.                                                       *
 *                                                                             *
 * BUGS:                                                                       *
 * This code is not well tested. Use at own risk. It has been developed        *
 * with MPI parallelization in mind, but there may be edge cases from          *
 * how Gromacs works under the hood which may become problems.                 *
 *                                                                             *
 * CREDITS:                                                                    *
 * I did not come up with this method, I only implemented it in Gromacs.       *
 * Credit for the method goes to someone who I have yet been able to look      *
 * up the name of, but will. I was introduced to it by Guillaume Galliero      * 
 * of LFCR, Université de Pau et des Pays de l'Adour.                          *
 *                                                                             *
 * Petter Johansson, Pau 2020                                                  *
 * pjohansson@univ-pau.fr                                                      *
 *                                                                             *
 * Originally based on gromacs-2020.3                                          *
 *-----------------------------------------------------------------------------*/

// #define MPI_SHEAR_DEBUG

/* These includes are used to sleep threads for small periods to print 
   output from PE's in order, in conjunction with MPI_Barrier */
#ifdef MPI_SHEAR_DEBUG
#include <chrono>
#include <thread>
using namespace std::chrono_literals;
#define MPI_SHEAR_SLEEP_DURATION 10ms
#endif


/*************************
 * ENUMS AND COLLECTIONS *
 *************************/

/*! Shorthand for an std rvec, since I prefer dealing with collections over pointers */
using Vec3 = std::array<real, DIM>;

/*! \brief Target direction for the flow in an exchange area */
enum class Direction {
    Positive,    /* Along the specified direction */
    Negative     /* Opposite to the specified direction */
};

/*! \brief Defines the extent of and target direction of an exchange area */
struct ExchangeArea {
    real target_velocity,   /* Target velocity of the area along the directed axis */
         zmin,              /* Minimum position along the specified axis for the area */
         zmax,              /* Maximum position along the specified axis */
         zmin2,             /* Second area minimum */
         zmax2;             /* Second area maximum */

    size_t group;           /* Index of the group for which to include atoms from */
    
    bool is_split = false;  /* Whether or not the area has two areas */

    Direction direction;    /* Whether the target velocity is positive or negative 
                               along the directed axis */
};

/*! \brief Tracks velocity data for a single exchange area */
struct ExchangeCounter {
    size_t index     = 0,               /* The atom index with the highest velocity 
                                           opposing the targeted flow direction 
                                           of the exchange area of this counter */

           num_atoms = 0;               /* The number of atoms in the exchange area */

    double atom_mass             = 0.0, /* Atom mass corresponding to the \p index */

           velocity_total        = 0.0; /* Sum of atom velocities along the targeted
                                           flow direction in this exchange area */
    
    /* Atom velocity corresponding to the \p index  */
    Vec3 velocity_max_vector = { 0.0, 0.0, 0.0 };
};

struct ExchangeAreaCounter {
    ExchangeAreaCounter(const ExchangeArea area) :area{area} {}

    ExchangeAreaCounter(const ExchangeArea area, const ExchangeCounter counter)
    :area{area},
     counter{counter}
     {}

    ExchangeArea    area;
    ExchangeCounter counter;
};


/******************************************************************
 * UTILITY FUNCTIONS FOR WORKING WITH POSITIONS, INDICES AND AXIS *
 ******************************************************************/

/*! \brief Return the name, i.e. x, y, or z, of the  \p axis */
constexpr static const char* get_axis_name(const Axis& axis) 
{
    constexpr std::array<const char*, DIM + 1> axis_names { "x", "y", "z", "null" };
    return axis_names[static_cast<size_t>(axis)];
}

/*! \brief Return the Axis enum corresponding to input \p axis from the mdp option */
static Axis get_axis(const ShearAxis_axis axis)
{
    switch (axis)
    {
        case ShearAxis_axis::X: return Axis::X;
        case ShearAxis_axis::Y: return Axis::Y;
        case ShearAxis_axis::Z: return Axis::Z;
        default: return Axis::NR;
    }
}

/*! \brief Return the Axis enum corresponding to input \p direction from the mdp option */
static Axis get_direction(const ShearAxis_direction direction)
{
    switch (direction)
    {
        case ShearAxis_direction::X: return Axis::X;
        case ShearAxis_direction::Y: return Axis::Y;
        case ShearAxis_direction::Z: return Axis::Z;
        default: return Axis::NR;
    }
}

/*! \brief Get the number of groups in User2 (i.e. shear-grps)

    This is slightly complicated by how Gromacs adds a "rest" group 
    to the array of names if the other groups do not add up to all 
    atoms in the system. Thus, we detect if the final group is called 
    exactly "rest" and if so do not count it as one of the groups. */
static size_t get_num_groups(const SimulationGroups *groups)
{
    size_t num_groups = 0;

    for (const auto global_group_index 
         : groups->groups[SimulationAtomGroupType::User2])
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

/*! \brief Return the global index corresponding to local atom \p index */
static int get_global_index_from_local(const size_t     index, 
                                       const t_commrec *cr)
{
    if (!havePPDomainDecomposition(cr))
    {
        return static_cast<int>(index);
    }

    /* For mpi_sync_counter_global_indices:
       There is a risk that no atoms exist on the local rank, in which case 
       the counter's num_atoms will be 0 so we can transmit a dummy number
       instead of a real index. The num_atoms 0 value will flag to ignore it. */
    if (index < cr->dd->globalAtomIndices.size())
    {
        return cr->dd->globalAtomIndices.at(index);
    }
    else 
    {
        return 0;
    }
}

/*! \brief If \p index_global corresponds to a \p index_local, set it 
    and return true, else return false */
static bool get_local_index_from_global(int             &index_local, 
                                        const int        index_global,
                                        const t_commrec *cr)
{
    if (!havePPDomainDecomposition(cr))
    {
        index_local = index_global;
        return true;
    }

    const auto entry = cr->dd->ga2la->find(index_global);

    if (entry != nullptr)
    {
        index_local = entry->la;
        return true;
    }
    else 
    {
        return false;
    }
}

/*! \brief Get the position of atom local index \p along axis \p eAxis 
    in the box, accounting for pbc = xyz */
static real get_position_in_box(const t_state *state,
                                const size_t   i,
                                const Axis     eAxis)
{
    const auto axis = static_cast<size_t>(eAxis);
    const real box_size = state->box[axis][axis];

    real z = fmod(state->x[i][axis], box_size); 

    while (z < 0.0) 
    {
        z += box_size;
    }

    return z;
}


/*****************************************
 * EXCHANGE COUNTER COLLECTION FUNCTIONS *
 *****************************************/

/*! \brief Return whether the input position \p z is inside the given exchange 
    area and is of the corresponding group */
static bool in_counter_area(const size_t               i, 
                            const real                 z, 
                            const ExchangeAreaCounter &area_counter,
                            const SimulationGroups    *groups,
                            const t_commrec           *cr)
{
    const auto& area = area_counter.area;

    const auto global_index = get_global_index_from_local(i, cr);
    const auto group = getGroupType(
        *groups, SimulationAtomGroupType::User2, global_index);

    if (group != area.group)
    {
        return false;
    }

    if (!area.is_split)
    {
        return ((z >= area.zmin) && (z <= area.zmax));
    }
    else
    {
        return (
            (z >= area.zmin) && (z <= area.zmax)
            || (z >= area.zmin2) && (z <= area.zmax2));
    }
}

/*! \brief Set the data for an atom opposing the flow in the counter */
static void set_atom_in_counter(ExchangeCounter &counter,
                                const size_t     index,
                                const double     mass,
                                const rvec       velocity)
{
    counter.index = index;
    counter.atom_mass = mass;

    for (size_t i = 0; i < DIM; ++i)
    {
        counter.velocity_max_vector[i] = velocity[i];
    }
}

/*! \brief Add the atom with local \p index velocity along 
    the target flow axis to the total in the \p area_counter 
    and determine whether it's the atom with the highest 
    *opposing* velocity */
static void add_atom_velocity(ExchangeAreaCounter &area_counter,
                              const size_t         index,
                              const ShearVelOpts  &opts,
                              const t_state       *state,
                              const t_mdatoms     *mdatoms)
{
    auto& counter = area_counter.counter;

    const auto mass = mdatoms->massT[index];
    const auto velocity_vector = state->v[index];

    const auto direction = static_cast<size_t>(opts.direction);

    const auto velocity             = velocity_vector[direction];
    const auto current_max_velocity = counter.velocity_max_vector[direction];

    switch (area_counter.area.direction)
    {
        case Direction::Positive:
            if ((velocity < current_max_velocity) || (counter.num_atoms == 0))
            {
                set_atom_in_counter(counter, index, mass, velocity_vector);
            }
            break;

        case Direction::Negative:
            if ((velocity > current_max_velocity) || (counter.num_atoms == 0))
            {
                set_atom_in_counter(counter, index, mass, velocity_vector);
            }
            break;
    }

    counter.velocity_total += velocity;
    counter.num_atoms++;
}


/*********************************
 * MPI SYNCHRONIZATION FUNCTIONS *
 *********************************/

/*! \brief Return the maximum box size from all nodes */
static real mpi_sync_box_size(real local_box_size, const t_commrec *cr)
{
    real global_box_size = local_box_size;

    MPI_Allreduce(
        &local_box_size, &global_box_size, 1, 
        GMX_MPI_REAL, MPI_MAX, 
        cr->mpi_comm_mysim);

    return global_box_size;
}
/* MPI Strategy:
   We want to sync the important information from all counters to all ranks.
   The ranks will then locally reduce this information into the swap indices
   and perform the exchange.

   For this we need to keep in mind that velocities need to be updated on 
   all ranks which own the swap atoms. This means that we need to work 
   with the global atom index, and transform that back to local before 
   doing an exchange.

   Basic setup:
    * Transmit all counters, with global atom indices and velocities
    * Determine top exchange candidates (will be same for all 
      ranks if they have the same counter information from all other)
    * Determine if a swap should be made
      - If so, check if the global atom indices of either exchange 
        candidate has a local index
      - If it does, set the velocity in it 
      - Else do nothing
    */

/*! \brief Synchronize the global atom indices from all nodes to all nodes and return

    The returned vector has elements
   
        [i0, i1, ..., in]

    where ij is the global atom index from node j. 
    
    Note: Input \p index_local *must* be the local index. This function transforms 
    it to a global index before synchronizing between nodes. */
static std::vector<int> 
mpi_sync_counter_global_indices(const int        local_index,
                                const t_commrec *cr)
{
    const int global_index = get_global_index_from_local(local_index, cr);
    const auto num_nodes = cr->nnodes;

    std::vector<int> send_buf (num_nodes, global_index);
    std::vector<int> recv_buf (num_nodes, 0);

    MPI_Alltoall(
        send_buf.data(), 1, MPI_INT, 
        recv_buf.data(), 1, MPI_INT, 
        cr->mpi_comm_mysim);

    return recv_buf;
}

/*! \brief Fill a buffer with the given \p velocity, where the velocity 
    repeats once for every node in \p num_nodes 
    
    The buffer will have size DIM * num_nodes and consist of: 

        [vx, vy, vz, vx, vy, vz, ..., vx, vy, vz] 
        |< -------   num_nodes repeats  ------ >|

    This buffer can easily be transmitted to other nodes using MPI_Alltoall. */   
static std::vector<real> fill_send_velocity_buffer(const Vec3   &velocity, 
                                                   const size_t  num_nodes)
{
    std::vector<real> buffer (DIM * num_nodes, 0.0);

    for (size_t i = 0; i < num_nodes; ++i)
    {
        for (size_t j = 0; j < DIM; ++j)
        {
            buffer.at(i * DIM + j) = velocity.at(j);
        }
    }

    return buffer;
}

/*! \brief Restructure a single dimension buffer into Vec3 arrays and return 

    The input buffer of velocity values will have the same form as from 
    fill_send_velocity_buffer, except with the velocity values from the 
    nodes after the MPI_Alltoall operation. The single dimension buffer 
    is restructured as: 

        [v0x, v0y, v0z, v1x, v1y, v1z, ..., vnx, vny, vnz] 
        ->
        [[v0x, v0y, v0z], [v1x, v1y, v1z], ..., [vnx, vny, vnz]]

    */
static std::vector<Vec3> 
construct_velocity_vectors_from_buffer(const std::vector<real> &buffer,
                                       const size_t             num_nodes)
{
    std::vector<Vec3> velocity_vectors;
    
    velocity_vectors.reserve(num_nodes);

    GMX_RELEASE_ASSERT(num_nodes * DIM == buffer.size(),
        "incorrect number of elements in sync'd velocity buffer");

    for (size_t i = 0; i < num_nodes; ++i)
    {
        velocity_vectors.push_back(Vec3 {
            buffer.at(i * DIM + XX),
            buffer.at(i * DIM + YY),
            buffer.at(i * DIM + ZZ)
        });
    }

    return velocity_vectors;
}

/*! \brief Synchronize the velocity_max_vectors for this exchange counter area 
    from all nodes
    
    The returned vector will be of form 

        [[v0x, v0y, v0z], [v1x, v1y, v1z], ..., [vnx, vny, vnz]]
    
    where vij is the velocity vector value from node i. */
static std::vector<Vec3> 
mpi_sync_counter_velocity_vectors(const Vec3      &local_velocity,
                                  const t_commrec *cr)
{
    auto send_buf = fill_send_velocity_buffer(
        local_velocity, static_cast<size_t>(cr->nnodes));

    std::vector<real> recv_buf (send_buf.size(), 0.0);

    MPI_Alltoall(
        send_buf.data(), DIM, GMX_MPI_REAL, 
        recv_buf.data(), DIM, GMX_MPI_REAL, 
        cr->mpi_comm_mysim);
    
    return construct_velocity_vectors_from_buffer(recv_buf, cr->nnodes);
}

/*! \brief Synchronize the number of atoms from all nodes to all nodes and return

    The returned vector has elements
   
        [n0, n1, ..., nn]

    where ni is the number of atoms from node i. */
static std::vector<int> 
mpi_sync_counter_num_atoms(const size_t     num_atoms,
                           const t_commrec *cr)
{
    std::vector<int> send_buf (cr->nnodes, static_cast<int>(num_atoms));
    std::vector<int> recv_buf (send_buf.size(), 0);

    MPI_Alltoall(
        send_buf.data(), 1, MPI_INT, 
        recv_buf.data(), 1, MPI_INT, 
        cr->mpi_comm_mysim);

    return recv_buf;
}

/*! \brief Synchronize a given value from all nodes to all nodes and return 

    The returned vector has elements
   
        [v0, v1, ..., vn]

    where vi is the number of atoms from node i. */
static std::vector<double> 
mpi_sync_counter_values(const double     value,
                        const t_commrec *cr)
{
    std::vector<double> send_buf (cr->nnodes, value);
    std::vector<double> recv_buf (send_buf.size(), 0);

    MPI_Alltoall(
        send_buf.data(), 1, MPI_DOUBLE, 
        recv_buf.data(), 1, MPI_DOUBLE, 
        cr->mpi_comm_mysim);

    return recv_buf;
}

/*! \brief From the sync'd buffers, reconstruct the exchange counters from all nodes 

    The returned vector will be of form

        [counter_0, counter_1, ..., counter_n]

    where counter_i is the exchange counter originally from node i. */
static std::vector<ExchangeCounter> 
construct_exchange_counters(const std::vector<int>    &global_inds,
                            const std::vector<int>    &num_atoms,
                            const std::vector<Vec3>   &velocity_vectors,
                            const std::vector<double> &total_velocities,
                            const std::vector<double> &atom_masses)
{
    GMX_RELEASE_ASSERT(global_inds.size() == num_atoms.size(), 
        "global_inds not same size as num_atoms after sync");
    GMX_RELEASE_ASSERT(global_inds.size() == velocity_vectors.size(), 
        "global_inds not same size as velocity_vectors after sync");
    GMX_RELEASE_ASSERT(global_inds.size() == total_velocities.size(), 
        "global_inds not same size as total_velocities after sync");
    GMX_RELEASE_ASSERT(global_inds.size() == atom_masses.size(), 
        "global_inds not same size as atom_masses after sync");

    const auto num_nodes = global_inds.size();

    std::vector<ExchangeCounter> counters;

    for (size_t i = 0; i < num_nodes; ++i)
    {
        ExchangeCounter counter;

        counter.index = global_inds.at(i);
        counter.num_atoms = num_atoms.at(i);
        counter.velocity_total = total_velocities.at(i);
        counter.atom_mass = atom_masses.at(i);
        counter.velocity_max_vector = velocity_vectors.at(i);

        counters.push_back(counter);
    }

    return counters;
}

/*! \brief Synchronize exchange counters from all nodes and return them 

    The returned vector will be of form

        [counter_0, counter_1, ..., counter_n]

    where counter_i is the exchange counter originally from node i.
    
    Note: After this synchronization, the index value for each counter 
    corresponds to the *global* atom index. Before, the node-local counters
    contain the *node-local* atom indices. */
static std::vector<ExchangeCounter> 
mpi_sync_counters(const ExchangeCounter &local_counter, 
                  const t_commrec       *cr)
{
    const auto inds = mpi_sync_counter_global_indices(
        static_cast<int>(local_counter.index), cr);
    
    const auto velocity_vectors = mpi_sync_counter_velocity_vectors(
        local_counter.velocity_max_vector, cr);

    const auto num_atoms = mpi_sync_counter_num_atoms(
        local_counter.num_atoms, cr);

    const auto total_vels = mpi_sync_counter_values(
        local_counter.velocity_total, cr);

    const auto atom_masses = mpi_sync_counter_values(
        local_counter.atom_mass, cr);
    
    return construct_exchange_counters(
        inds, num_atoms, velocity_vectors, total_vels, atom_masses);
}

/*! \brief Copy the atom velocity data \p from an exchange counter \p to a target 

    We do not touch the total velocity or number of atoms, since they are aggregated 
    separately. */
static void copy_counter_atom_velocities(const ExchangeCounter &from,
                                         ExchangeCounter       &to)
{
    to.index                 = from.index;
    to.atom_mass             = from.atom_mass;
    to.velocity_max_vector   = from.velocity_max_vector;
}

/*! \brief Reduce the sync'd exchange \p counters from all ranks to one for 
    the entire system */
static ExchangeAreaCounter 
get_final_area_counter(const std::vector<ExchangeCounter> &counters,
                       const ExchangeAreaCounter          &local_area_counter,
                       const ShearVelOpts                 &opts)
{
    ExchangeCounter final_counter;
    const auto direction = static_cast<size_t>(opts.direction);

    for (const auto& counter : counters)
    {
        const auto velocity       = counter.velocity_max_vector.at(direction);
        const auto final_velocity = final_counter.velocity_max_vector.at(direction);

        switch (local_area_counter.area.direction)
        {
            case Direction::Positive:
                if ( ((velocity < final_velocity) && (counter.num_atoms > 0))
                    || (final_counter.num_atoms == 0) )
                {
                    copy_counter_atom_velocities(counter, final_counter);
                }
                
                break;
            
            case Direction::Negative:
                if ( ((velocity > final_velocity) && (counter.num_atoms > 0))
                    || (final_counter.num_atoms == 0) )
                {
                    copy_counter_atom_velocities(counter, final_counter);
                }

                break;
        }

        final_counter.velocity_total += counter.velocity_total;
        final_counter.num_atoms      += counter.num_atoms;
    }

    return ExchangeAreaCounter {
        local_area_counter.area,
        final_counter
    };
}


/************************************
 * ATOM VELOCITY EXCHANGE FUNCTIONS *
 ************************************/

/*! \brief Return the average atom velocity measured by \p counter */
static real avg_velocity(const ExchangeCounter &counter)
{
    return counter.velocity_total / static_cast<real>(counter.num_atoms);
}

/*! \brief Return whether or not an atom velocity exchange should be made */
static bool check_if_exchange(const ExchangeAreaCounter &area_counter0,
                              const ExchangeAreaCounter &area_counter1,
                              const int64_t              step)
{
    static size_t num_warns = 0;
    constexpr size_t num_warns_max = 20;

    const auto& counter0 = area_counter0.counter;
    const auto& counter1 = area_counter1.counter;

    if ((counter0.num_atoms == 0) || (counter1.num_atoms == 0)) 
    {
        num_warns++;

        if (num_warns < num_warns_max)
        {
            if (counter0.num_atoms == 0)
            {
                gmx_warning(
                    "step %lu: no atoms in area 0, cannot exchange",
                    step);
            }
            if (counter1.num_atoms == 0) 
            {
                gmx_warning(
                    "step %lu: no atoms in area 1, cannot exchange",
                    step);
            }
        }
        else if (num_warns == num_warns_max)
        {
            gmx_warning(
                "step %lu: Reached maximum number of warnings (%d) for no shear "
                "exchange reached, shutting up now. Are the areas correctly "
                "defined to contain the liquids?",
                step, num_warns_max);
        }

        return false;
    }

    const auto& area0 = area_counter0.area;
    const auto& area1 = area_counter1.area;

    const auto target_diff = area1.target_velocity - area0.target_velocity;
    const auto diff = avg_velocity(counter1) - avg_velocity(counter0);

    return (target_diff > diff);
}

static void set_velocity(t_state                     *state,
                         const ExchangeCounter       &from,
                         const ExchangeCounter       &to,
                         const t_commrec             *cr)
{
    const auto to_index_global = to.index;

    GMX_RELEASE_ASSERT(to.atom_mass != 0.0, 
        "An ExchangeCounter has an exchanging atom with mass = 0, "
        "which is not allowed. Check the atom groups for shear coupling "
        "and ensure that they do not contain virtual atoms.");
    const auto mass_fac = sqrt(from.atom_mass / to.atom_mass);

    /* Check if this global index exists on this rank
       and which local index it corresponds to */
    int to_index_local;
    if (get_local_index_from_global(to_index_local, to_index_global, cr))
    {
        const auto& velocity = from.velocity_max_vector;

        for (size_t i = 0; i < DIM; ++i)
        {
            state->v[to_index_local][i] = velocity[i] * mass_fac;
        }
    }
}

static void exchange_velocities(t_state               *state,
                                const ExchangeCounter &counter0,
                                const ExchangeCounter &counter1,
                                const t_commrec       *cr)
{
    set_velocity(state, counter0, counter1, cr);
    set_velocity(state, counter1, counter0, cr);
}


/***************************************** 
 * SETUP FUNCTION FOR THE EXCHANGE AREAS *
 *****************************************/

/*! \brief Determine and set the current exchange areas \p area0 and \p area1

    The areas are created depending on the selected strategy and the current 
    box size. */
static void set_exchange_areas(ExchangeArea       &area0,
                               ExchangeArea       &area1,
                               const ShearVelOpts &opts, 
                               const matrix        box,
                               const t_commrec    *cr) 
{
    const auto axis = static_cast<size_t>(opts.axis);

    const auto local_box_size = box[axis][axis];
    const auto box_size = mpi_sync_box_size(local_box_size, cr);

    area0.target_velocity = -opts.ref_velocity;
    area1.target_velocity = opts.ref_velocity;
    area0.direction = Direction::Negative;
    area1.direction = Direction::Positive;

    switch (opts.strategy) {
        case ShearCouplStrategy::Edges: 
            area0.zmin = opts.zedge_adj;
            area0.zmax = area0.zmin + opts.size;

            area1.zmax = box_size - opts.zedge_adj;
            area1.zmin = area1.zmax - opts.size;

            break;
        
        case ShearCouplStrategy::EdgeCenter: 
            area0.is_split = true;
            area0.zmin = opts.zedge_adj;
            area0.zmax = area0.zmin + opts.size / 2.0;
            area0.zmax2 = box_size - opts.zedge_adj;
            area0.zmin2 = area0.zmax2 - opts.size / 2.0;

            {
                const auto zmid = box_size / 2.0;
                area1.zmin = zmid - opts.size / 2.0;
                area1.zmax = zmid + opts.size / 2.0;
            }

            break;
        
        default:
            gmx_fatal(FARGS, 
                      "Invalid ShearCouplStrategy selected. "
                      "This should not be possible.");
            break;
    }

    area0.group = 0;

    switch (opts.num_groups)
    {
        case 1:
            area1.group = 0;
            break;
        case 2:
            area1.group = 1;
            break;
        default:
            gmx_fatal(FARGS, 
                    "Number of shear-grps was %d, not 1 or 2. "
                    "This should not be possible.",
                    opts.num_groups);
    }
}


/*************************************
 * LOGGING AND TEXT OUTPUT FUNCTIONS *
 *************************************/

static void log_shear_area_info(const ExchangeArea     &area,
                                const size_t            i,
                                const ShearVelOpts     &opts,
                                const SimulationGroups *groups,
                                const gmx::MDLogger    &mdlog)
{
    const auto global_group_index 
        = groups->groups[SimulationAtomGroupType::User2].at(area.group);

    if (!area.is_split)
    {
        GMX_LOG(mdlog.info)
            .appendTextFormatted(
                "  Area %d:\n"
                "    Group:                      %s\n"
                "    Range (along %s):           [%g, %g]\n"
                "    Target velocity (along %s): %g\n",
                i,
                *groups->groupNames[global_group_index],
                get_axis_name(opts.axis), 
                area.zmin, area.zmax, 
                get_axis_name(opts.direction),
                area.target_velocity); 
    }
    else
    {
        GMX_LOG(mdlog.info)
            .appendTextFormatted(
                "  Area %d:\n"
                "    Group:                      %s\n"
                "    Range (along %s):           [%g, %g] & [%g, %g]\n"
                "    Target velocity (along %s): %g\n",
                i,
                *groups->groupNames[global_group_index],
                get_axis_name(opts.axis), 
                area.zmin, area.zmax, 
                area.zmin2, area.zmax2,
                get_axis_name(opts.direction),
                area.target_velocity); 
    }
}

static void log_shear_coupling_info(const ShearVelOpts     &opts,
                                    const double            tcoupl,
                                    const matrix            box,
                                    const SimulationGroups *groups,
                                    const t_commrec        *cr,
                                    const gmx::MDLogger    &mdlog)
{
    const auto axis_name = get_axis_name(opts.axis);
    const auto direction_name = get_axis_name(opts.direction);

    GMX_LOG(mdlog.info).appendText("");

    GMX_LOG(mdlog.info)
        .appendTextFormatted(
            "Shear velocity coupling is turned on.\n"
            "  Axis:            %s\n"
            "  Shear direction: %s\n"
            "  Time coupling:   %g (every %d steps)\n"
            "  Shear area size: %g\n"
            "  Edge adjustment: %g\n"
            "  ---\n"
            "  Shear creation zones at start (changes with box size):", 
            axis_name, 
            direction_name,
            tcoupl, 
            opts.step, 
            opts.size,
            opts.zedge_adj);
    
    ExchangeArea area0, area1;
    set_exchange_areas(area0, area1, opts, box, cr);

    log_shear_area_info(area0, 0, opts, groups, mdlog);
    log_shear_area_info(area1, 1, opts, groups, mdlog);

    GMX_LOG(mdlog.info).appendText("");
}

/* Log momentum exchange information along the flow direction */
static void log_exchange(FILE                  *fp,
                         const double           time,
                         const ExchangeCounter &counter0,
                         const ExchangeCounter &counter1,
                         const Axis             direction)
{
    const auto d = static_cast<size_t>(direction);
    const auto p0 = counter0.atom_mass * counter0.velocity_max_vector.at(d);
    const auto p1 = counter1.atom_mass * counter1.velocity_max_vector.at(d);

    fprintf(fp, "%12.5e %12.5e\n", time, p1 - p0);
}

#ifdef MPI_SHEAR_DEBUG
static void print_shear_area_info(const ExchangeArea     &area,
                                  const size_t            i,
                                  const ShearVelOpts     &opts,
                                  const SimulationGroups *groups)
{
    const auto global_group_index 
        = groups->groups[SimulationAtomGroupType::User2].at(area.group);

    if (!area.is_split)
    {
        fprintf(stderr, 
                "  Area %d:\n"
                "    Group:                      %s\n"
                "    Range (along %s):           [%g, %g]\n"
                "    Target velocity (along %s): %g\n",
                i,
                *groups->groupNames[global_group_index],
                get_axis_name(opts.axis), 
                area.zmin, area.zmax, 
                get_axis_name(opts.direction),
                area.target_velocity); 
    }
    else
    {
        fprintf(stderr, 
                "  Area %d:\n"
                "    Group:                      %s\n"
                "    Range (along %s):           [%g, %g] & [%g, %g]\n"
                "    Target velocity (along %s): %g\n",
                i,
                *groups->groupNames[global_group_index],
                get_axis_name(opts.axis), 
                area.zmin, area.zmax, 
                area.zmin2, area.zmax2,
                get_axis_name(opts.direction),
                area.target_velocity); 
    }
}

static void print_shear_coupling_info(const ShearVelOpts     &opts,
                                      const ExchangeArea     &area0,
                                      const ExchangeArea     &area1,
                                      const double            tcoupl,
                                      const matrix            box,
                                      const SimulationGroups *groups,
                                      const t_commrec        *cr)
{
    const auto axis_name = get_axis_name(opts.axis);
    const auto direction_name = get_axis_name(opts.direction);

    fprintf(stderr, "\n");

    fprintf(stderr, 
            "[Rank %d] Shear velocity coupling is turned on.\n"
            "  Axis:            %s\n"
            "  Shear direction: %s\n"
            "  Time coupling:   %g (every %d steps)\n"
            "  Shear area size: %g\n"
            "  Edge adjustment: %g\n"
            "  ---\n"
            "  Shear creation zones at start (changes with box size):\n", 
            cr->nodeid,
            axis_name, 
            direction_name,
            tcoupl, 
            opts.step, 
            opts.size,
            opts.zedge_adj);
    
    print_shear_area_info(area0, 0, opts, groups);
    print_shear_area_info(area1, 1, opts, groups);

    fprintf(stderr, "\n");
}

static void print_single_counter(const ExchangeCounter &counter)
{
    fprintf(stderr, "  velocity_total        = %f\n", counter.velocity_total);
    fprintf(stderr, "  index                 = %d\n", counter.index);
    fprintf(stderr, "  atom_mass             = %f\n", counter.atom_mass);
    fprintf(stderr, "  num_atoms             = %d\n", counter.num_atoms);
    fprintf(stderr, "  velocity_max_vector   = (%f, %f, %f)\n", 
        counter.velocity_max_vector[XX], 
        counter.velocity_max_vector[YY], 
        counter.velocity_max_vector[ZZ]);
}

static void 
print_counter_information(const ExchangeCounter              &local_counter,
                          const ExchangeCounter              &final_counter,
                          const int                           area_num,
                          const std::vector<ExchangeCounter> &counters,
                          const t_commrec                    *cr)
{
    fprintf(stderr, "----- Rank %d, area %d -----\n", cr->sim_nodeid, area_num);

    fprintf(stderr, "Local counter:\n");
    print_single_counter(local_counter);
    fprintf(stderr, "\n");

    for (size_t i = 0; i < counters.size(); ++i)
    {
        fprintf(stderr, "Counter %d:\n", i);
        print_single_counter(counters.at(i));
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "Final counter:\n");
    print_single_counter(final_counter);
    fprintf(stderr, "----- END -----\n\n");
}
#endif


/*************************************************************************
 * PUBLIC FUNCTIONS WHICH SET UP AND PERFORM THE SHEAR VELOCITY COUPLING *
 *************************************************************************/

ShearVelOpts init_shear_velocity_coupling_opts(const t_inputrec       *ir,
                                               const matrix            box,
                                               const SimulationGroups *groups,
                                               const t_commrec        *cr,
                                               const char             *fnlog,
                                               const struct gmx_output_env_t *oenv,
                                               const gmx::MDLogger    &mdlog)
{
    const gmx_bool bShearCoupl = ir->bShearCoupling;
    const auto tcoupl = static_cast<double>(ir->shear_tcoupl);
    const auto nstcoupl = static_cast<size_t>(tcoupl / ir->delta_t);

    if (bShearCoupl && (fabs(nstcoupl * ir->delta_t - tcoupl) > 0.000001))
    {
        gmx_warning(
            "shear_tcoupl (%f) is not a multiple of the timestep (%f), "
            "setting it to %f.",
            tcoupl, ir->delta_t, static_cast<double>(nstcoupl) * ir->delta_t);
    }

    const auto axis = get_axis(static_cast<ShearAxis_axis>(ir->shear_axis));
    const auto direction = get_direction(static_cast<ShearAxis_direction>(ir->shear_direction));

    FILE *fp = nullptr;
    if (bShearCoupl)
    {
        char xaxis[STRLEN], 
             yaxis[STRLEN],
             subtitle[STRLEN];

        snprintf(xaxis, STRLEN, 
                 "t (%s)", 
                 unit_time);

        snprintf(yaxis, STRLEN, 
                 "\\Deltap\\s%s\\N (%s %s %s\\S-1\\N)", 
                 get_axis_name(direction),
                 unit_mass, unit_length, unit_time);

        fp = xvgropen_type(fnlog, "Momentum exchange", xaxis, yaxis, exvggtNONE, oenv);

        snprintf(subtitle, STRLEN,
                 "From area 1 to area 0 (p\\s%s,1\\N - p\\s%s,0\\N)",
                 get_axis_name(direction), get_axis_name(direction));
        xvgr_subtitle(fp, subtitle, oenv);
    }

    const ShearVelOpts opts {
        fp,
        axis,
        direction,
        static_cast<ShearCouplStrategy>(ir->shear_strategy),
        get_num_groups(groups),
        nstcoupl,
        ir->shear_area_size,
        ir->shear_zadj,
        ir->shear_ref_velocity
    };

    if (bShearCoupl)
    {
        if (MASTER(cr))
        {
            gmx_warning("Shear velocity coupling may not yet work with "
                        "dedicated PME nodes!");
        }
        log_shear_coupling_info(opts, tcoupl, box, groups, cr, mdlog);

#ifdef MPI_SHEAR_DEBUG
        if (MASTER(cr))
        {
            gmx_warning("Shear debugging output is turned on. This will "
                        "massively impact performance.");
        }

        ExchangeArea area0, area1;
        set_exchange_areas(area0, area1, opts, box, cr);

        for (size_t nodeid = 0; nodeid < cr->nnodes; nodeid++)
        {

            if (cr->sim_nodeid == nodeid)
            {
                print_shear_coupling_info(opts, area0, area1, tcoupl, box, groups, cr);
            }

            std::this_thread::sleep_for(MPI_SHEAR_SLEEP_DURATION);
            MPI_Barrier(cr->mpi_comm_mysim);
        }
#endif
    }

    return opts;
}

void do_shear_velocity_coupling(t_state                *state,
                                const t_mdatoms        *mdatoms,
                                const int64_t           current_step,
                                const double            current_time,
                                const ShearVelOpts     &opts,
                                const SimulationGroups *groups,
                                const t_commrec        *cr)
{
    ExchangeArea area0, area1;
    set_exchange_areas(area0, area1, opts, state->box, cr);

    ExchangeAreaCounter area_counter0(area0),
                        area_counter1(area1);

    for (size_t i = 0; i < mdatoms->homenr; ++i)
    {
        const auto z = get_position_in_box(state, i, opts.axis);

        if (in_counter_area(i, z, area_counter0, groups, cr))
        {
            add_atom_velocity(area_counter0, i, opts, state, mdatoms);
        }
        if (in_counter_area(i, z, area_counter1, groups, cr))
        {
            add_atom_velocity(area_counter1, i, opts, state, mdatoms);
        }
    }

    const auto all_rank_counters0 = mpi_sync_counters(area_counter0.counter, cr);
    const auto all_rank_counters1 = mpi_sync_counters(area_counter1.counter, cr);

    const auto system_area_counter0 = get_final_area_counter(
        all_rank_counters0, area_counter0.area, opts);
    const auto system_area_counter1 = get_final_area_counter(
        all_rank_counters1, area_counter1.area, opts);

#ifdef MPI_SHEAR_DEBUG
    for (size_t nodeid = 0; nodeid < cr->nnodes; ++nodeid)
    {
        if (cr->sim_nodeid == nodeid)
        {
            print_counter_information(
                area_counter0.counter, system_area_counter0.counter, 
                0, all_rank_counters0, cr);
            print_counter_information(
                area_counter1.counter, system_area_counter1.counter, 
                1, all_rank_counters1, cr);
        }

        std::this_thread::sleep_for(MPI_SHEAR_SLEEP_DURATION);
        MPI_Barrier(cr->mpi_comm_mysim);
    }
#endif

    if (check_if_exchange(system_area_counter0, system_area_counter1, current_step))
    {
        exchange_velocities(
            state, 
            system_area_counter0.counter, 
            system_area_counter1.counter,
            cr);

        if (MASTER(cr))
        {
            log_exchange(
                opts.log_pexchange,
                current_time,
                system_area_counter0.counter, 
                system_area_counter1.counter,
                opts.direction);
        }
    }
}