#ifndef MD_PETTER
#define MD_PETTER

// Indices for X and Z axes, to not confuse with XX and YY
typedef enum {
    xi,
    zi,
    ni
} Axes;

// Indices for different data in array
typedef enum {
    NumAtoms,
    Temp,
    Mass,
    UU,         // Mass flow along XX
    VV,         //             and ZZ
    NumVar
} Index;

typedef struct flowdata {
    char    fnbase[STRLEN];
    int     step_collect,
            step_output,
            step_ratio,
            num_bins[ni];     // Number of bins in X and Z
    float   bin_size[ni],     // Bin sizes in X and Z
            inv_bin[ni];      // Inverted bin sizes for grid calculations
    double  *data;            // A 2D grid is represented by this 1D array
} t_flow_container;

// Prepare and return a container for flow field data
t_flow_container *
get_flow_container(const t_commrec  *cr,
                   const int         nfile,
                   const t_filenm    fnm[],
                   const t_inputrec *ir,
                   const t_state    *state);

// If at a collection or output step, perform actions
void
flow_collect_or_output(t_flow_container   *flowcr,
                       const gmx_int64_t   step,
                       const t_commrec    *cr,
                       const t_inputrec   *ir,
                       const t_mdatoms    *mdatoms,
                       const t_state      *state,
                       const gmx_groups_t *groups);

#endif
