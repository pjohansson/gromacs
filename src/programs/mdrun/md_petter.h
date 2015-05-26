#ifndef MD_PETTER
#define MD_PETTER
#endif

typedef enum {
    xi,
    zi,
    ni
} Axes;

typedef enum {
    NumAtoms,
    Temp,
    Mass,
    FlowUU,
    FlowVV,
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
} t_flowdata;

t_flowdata*
prepare_flow_field_data(t_commrec *cr, int nfile, const t_filenm fnm[],
			t_inputrec *ir, t_state *state);

void
collect_flow_data(t_flowdata *flow_data, t_commrec *cr, t_mdatoms *mdatoms,
		  t_state *state, gmx_groups_t *groups);

void
output_flow_data(t_flowdata *flow_data, t_commrec *cr, gmx_int64_t step);

void
check_flow_data_out(gmx_int64_t step, t_flowdata *flow_data, t_commrec *cr,
        	    t_mdatoms *mdatoms, t_state *state, gmx_groups_t *groups);
