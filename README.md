# Gromacs Flow Field 

This fork of [Gromacs](http://www.gromacs.org/) modifies it to enable output 
of two-dimensional flow fields, collected during a simulation. 

## Installation

Follow the regular Gromacs [installations instructions](http://manual.gromacs.org/current/install-guide/index.html). 
No changes are needed.

## Usage

The flow field collection uses the following [MDP options](http://manual.gromacs.org/documentation/current/user-guide/mdp-options.html):

```mdp
; How often to sample the velocity field
userint1 = 40 

; How often to save the velocity field to disk
userint2 = 5000

; Number of bins along the x axis
userint3 = 200

; Number of bins along the z axis
userint4 = 100

; List of groups to collect data for
user1-grps = SOL ; water 
```

Additionally, `mdrun` needs to know where to save the output, for which the `-flow` option has been added:

```bash
$ gmx mdrun -flow maps/flow   # Saves to `maps/flow_00001.dat`, 
                              #          `maps/flow_00002.dat`,
                              #          `maps/flow_00003.dat`,
                              # etc.
```

## Limitations

*   Can currently only sample data in the x-z plane.
*   Temperature calculation is only correct for water
*   Currently only works for static box size

## File formats

Documentation to come.

## License

These changes are distributed under the same terms as Gromacs. See 
[COPYING](COPYING) for more information.