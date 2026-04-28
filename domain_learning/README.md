To build the container run:  
`apptainer build ../graph-separator.sif graph-seperator.def`  
To use the container on a single python comand use:  
`apptainer run ../graph-separator.sif graph_seperator.py -d pddl_files/<domain> -i pddl_files/<instance>`  
To get an interactive shell use:  
`apptainer shell ../graph-separator.sif`
