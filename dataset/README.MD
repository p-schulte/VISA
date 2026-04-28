# Dataset Generator submodule

This submodule is for generating our datasets. How to run and execute is explained in the main README file. Here we just provide an overview over this subdirectory for better understanding.

The main file is the `generate.py` file. It loads the configuration from the environment config file as mentioned in the global README file. Then, it generates the specified sequence length of synthetic planning domain images. It does so by calling the respective domains random scene generator, initialized with a random state and goal. Then, it computes a plan using the fast-downward planner and executes it. Along the way, it renders the images, using a per-domain specified rendering engine and logs the planning state. This logging is then captued as our desired dataset.


## New Domains

Adding a new domain can be achieved by first specifying the desired pddl domain in the `pddl_files` folder. Then, the random problem generation and pddl initialization have to be defined in the `lib/sequence_generation` folder. Finally, a custom rendering method has to be added to the `lib/rendering` folder for the new domain. Afterwards, it can be easily added to the generation file.