# Diffusional-Fingerprinting
This repository contains code which allows for the generation and analysis of Diffusional Fingerprints. 
The repository consists of three library files and one example file, displaying how the functions may be used. 

1. `RandomWalkSims.py`<br/>
Contains functions for simulating four types of random walks: Brownian motion, confined diffusion, directed motion and anomalous diffusion. 
The simulation methodology follows that of P. Kowalek, H. Loch-Olszewska, and J. Szwabiński, Physical Review E 100, (2019).

2. `Fingerprint_feat_gen.py`<br/>
Contains functions which will compute the diffusional fingerprint for 2d random walk trajectories. 

3. `MLGeneral.py`<br/>
Contains functions to train machine learning classifiers and plot dimensionality reduced fingerprints. 
Furthermore, it includes code to rank features based on importance. 
The core functional unit is a class called ML which handles all the machine learning through method calls.

4. `Usage_example.py`<br/>
Example script showing how to simulate four types of random walk trajectories, compute their fingerprints, and train a machine learning classifier to seperate them. 
The example also contains a few plotting capabilities, showcasing how one could analyze the output results. 

## Other files in the repository
The other files in the repository are non-vital files which increase the convenience in running the example and using the libraries. 
The files `HMMjson`, `X_fingerprints.npy`, `X.pkl`, and `y.pkl` are pre-computed fingerprints, a pre-trained HMM model, and pre-simulated data which reduce the runtime of the example. 
If the user wish to generate their own data, any of these files may be deleted in which case the `Usage_example.py` will generate new instances of these files upon runtime.
The repository also contains a `requirements.txt` showcasing a dependency setup in which the code ran. 
It is expected that other versions for the required dependencies would work, but for example a 2.x version of iminuit is required over a 1.x version which will not run. 
