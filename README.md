# GeSDD

GeSDD is a genetic algorithm for learning Markov logic networks (MLNs). This project was developed by me, Michiel Baptist
as part of my master's thesis. The thesis text can be found as thesis.pdf in this repository.

# Requirements
- Python 3.6 >
- SDD 2.0 available at: [http://reasoning.cs.ucla.edu/sdd/](http://reasoning.cs.ucla.edu/sdd/)
- PySDD, a python wrapper for the SDD library: [https://github.com/wannesm/PySDD](https://github.com/wannesm/PySDD)
- The requirements in requirements.txt

# Getting started
1. Follow the instructions for compiling the PySDD library from source code: [https://github.com/wannesm/PySDD](https://github.com/wannesm/PySDD)
2. Install the requirements in requirements.txt: 
```pip install requirements.txt``` 
using a conda environment is highly advised.
3. Test the genetic algorithm:
```./run_arguments_example```
4. Check out the training results in ```run/xyz/```
