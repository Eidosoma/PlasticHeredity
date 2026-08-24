# AI-Guided Resetting of Memories in Gene Regulatory Network Models: biomedical and evolutionary implications
This is the official repository for the paper (under review), hosting all the code for replication.

## Installation
Clone the repo:
```
https://github.com/pigozzif/BreakingGRNMemories.git
```
Install Python dependencies with pip:
```
pip install -r requirements.txt
```

## Scope
By running:
```
python plotting.py
```
You will produce all the figures plotting experimental data from the paper (3, 6, 7, 8, and 9) and save them in the directory `figures`. All the raw data necessary are stored in the `output` directory.

Also:
```
python train.py {args}
```
will launch an optimization of a target memory (see `memories` for the full directory of memories available) to reset, using a specific optimizer. Available optimizers are es, ga, rl, and single (the baseline treatment). See `utils.py` for more information about the arguments.

Moreover, running:
```
python phi.py
```
will measure the causal emergence of each memory (whether reset or not).

Finally, running:
```
python optima.py
```
will simulate several synthetic networks and compute their properties, after which the approximate optimal networks can be estimated.
