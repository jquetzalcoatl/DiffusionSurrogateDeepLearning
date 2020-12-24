# Stationary Diffusion State ML Surrogate using Flux and CuArrays.

![alt text](https://github.com/jquetzalcoatl/DiffusionSurrogate/blob/master/Figs/Fig1.png)

This project aims at training a CNN that receives as input an initial condition of a Diffusion-like equation and outputs the stationary solution.

First we generate the (initial condition)-(stationary state) tuples from terminal as:

```
$ julia genData.jl <index 1> <index 2> <dir name> <number of sources> <path out>
```

where data_set_size = index_2 - index_1.

Then, from terminal:

```
$ julia run_script.jl -w <weight> --n <dir> --loss <loss function> --e <epochs> --snap <snapshot> --p1 <p1> --p2 <p2> -c <D1> <D1> <D1> <D1>
```

where:
```
<weight> helps balance the low and high field values (default = 1).
<dir> Directory where things are saved (default = 1).
<loss function> MSE = 0, MAE = 2, Foruth Power = 1 (default = 0).
<epochs> (default = 1000)
<snapshots> Saves model, dictionary and plots after <snapshots> epochs (default = 500)
<p1> Think of it as a Boolean variable that can either turn off (0) or on (1) NN 1. But it is actually a Float32. (default = 1.0f0)
<p2> Think of it as a Boolean variable that can either turn off (0) or on (1) NN 2. But it is actually a Float32. (default = 1.0f0)
<continue> 0: starts new training. 1: Continues training loading a back-up model or a model saved from one of the snapshots. (default = 0)
<D_i> Dropbout values (default = (0.2,0.2,0.2,0.4)).
```

Before training, must specify path to data and path to output dir in run_script.jl code (lines 6 and 7):
```
PATH_in = "/PATH/TO/GENERATED/DATA/TwoSources/"
PATH_out = "/PATH/OUT/"
```

The dictionary stores a lot of parameters including the indexes of the training set, test set and validation.

Then,
```
$ julia get_stats.jl
```
generate a bunch of statistics plot (I know it's vague...).

Before this, must specify path to data and path to output dir in get_stats.jl code (lines 6 and 7):
```
PATH_in = "/PATH/TO/GENERATED/DATA/TwoSources/"
PATH_out = "/PATH/OUT/"
```

