# Stationary Diffusion State ML Surrogate

![alt text](https://github.com/jquetzalcoatl/DiffusionSurrogate/blob/master/Figs/Fig1.png)

This project aims at training a CNN that receives as input an initial condition of a Diffusion-like equation and outputs the stationary solution.

First we generate the (initial condition)-(stationary state) tuples from terminal as:

```
$ julia genData.jl <index 1> <index 2> <dir name> <number of sources> <path out>
```

where data_set_size = index_2 - index_1.
