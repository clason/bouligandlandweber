# bouligandlandweber

This repository contains implementations of the iterative regularization method described in the paper
[Bouligand-Landweber iteration for a non-smooth ill-posed problem](https://arxiv.org/abs/1803.02290)
by Christian Clason and Vũ Hữu Nhự.

### Python code

The results in the paper were generated using the provided Python implementation (`bouligandlandweber.py`) (with Python 3.6.5, Numpy 1.15.1, and Scipy 1.1.0) with the following notable differences to allow faster testing:

1. The current code defaults to `N=128`, while the reported results were obtained with `N=512` (see line 24).

2. Plotting is disabled by default but can be enabled by setting `figs = True`  in line 27.

3. Warm starting is used in the semi-smooth Newton method but can be disabled by replacing `F(un,yn)` by `F(un)` in line 140.

To run a representative example (`N=128`, `delta = 1e-4`, `beta=0.005`), run `python3 bouligandlandweber.py`.


### Julia code

We also provide an equivalent [Julia](https://julialang.org) (version 1.0) implementation in the module `BouligandLandweber.jl`. To run the same example, start `julia` in the same directory as the module and enter
```julia
include("./BouligandLandweber.jl")
BouligandLandweber.run_example(128,1e-4,0.005);
```
(Note that the first time this is done, the Julia code will be compiled to native code; subsequent calls to `run_example` (even with changed parameters) will then be much faster.)


### Reference

If you find this code useful, you can cite the paper as

    @article{bouligandlandweber,
        author = {Clason, Christian and Nhu, Vu Huu},
        title = {Bouligand-Landweber iteration for a non-smooth ill-posed problem},
        year = {2018},
        eprinttype = {arxiv},
        eprint = {1803.02290},
    }


