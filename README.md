# Instance-Wise Monotonic Calibration by Constrained Transformation

This repository contains the code for our UAI 2025 submission: *Instance-Wise Monotonic Calibration by Constrained Transformation*. The implementation of the proposed calibration methods, MCCT and MCCT-I, can be found in `Mono_cali.py`, specifically in the classes `MCCT` and `MCCT-I`.

The file `parallelized_benchmark.py` contains the experiment code for benchmarking, including all baseline settings. For all baseline methods, we use the hyperparameters specified in their original papers. If the hyperparameters are not provided in the paper, we use those from the official implementation.  

The shell script `run_pal.sh` is provided for running `parallelized_benchmark.py` in parallel.
