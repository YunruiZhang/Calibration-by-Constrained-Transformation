# Instance-Wise Monotonic Calibration by Constrained Transformation

This repository contains the official implementation of our UAI 2025 paper: *Instance-Wise Monotonic Calibration by Constrained Transformation*. The proposed calibration methods, **MCCT** and **MCCT-I**, are implemented in `Mono_cali.py`, specifically in the classes `MCCT` and `MCCT_I`. A simple example demonstrating how to use them is provided below.


```python  
mcct = MCCT(topk=num_classes, maxiter=100, bounds=False, filter=False)
mcct.fit(logits_vali, one_hot_vali_labels)
mcct_res = mcct.predict(test_logits)

mcct_i = MCCT_I(topk=num_classes, maxiter=100, bounds=False, filter=False)
mcct_i.fit(logits_vali, one_hot_vali_labels)
mcct_i_res = mcct_i.predict(test_logits)

```

The file `parallelized_benchmark.py` contains the experiment code for benchmarking, including experiments for all baselines.

The shell script `run_pal.sh` is provided to run `parallelized_benchmark.py` in parallel. The `env.yml` file specifies the environment required to run the benchmark. However, to run MCCT and MCCT-I, you only need `numpy`, `scikit-learn`, and `scipy`.

