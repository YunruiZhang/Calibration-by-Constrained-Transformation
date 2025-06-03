#!/bin/bash

# Define the options for the parameters
method_options=(0 1 2 3 4 5 6 7)
seed_options=(1 2 3 4 5 6 7 8 9 10)
# Generate all combinations of the parameter options
combinations=()
for method in "${method_options[@]}"; do
  for seed in "${seed_options[@]}"; do
    combinations+=("--method $method --seed $seed")
  done
done


# Export the script path for use in parallel
script="parallelized_benchmark.py"

# Execute the combinations in parallel using xargs with a configurable number of parallel jobs
num_jobs=${1:-4}  # Default to 4 parallel jobs if not specified

# printf "%s\n" "${combinations[@]}"
printf "%s\n" "${combinations[@]}" | xargs -n 1 -P "$num_jobs" -I {} sh -c "python $script {}"
