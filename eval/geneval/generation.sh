#!/bin/bash


# conda  activate  blip3o


# export HF_HOME=/fsx/sfr/data/jiuhai




MODEL="/fsx/sfr/data/jiuhai/hub/models--BLIP3o--BLIP3o-Model/snapshots/3d48160921990db9e65c0dfab6915fe08e6c1565"



# Total number of GPUs/chunks.
N_CHUNKS=1

# Launch processes in parallel for each GPU/chunk.
for i in $(seq 0 $(($N_CHUNKS - 1))); do
    echo "Launching process for GPU $i (chunk index $i of $N_CHUNKS)"
    CUDA_VISIBLE_DEVICES=$i python generate.py --model "$MODEL" --index $i --n_chunks $N_CHUNKS &
done

# Wait for all background processes to finish.
wait
echo "All background processes finished."


