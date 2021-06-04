GPU=0
for RUN in 0
do
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python training/cnp.py \
    --run $RUN \
    --hidden_dim 64 \
    --dataset celeba \
    --size 64 64	\
    --num_points 500 \
    --epochs 200 \
    --aggregator mean 
done
