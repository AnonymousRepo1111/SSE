GPU=0
for RUN in 0
do
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python training/cnp.py \
    --run $RUN \
    --dataset celeba \
    --num_points 500 \
    --size 64 64  \
    --epochs 200 \
    --aggregator settransformer \
    --hidden_dim 64 \
    --s_hidden_dim 64 \
    --num_seeds 1 \
    --num_heads 1 
done
