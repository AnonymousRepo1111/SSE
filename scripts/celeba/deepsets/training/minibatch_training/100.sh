GPU=0
for RUN in 0
do
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python training/cnp.py \
    --run $RUN \
    --hidden_dim 64 \
    --dataset celeba \
    --size 32 32	\
    --num_points 100 \
    --epochs 200 \
    --aggregator mean &
done
