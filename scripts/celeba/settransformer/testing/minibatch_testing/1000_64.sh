ROOT="/st1/bruno/Datasets/CelebA"

GPU=0
for RUN in 0
do
  for EPOINTS in 100 200 300 400 500
  do
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python testing/cnp.py \
      --root $ROOT \
      --run $RUN \
      --dataset celeba \
      --num_points 1000 \
      --size 64 64  \
      --aggregator settransformer \
      --hidden_dim 64 \
      --s_hidden_dim 64 \
      --num_seeds 1 \
      --num_heads 1 \
      --eval_mode all \
      --num_points_eval $EPOINTS
  done
done
