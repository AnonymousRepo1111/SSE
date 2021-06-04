GPU=0
for RUN in 0
do
  for EPOINTS in 2000 3000 4000
  do
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python testing/cnp.py \
      --run $RUN \
      --hidden_dim 64 \
      --dataset celeba \
      --size 64 64	\
      --num_points 1000 \
      --aggregator mean \
      --eval_mode all \
      --num_points_eval $EPOINTS
  done
done
