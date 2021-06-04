GPU=0
for RUN in 0
do
  for EPOINTS in 2000 3000 4000
  do
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python testing/cnp.py \
      --run $RUN \
      --dataset celeba \
      --num_points 1000 \
      --size 64 64  \
      --lr 1e-3 \
      --aggregator consistent \
      --ln True \
      --hidden_dim 64 \
      --K 1 \
      --h 128 \
      --d 64  \
      --d_hat 64 \
      --g sum \
      --_slots Learned \
      --eval_mode all \
      --num_points_eval $EPOINTS \
      --split_size 1000
  done
done
