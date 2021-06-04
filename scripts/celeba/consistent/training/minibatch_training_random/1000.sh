GPU=0
for RUN in 0
do
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python training/cnp.py \
    --run $RUN \
    --dataset celeba \
    --num_points 1000 \
    --size 64 64  \
    --epochs 200 \
    --lr 1e-3 \
    --wd 5e-3 \
    --aggregator consistent \
    --ln True \
    --hidden_dim 64 \
    --K 1 \
    --h 128 \
    --d 64  \
    --d_hat 64 \
    --_slots Random
done
