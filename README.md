# Mini-Batch Consistent Slot Set Encoder forScalable Set Encoding

This repository contains the accompanying code for SSE. 

## Dependencies
```
torch==1.8.0
torchvision==0.9.0
```

## Empirical Verification of Proposition 1 \& 2
To empirically verify Propositions 1 \& 2, run the following:
```train
python models/slotsetencoder.py
```
This will check three things: permutation equivariance, permutation invariance and Mini-Batch Consistency.
The test code verifies these for the Slot Set Encoder with both random and deterministic slot initialization.

## Training
Run the following files to reproduce the results on CelebA image reconstruction.

```DeepSets
bash scripts/celeba/deepsets/training/minibatch_training/_setsize_sh
```
```SetTransformer
bash scripts/celeba/settransformer/training/minibatch_training/_setsize_.sh
```
```SlotSetEncoder
bash scripts/celeba/consistent/training/minibatch_training_xxx/_setsize_.sh
```
\_setsize\_ is one of 100, 200, 500 and 1000. For the Slot Set Encoder model, xxx is either learned or random.

## Mini-Batch Testing for MBC models.
After training, you can perform Mini-Batch testing with the model trained on 1000 elements by running:
```DeepSets
bash scripts/celeba/deepsets/testing/generalization/1000.sh
bash scripts/celeba/deepsets/testing/generalization/1000_small.sh
```

```SlotSetEncoder
bash scripts/celeba/consistent/testing/generalization_xxx/1000.sh
bash scripts/celeba/consistent/testing/generalization_xxx/1000_small.sh
```
