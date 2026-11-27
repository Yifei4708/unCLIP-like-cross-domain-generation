# Reproducible MNIST–SVHN data preparation

Copy these files into the project repository, then run:

```bash
cd /Users/ysun/Documents/DL/unCLIP-like-cross-domain-generation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_data.py --seed 2026
python -m pip freeze > requirements-lock.txt
```

Downloads require internet and may take several minutes. SVHN requires scipy.
The first script run downloads the official MNIST/SVHN train/test sets using
torchvision (never SVHN extra). It refuses to overwrite an existing index folder.
If interrupted after index creation, retain or rename that incomplete directory
before rerunning; a complete run ends with manifest.json and checks: passed.

## Outputs

Under `data/indices/seed_2026/`:

| File | Meaning |
|---|---|
| `{domain}_splits.npz` | train/val/test original dataset indices |
| `{domain}_labels.npz` | Original train/test labels, for pairing and evaluation |
| `pairs_{split}.npy` | Two columns: original MNIST index, original SVHN index |
| `{domain}_prompts.npy` | Shape (10,10): training prompt indices, row = digit |
| `{domain}_evaluation.npy` | 1,000 uniformly selected official test indices |
| `{domain}_figures.npy` | Shape (10,10): fixed test examples, row = digit |
| `dataset_preview.png` | Fixed example per digit in each domain |
| `manifest.json` | Seeds, versions, counts, policies and artifact checksums |

Validation is a uniformly random 10% holdout, NOT stratified: MNIST 54,000/6,000;
SVHN 65,931/7,326. Official test sets remain 10,000 and 26,032 images.
Pair rows total 2,398,620 training, 266,520 validation and 720,640 test.
The script checks complete/disjoint train/val coverage, split membership,
matching labels, exactly 20 partners per source per direction, and distinct
partners within each group. Duplicates between the two direction blocks stay.

All indices address ORIGINAL datasets, not Subset-relative positions. For val,
load the official training dataset; for test load the official test dataset.
Both domains use ToTensor(): MNIST stays 1x28x28 and SVHN stays 3x32x32,
with values in [0,1]. Do not standardize decoder targets away from this range.
torchvision already converts the original SVHN zero label from 10 to 0.

Commit the scripts, requirements-lock.txt, manifest, and saved indices to Git;
raw downloads and the virtual environment are ignored. Share the SAME index
files with all teammates. Package versions are recorded because seed equality
alone does not guarantee reproducibility across different environments.

## Using the prepared data in PyTorch

```python
import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

indices = np.load('data/indices/seed_2026/mnist_splits.npz')
full_train = MNIST('data/raw', train=True, transform=ToTensor())
train_set = Subset(full_train, indices['train'].tolist())
val_set = Subset(full_train, indices['val'].tolist())
```

Labels may form pairs, train evaluation judges, or fit evaluation probes. Do
not use digit classification losses for encoders/decoders/priors. Use validation
for selection and test for final evaluation. Figure indices are selected before
training; never select figures based on how attractive generated outputs look.

For future training call prepare_data.seed_everything(seed); seed DataLoader
workers and supply a seeded torch.Generator when shuffling. Save model configs
and use the manifest's two training seeds for repeated decoder/prior runs.
