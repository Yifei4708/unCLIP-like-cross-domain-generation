# unCLIP-like Cross-Domain Generation

ECE 685D final project: learn a shared representation of handwritten digits
(MNIST) and photographed house numbers (SVHN), then generate images across domains
while preserving digit identity.

## Current Progress

- Completed reproducible data preparation using seed **2026**.
- Saved training, validation, and test splits, same-digit pairs, prompts,
  evaluation indices, figure indices, and checksums.
- Configured Google Colab GPU training with checkpoint and resume support.
- The saved training notebook reports **30,000 completed optimizer updates**
  and confirms that `frozen_encoders.pt` exists.
- Next: evaluate the embeddings, then train decoders and cross-domain priors.

## 1. Data Preparation

### Datasets and Splits

Datasets are downloaded using `torchvision`.

- **MNIST:** grayscale handwritten digits, 28 × 28 pixels.
- **SVHN:** RGB house-number images, 32 × 32 pixels.
- SVHN's **extra set is excluded**.
- A uniformly random 10% of each official training set is held out for
  validation, using a fixed seed.
- Validation size is rounded up with `ceil(0.1 * N)`.
- Splits are **not stratified**.
- Official test sets remain separate.

| Dataset | Original Training | Project Training | Validation | Test |
|---|---:|---:|---:|---:|
| MNIST | 60,000 | 54,000 | 6,000 | 10,000 |
| SVHN | 73,257 | 65,931 | 7,326 | 26,032 |

Training data is used to fit models. Validation data is used for permitted
hyperparameter selection. Test data is reserved for final evaluation.

The baseline contrastive encoders follow the fixed 30,000-update recipe,
rather than validation-based early stopping.

### Image Preprocessing

| Setting | MNIST | SVHN |
|---|---|---|
| Input shape | `1 × 28 × 28` | `3 × 32 × 32` |
| Pixel range | `[0, 1]` | `[0, 1]` |
| Color | Grayscale | RGB |
| Resizing | None | None |

Torchvision converts SVHN's original label `10` into digit label `0`.

### Same-Digit Pair Construction

MNIST and SVHN do not contain naturally corresponding images.

Within each split:

1. Pair every MNIST image with **20 distinct random SVHN images of the same digit**.
2. Pair every SVHN image with **20 distinct random MNIST images of the same digit**.
3. Concatenate both lists, keeping duplicate pairs between directions.

| Split | MNIST Images | SVHN Images | Pair Rows |
|---|---:|---:|---:|
| Training | 54,000 | 65,931 | 2,398,620 |
| Validation | 6,000 | 7,326 | 266,520 |
| Test | 10,000 | 26,032 | 720,640 |

Each pair row has the format:

```python
[mnist_index, svhn_index]
```

Indices refer to the **original torchvision datasets**, not positions within a
`Subset`.

- Training and validation indices address the official training datasets.
- Test indices address the official test datasets.
- No pairs cross split boundaries.

### Saved Files

Prepared files are stored in:

```text
data/indices/seed_2026/
```

| File | Contents |
|---|---|
| `mnist_splits.npz` | MNIST indices under `train`, `val`, and `test` |
| `svhn_splits.npz` | SVHN indices under `train`, `val`, and `test` |
| `{domain}_labels.npz` | Original labels under `train` and `test` |
| `pairs_train.npy` | Training pairs |
| `pairs_val.npy` | Validation pairs |
| `pairs_test.npy` | Test pairs |
| `{domain}_prompts.npy` | Shape `(10, 10)`: 10 training prompts per digit |
| `{domain}_evaluation.npy` | 1,000 fixed test-image indices per domain |
| `{domain}_figures.npy` | Shape `(10, 10)`: 10 fixed test images per digit |
| `dataset_preview.png` | Fixed examples from both domains |
| `manifest.json` | Seeds, counts, package versions, policies, and SHA-256 checksums |

For prompt and figure arrays, the row number corresponds to the digit.

Figure examples are selected before training to avoid choosing only attractive
generated results.

### Reproducibility and Checks

The master seed is **2026**. Separate derived random streams are saved for:

- MNIST and SVHN splits.
- Training, validation, and test pairing.
- Prompt selection.
- Evaluation selection.
- Figure selection.

Preparation checks include:

- Complete coverage and no overlap between training and validation.
- Correct split membership for pairs.
- Matching digit labels in every pair.
- Exactly 20 distinct partners per source image in each pairing direction.

The completed local preparation contains `manifest.json`, and all **14 saved
artifact checksums passed verification**.

Reuse these saved index files across teammates and machines. Matching seeds
alone do not guarantee identical results across different package versions.

### Label Restrictions

Digit labels may be used only to:

- Construct same-digit pairs.
- Train evaluation judges.
- Fit evaluation probes.

Do not train encoders, decoders, or priors with digit-classification losses or
gradients from judges or probes.

### Running Data Preparation

Example setup using Conda:

```bash
conda create -n unclip python=3.12 -y
conda activate unclip

python -m pip install -r requirements.txt
python prepare_data.py --seed 2026
python -m pip freeze > requirements-lock.txt
```

Raw datasets are downloaded into `data/raw/`.

The script refuses to overwrite an existing index folder. If preparation is
interrupted, preserve the incomplete folder under another name before retrying.

If preparation has already completed, reuse the saved indices instead of
generating another split.

To package the indices for Colab:

```bash
zip -r seed_2026.zip data/indices/seed_2026
```

This ZIP contains indices and metadata, not the actual MNIST/SVHN images.

## 2. Contrastive Encoder Training

### Goal

Train two encoders to map MNIST and SVHN images into a shared embedding space.

A handwritten digit and a house-number image of the same digit should receive
similar embeddings, despite differences in appearance.

### Required Architectures

#### MNIST Encoder

```text
Input: 1 × 28 × 28
Flatten: 784
Linear: 784 → 512
ReLU
Linear: 512 → 256
ReLU
Linear: 256 → 64
L2 normalization
```

#### SVHN Encoder

Four convolutional blocks, each using:

```text
Conv2d → BatchNorm → ReLU
```

| Block | Channels | Spatial Size |
|---|---|---|
| 1 | 3 → 32 | 32 → 16 |
| 2 | 32 → 64 | 16 → 8 |
| 3 | 64 → 128 | 8 → 4 |
| 4 | 128 → 256 | 4 → 2 |

Every convolution uses:

- Kernel size: `4 × 4`
- Stride: `2`
- Padding: `1`

The final feature map is flattened and projected:

```text
256 × 2 × 2 → 1024 → Linear → 64 → L2 normalization
```

### Required Training Parameters

| Parameter | Required Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `0.001` |
| Batch size | `256` pairs |
| Embedding dimension | `64` |
| Output normalization | L2 normalization |
| Loss | Symmetric InfoNCE |
| Initial temperature | `0.07`, learnable |
| Positive examples | Matching pairs on the batch diagonal only |
| Training length | `30,000` successful optimizer updates |
| After training | Freeze both encoders |

**30,000 updates means 30,000 batch-level optimizer steps, not 30,000 epochs.**

Other images of the same digit within a batch remain negatives for this required
baseline. Using all same-digit examples as positives is a later ablation.

### Implementation Defaults

These are implementation choices rather than additional course requirements.

| Setting | Default |
|---|---|
| Data seed | `2026` |
| Encoder training seed | `2026` |
| Precision | Float32; mixed precision disabled |
| Checkpoint interval | Every `500` updates and at the final update |
| Console logging | Every `100` updates and the first update |
| Batch order | Shuffled epochs |
| Incomplete final batch | Dropped |
| Learning-rate scheduler | None |
| Adam weight decay | `0` |
| Adam betas | `(0.9, 0.999)` |
| Adam epsilon | `1e-8` |

### Contrastive Loss

For a batch of paired embeddings:

```python
logits = (e_mnist @ e_svhn.T) / temperature
targets = torch.arange(batch_size, device=device)

loss = (
    cross_entropy(logits, targets)
    + cross_entropy(logits.T, targets)
) / 2
```

The similarity matrix has shape `256 × 256`.

- Rows classify the matching SVHN image for each MNIST image.
- Columns classify the matching MNIST image for each SVHN image.
- Targets represent pair positions, not digit labels.

The loss need not approach zero because several images within a batch may share
the same digit.

Only training pairs are used for encoder optimization.

### Colab Workflow

1. Open `unclip_colab_training.ipynb`.
2. Select a GPU runtime and confirm CUDA availability.
3. Mount Google Drive.
4. Upload and extract `unclip-colab-code.zip`.
5. Upload or restore the completed `seed_2026` index files.
6. Verify manifest checksums.
7. Download raw images to Colab's local storage.
8. Run the separate 10-update smoke test.
9. Run the full 30,000-update training.

The code bundle contains:

```text
prepare_data.py
train_embedding.py
requirements.txt
```

Keep Colab's provided CUDA-enabled PyTorch installation. Record its environment
separately in `requirements-gpu-lock.txt`; do not install the Mac environment
lock file on Colab.

### Training Commands

After extracting the code bundle, run from its working folder.

Smoke test:

```bash
python train_embedding.py \
  --output smoke_run \
  --steps 10 \
  --save-every 10 \
  --resume
```

Full baseline:

```bash
python -u train_embedding.py \
  --data-root data \
  --data-seed 2026 \
  --seed 2026 \
  --output /content/drive/MyDrive/unclip_project/runs/embedding_seed_2026 \
  --steps 30000 \
  --save-every 500 \
  --resume
```

The smoke test uses a separate output folder and does not count toward the
reported baseline training.

### Checkpoints and Outputs

| File | Purpose |
|---|---|
| `latest.pt` | Model, optimizer, completed update, batch state, RNG states, and loss history |
| `frozen_encoders.pt` | Final encoder weights and configuration, written at 30,000 updates |
| `config.json` | Seeds, training settings, data fingerprint, and runtime information |
| `loss.json` | Per-update loss and temperature |
| `training_loss.png` | Training-loss curve |
| `requirements-gpu-lock.txt` | Packages installed in the Colab environment |

Checkpoints preserve:

- Model and optimizer states.
- Completed optimizer-update count.
- Batch permutation and current position.
- Python and NumPy random states.
- PyTorch CPU and CUDA random states.

After a disconnect, restore the code and data indices, then rerun the full
training command with `--resume`.

Keep the same code, data, and environment when resuming. Different hardware or
package versions can still affect numerical reproducibility.

Store checkpoints in Google Drive because Colab's local runtime storage is
temporary. Load only trusted project checkpoints.

### Recorded Training Status

The saved notebook reports:

```text
Completed updates: 30000
Final encoder artifact: True
```

This confirms completion reported by the notebook. Embedding quality still
needs to be measured using the required evaluation metrics.

## 3. Next Steps

### Embedding Evaluation

- Full-test cross-domain precision@10 in both directions.
- Prompt-based digit classification accuracy.
- Digit logistic-regression probes.
- Style ridge-regression probes.

### Decoder and Generation Stages

- Train deterministic decoders, D1, for both domains.
- Train conditional VAEs, D2, for both domains.
- Train deterministic and stochastic priors, P1 and P2, in both directions.
- Compare against direct embeddings, P0, and the oracle.
- Produce reconstruction, variation, swap, generation, interpolation,
  style-transfer, and ablation results.

The manifest reserves seeds **2026 and 2027** for repeated later-stage runs.

The course requires at least two seeds for D2 in each domain and priors in each
direction. The baseline contrastive embedding is trained once.

## 4. GitHub Storage

Commit:

- Source code and notebooks.
- Dependency descriptions.
- Saved split, pair, prompt, evaluation, and figure indices.
- Manifests and relevant result summaries.

Do not commit:

- Raw MNIST/SVHN datasets.
- Virtual environments.
- Model checkpoints.

The `seed_2026.zip` file is optional because it duplicates the index folder.

Suggested `.gitignore`:

```gitignore
.venv/
.venv*/
__pycache__/
.DS_Store
data/raw/
*.pt
*.pth
seed_2026.zip
```
