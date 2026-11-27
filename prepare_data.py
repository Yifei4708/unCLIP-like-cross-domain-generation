"""Prepare the course's MNIST/SVHN protocol. Run from the repository root."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import random
from pathlib import Path

import numpy as np


def split_indices(n, rng):
    order = rng.permutation(n)
    nval = math.ceil(0.1 * n)
    return {"train": np.sort(order[nval:]), "val": np.sort(order[:nval])}


def make_pairs(mi, si, ml, sl, rng, k=20):
    """Columns always index (MNIST, SVHN) in the original torchvision datasets.

    First block: k targets per MNIST image. Second: k targets per SVHN
    image. Keep duplicate rows across blocks, as required by the assignment.
    """
    blocks = []
    for src, target, src_labels, target_labels, reverse in (
        (mi, si, ml, sl, False), (si, mi, sl, ml, True)
    ):
        pools = [target[target_labels[target] == d] for d in range(10)]
        if min(map(len, pools)) < k:
            raise ValueError("Each digit needs at least 20 target images per split")
        partners = np.concatenate([
            rng.choice(pools[int(src_labels[i])], k, replace=False) for i in src
        ])
        sources = np.repeat(src, k)
        blocks.append(np.column_stack((partners, sources) if reverse else
                                      (sources, partners)))
    return np.concatenate(blocks).astype(np.int64)


def validate_pairs(pairs, mi, si, ml, sl, k=20):
    assert len(pairs) == k * (len(mi) + len(si))
    assert np.isin(pairs[:, 0], mi).all()
    assert np.isin(pairs[:, 1], si).all()
    assert np.array_equal(ml[pairs[:, 0]], sl[pairs[:, 1]])
    for block, source, reverse in (
        (pairs[:k * len(mi)], mi, False),
        (pairs[k * len(mi):], si, True),
    ):
        groups = block.reshape(-1, k, 2)
        assert (groups[:, :, int(reverse)] == source[:, None]).all()
        partners = np.sort(groups[:, :, int(not reverse)], axis=1)
        assert (np.diff(partners, axis=1) > 0).all()


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be between 0 and 2**32 - 1")
    out = args.root / "indices" / f"seed_{args.seed}"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite saved indices: {out}")
    import torch
    from torchvision.datasets import MNIST, SVHN
    from torchvision.transforms import ToTensor
    seed_everything(args.seed)
    datasets = {
        "mnist": {split: MNIST(str(args.root / "raw"), train=split == "train",
                              download=True, transform=ToTensor())
                  for split in ("train", "test")},
        "svhn": {split: SVHN(str(args.root / "raw" / "SVHN"), split=split,
                            download=True, transform=ToTensor())
                 for split in ("train", "test")},
    }
    labels = {domain: {split: np.asarray(ds.targets if domain == "mnist"
                                        else ds.labels, dtype=np.int64)
                       for split, ds in sets.items()}
              for domain, sets in datasets.items()}
    streams = ("split_mnist", "split_svhn", "pairs_train", "pairs_val",
               "pairs_test", "prompts_mnist", "prompts_svhn",
               "eval_mnist", "eval_svhn", "figures_mnist", "figures_svhn")
    children = np.random.SeedSequence(args.seed).spawn(len(streams))
    seeds = {name: int(child.generate_state(1)[0])
             for name, child in zip(streams, children)}
    rng = {name: np.random.default_rng(seed) for name, seed in seeds.items()}
    out.mkdir(parents=True)
    splits, summary = {}, {}
    for domain in datasets:
        y = labels[domain]
        assert set(np.unique(y["train"])) == set(range(10))
        splits[domain] = split_indices(len(y["train"]), rng[f"split_{domain}"])
        splits[domain]["test"] = np.arange(len(y["test"]))
        train, val = splits[domain]["train"], splits[domain]["val"]
        assert not np.intersect1d(train, val).size
        assert np.array_equal(np.sort(np.concatenate((train, val))),
                              np.arange(len(y["train"])))
        np.savez_compressed(out / f"{domain}_splits.npz", **splits[domain])
        np.savez_compressed(out / f"{domain}_labels.npz", **y)
        prompts = np.stack([rng[f"prompts_{domain}"].choice(
            train[y["train"][train] == digit], 10, replace=False)
                            for digit in range(10)])
        np.save(out / f"{domain}_prompts.npy", prompts)
        evaluation = rng[f"eval_{domain}"].choice(splits[domain]["test"],
                                                 1000, replace=False)
        figures = np.stack([rng[f"figures_{domain}"].choice(
            np.flatnonzero(y["test"] == digit), 10, replace=False)
                            for digit in range(10)])
        np.save(out / f"{domain}_evaluation.npy", evaluation)
        np.save(out / f"{domain}_figures.npy", figures)
        summary[domain] = {
            split: {"size": len(ids), "digit_counts": np.bincount(
                y["test" if split == "test" else "train"][ids], minlength=10).tolist()}
            for split, ids in splits[domain].items()}
    pair_counts = {}
    for split in ("train", "val", "test"):
        base = "test" if split == "test" else "train"
        mi, si = splits["mnist"][split], splits["svhn"][split]
        ml, sl = labels["mnist"][base], labels["svhn"][base]
        pairs = make_pairs(mi, si, ml, sl, rng[f"pairs_{split}"])
        validate_pairs(pairs, mi, si, ml, sl)
        np.save(out / f"pairs_{split}.npy", pairs)
        pair_counts[split] = len(pairs)
    # Preselected figure: one pair per digit; no visual cherry-picking.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for row, domain in enumerate(("mnist", "svhn")):
        ids = np.load(out / f"{domain}_figures.npy")
        for digit in range(10):
            image, label = datasets[domain]["test"][int(ids[digit, 0])]
            assert label == digit
            axes[row, digit].imshow(image.permute(1, 2, 0).squeeze(),
                                    cmap="gray" if domain == "mnist" else None,
                                    vmin=0, vmax=1)
            axes[row, digit].set_title(f"{domain}: {digit}")
            axes[row, digit].axis("off")
    fig.tight_layout()
    fig.savefig(out / "dataset_preview.png", dpi=150)
    plt.close(fig)
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(out.iterdir())}
    manifest = {
        "seed": args.seed, "random_stream_seeds": seeds,
        "future_training_seeds": [args.seed, args.seed + 1],
        "split_policy": "uniform permutation, ceil(10% train) validation; not stratified",
        "index_space": "original torchvision train for train/val; official test for test",
        "pair_columns": ["mnist_index", "svhn_index"],
        "pair_policy": "20 distinct partners per source, union of both directions; duplicates kept",
        "pixel_range": [0, 1], "svhn_extra": False,
        "counts": summary, "pair_counts": pair_counts,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "matplotlib")},
        "sha256": checksums,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(out), "counts": summary,
                      "pairs": pair_counts, "checks": "passed"}, indent=2))


if __name__ == "__main__":
    main()
