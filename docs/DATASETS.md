# Dataset Preparation

The repository does not vendor datasets. The training scripts expect local data under `datasets/`, and the preparation script can recreate the public datasets used by the experiments.

```powershell
python support\scripts\prepare_real_datasets.py --all-public
python support\scripts\prepare_real_datasets.py --status
```

Expected local layout after preparation:

```text
datasets/
  HWDB1/
    train.txt
    test.txt
  msrc2_seg/
    images/*.bmp
    gt/*_GT.bmp
  bsds500/
    trainval/*.jpg
    test/*.jpg
  jena_climate_2009_2016.csv
  shakespeare.txt
```

The script downloads or organizes MNIST, Fashion-MNIST, MSRC-V2, BSDS500, Jena Climate, Tiny Shakespeare, and a 10-class HWDB1-compatible subset.

Datasets are kept out of Git because several archives exceed GitHub's regular file-size limits and public datasets may have their own redistribution terms. Keeping the downloader and preparation logic in source control makes the experiment reproducible without turning the repository into a dataset mirror.
