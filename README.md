# CatFLW Facial Landmark Detection

This project investigates **48-point cat facial landmark detection** using three lightweight convolutional neural networks: ShuffleNetV2, EfficientNetB0, and MobileNetV3. It compares **coordinate regression** and **heatmap regression**, with additional ablation and repeated-seed experiments. The project was developed in Google Colab using PyTorch.

The accompanying report contains the experimental results and discussion. This repository provides the code and configurations used to produce them.

## Project files

| File | Purpose |
|---|---|
| `CatFLW.ipynb` | Colab notebook containing setup, training experiments, evaluation, and plots |
| `train.py` | Trains a model, evaluates the best validation checkpoint on the test split, and saves example predictions |
| `models.py` | CNN backbones and coordinate/heatmap models |
| `catflw_dataset.py` | Annotation loading, preprocessing, and dataset splitting |
| `eval_utils.py` | Evaluation metrics and visualizations |
| `compare_experiments.py` | Additional model-comparison utility |
| `download_catflw.py` | Kaggle dataset downloader |
| `inspect_annotations.py` | Dataset inspection utility |
| `requirements.txt` | Required Python packages |

## Setup and data preparation

1. Open `CatFLW.ipynb` in **Google Colab** and select a GPU runtime if available.
2. In the first setup cell, replace `YOUR_USERNAME` with the owner of this GitHub repository and check that the repository name is correct. The cell clones the project and installs its dependencies from `requirements.txt` (plus SciPy for the statistical analysis).
3. Run the dataset setup cells to download [CatFLW from Kaggle](https://www.kaggle.com/datasets/georgemartvel/catflw). Kaggle authentication may be required.
4. Run the remaining cells for the experiments you wish to reproduce.

The dataset is downloaded into the temporary Colab environment and **is not included in this repository**. Do not upload Kaggle credentials or API keys.

## Training and evaluation

The notebook contains the exact commands for each experiment. For example, the main MobileNetV3 coordinate-regression configuration uses:

```bash
python train.py \
  --approach coord \
  --backbone mobilenetv3_small_100 \
  --data_dir /content/catflw_data \
  --out_dir runs/coord_mobilenet \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4
```

`train.py` splits the data into training, validation, and test sets, trains the selected model, saves the checkpoint with the best validation NRMSE, and reports **test NRMSE** after training. It also saves sample prediction images. There is no separate evaluation command required for this workflow.

## Configurations used in the report

All main experiments use **30 epochs**, **learning rate `1e-4`**, and the default **seed `0`**, unless otherwise specified.

| Backbone | Coordinate batch size | Heatmap batch size |
|---|---:|---:|
| ShuffleNetV2 (`shufflenet_v2_x1_0`) | 16 | 16; additional run at 8 |
| EfficientNetB0 (`efficientnet_b0`) | 16 | 8 |
| MobileNetV3 (`mobilenetv3_small_100`) | 16 | 8 |

The MobileNetV3 coordinate-regression ablations use `--no-alignment`, `--edge-filter`, or `--mse-loss`. The repeated runs use seeds **1, 2, 3, and 110**, alongside the main seed-0 run. Each experiment's output directory and settings are recorded in the notebook.

## Notes

- Training outputs are saved under `runs/`; these files are temporary in Colab unless exported.
- The notebook also includes recorded experiment plots and statistical analysis. Full results and interpretation are in the report.
- Reproducing results exactly may depend on software versions, hardware, and randomness in training.
