# Breast Cancer Classification

Classification of images as benign or malignant using ResNet50, with GradCAM visualization and comparison between pretrained and from-scratch models.

## Notebook

The training notebook is available on Kaggle:
https://www.kaggle.com/code/danamirescu/breast-cancer-classification-v1

## Dataset

The dataset used is BreaKHis 400X:
https://www.kaggle.com/datasets/forderation/breakhis-400x

After downloading, the folder structure should look like:
BreaKHis 400X/

```
├── train/
│   ├── benign/
│   └── malignant/
└── test/
│   ├── benign/
│   └── malignant/
```

## Project Structure
```
├── app.py                            # Streamlit UI
├── model_utils.py                    # shared model functions
├── best_pretrained_v4.pth            # not included - download from Training section
└── best_scratch_v4.pth               # not included - download from Training section
```

## Training

**Option 1: Train from scratch (Kaggle)**
Open the notebook in Kaggle and run all cells.
GPU is recommended — enable it in Settings → Accelerator → GPU T4.

**Option 2: Load pretrained weights (Kaggle)**
Open the notebook in Kaggle, attach the following datasets, set `KAGGLE = True`
at the top of the notebook, and skip to the evaluation cells.
- Models: https://www.kaggle.com/models/danamirescu/version4
- Loss history: https://www.kaggle.com/models/danamirescu/loss-history

**Option 3: Load pretrained weights (local)**
Download the files from the links above, place them in the root folder,
set `KAGGLE = False` at the top of the notebook, and skip to the evaluation cells.

## Running the App

Install dependencies:
```bash
pip install torch torchvision streamlit grad-cam pillow scikit-learn matplotlib
```

Run:
```bash
streamlit run app.py
```

## References

- GradCAM: https://arxiv.org/abs/1610.02391
- Dataset: https://www.kaggle.com/datasets/forderation/breakhis-400x