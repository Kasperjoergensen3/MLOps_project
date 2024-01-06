# MLOps_project - Brain-tumor classification with transformers using MRI images

Alexander Samuel Bendix Gosden sxxyyzz
Asger Valbjørn Schødt sxxyyzz
Jonatan Hauge Steffensen s230368
Kasper Jørgensen sxxyyzz

### Project goal

The overall goal of this project is to apply the methods we have learned in class to create an agile and reproducable DL project.

### DL model

We will demonstrate the use of these methods by finetuning a pretrained image-transformer model to classify different types of brain tumors (or non brain-tumors) based on MRI images. 

### Framework 

For the project we will use a pretrained model from [Huggingface-Transformers](https://huggingface.co/docs/transformers/index) and finetune it for our classification task.

### Data

We use the dataset [MRI-Images-of-Brain-Tumor](https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor) which is a HuggingFace Dataset and therefore it is allready split into a training set with 3760 images, a validation set with 1070 images and a test set with 537 images.


### Project flowchart

!!Insert nice image of the operations-flowchart of our project!!

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
