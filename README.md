# ConRad
This repository contains the code to reproduce the results of our ConRad paper (update link).

Requirements:
* Download [LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)
* [PyTorch](https://pytorch.org)
* [PyTorch Lightning](https://pylidc.github.io)
* [pylidc](https://pylidc.github.io)
* [TorchIO](https://torchio.readthedocs.io)


The notebooks should be executed in the following order:
1. [create_dataset.ipynb](create_dataset.ipynb), extract nodule volumes and associated annotations (segmentations, biomarkers, malignancy)
2. [finetune_models.ipynb](finetune_models.ipynb), finetune ResNet end-to-end CNN model and biomarker regression model
3. [extract_radiomics.ipynb](extract_radiomics.ipynb), extract radiomics features using nodule volumes and segmentations
4. [classifiers.ipynb](classifiers.ipynb), train classifiers on various feature sets

The repository contains all the necessary code for finetuning the CNN and biomarker regression models. To allow for a quick reproduction of our results we also make [pre-trained weights](https://drive.google.com/file/d/1x2ctl62Duo2w8rMV7b-hQrDxt6rGchic/view?usp=share_link) available. They should be placed in a folder called "weights".

The model can be interacted with [here](conrad.streamlit.app).
