
# Brain-Tumor-Segmentation
[![Heroku](https://heroku-badge.herokuapp.com/?app=brain-mri-segmentation)](https://brain-mri-segmentation.herokuapp.com/)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/s0mnaths/Brain-Tumor-Segmentation/blob/master/notebooks/brain_tumor_segmentation.ipynb)

### Semantic Segmentation of tumor from Brain MRI images using PyTorch.















The model architecture used is [UNET](https://arxiv.org/abs/1505.04597v1) which is trained using PyTorch, and then converted to ONNX format for deployment using Heroku.
Evaluation metric used is `DICE coefficient`, with loss as `(1-DICE) + BCELoss`.
## Dataset 📂
Dataset used for training is from Kaggle [LGG Segmentation Dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation) which which contains over 3900 samples obtained from The Cancer Imaging Archive. 
## Notebook 📒

View the notebook here: [brain_tumor_segmentation.ipynb](https://nbviewer.org/github/s0mnaths/Brain-Tumor-Segmentation/blob/master/notebooks/brain_tumor_segmentation.ipynb)
## Deployment 🚀

The model has been been converted to ONNX format and deployed using Gradio & hosted on Heroku: [Brain MRI Tumor Detection](https://brain-mri-segmentation.herokuapp.com/)


## Predictions 🔍
Predictions on unseen test data:


![predgif](https://github.com/s0mnaths/Brain-Tumor-Segmentation/blob/master/demo/predictions.gif)


![samplepred](https://github.com/s0mnaths/Brain-Tumor-Segmentation/blob/master/demo/sample-pred.png)
## Dataset 📂
Dataset used for training is from Kaggle [LGG Segmentation Dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation) which which contains over 3900 samples obtained from The Cancer Imaging Archive. 