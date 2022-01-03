import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess
from predict import predict

def inference(filepath):
    input_batch = preprocess(filepath)
    result = predict(input_batch)
    pred_mask = np.array(result).astype(np.float32)
    pred_mask = pred_mask * 255
    pred_mask = pred_mask[0, 0, 0, :, :].astype(np.uint8)
    plt.imshow(pred_mask)
    plt.title("Predicted Tumor Mask")

    return plt


title = "Brain MRI Tumor Detection - Semantic Segmentation using PyTorch"
description = "Segmentation of tumor areas from Brain MRI images"
article = "<p style='text-align: center'><a href='https://www.kaggle.com/s0mnaths/brain-mri-unet-pytorch/' target='_blank'>Kaggle Notebook: Brain MRI-UNET-PyTorch</a> | <a href='https://github.com/s0mnaths/Brain-Tumor-Segmentation' target='_blank'>Github Repo</a></p>"
examples = [['test-samples/TCGA_CS_4942_19970222_10.png'], 
            ['test-samples/TCGA_CS_4942_19970222_11.png'], 
            ['test-samples/TCGA_CS_4942_19970222_12.png'], 
            ['test-samples/TCGA_CS_4941_19960909_15.png']]  

gr.Interface(inference, inputs=gr.inputs.Image(type="filepath"), outputs=gr.outputs.Image('plot'), title=title,
            description=description,
            article=article,
            examples=examples,
            server_name="0.0.0.0").launch(debug=False, enable_queue=True)
