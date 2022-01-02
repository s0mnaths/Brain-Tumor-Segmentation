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






#GRADIO APP
title = "Brain MRI Tumor Detection - Sematic Segmentation using PyTorch"
description = "Detection of tumor areas of Brain MRI images from the Kaggle dataset"
# article = "<p style='text-align: center'><a href='https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/' target='_blank'>Detectron2: A PyTorch-based modular object detection library</a> | <a href='https://github.com/facebookresearch/detectron2' target='_blank'>Github Repo</a></p>"
examples = [['test-samples/TCGA_CS_4942_19970222_10.png'], 
            ['test-samples/TCGA_CS_4942_19970222_11.png'], 
            ['test-samples/TCGA_CS_4942_19970222_12.png'], 
            ['test-samples/TCGA_CS_4941_19960909_15.png']]  

gr.Interface(inference, inputs=gr.inputs.Image(type="filepath"), outputs=gr.outputs.Image('plot'), title=title,
    description=description,
    # article=article,
    examples=examples).launch(debug=False, enable_queue=True)