import gradio as gr
import onnxruntime

import cv2
# from PIL import Image
import numpy as np
# import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt

#PREPROCESS IMAGE
def load_sample(img_filepath):
    image = cv2.imread(img_filepath)
    image = (np.array(image).astype(np.float32))/255.
    
    test_transform = A.Compose([
                    A.Resize(width=128, height=128, p=1.0)
                    ])
    
    aug = test_transform(image=image)
    image = aug['image']
            
    image = image.transpose((2,0,1))
    
    #image normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])

    for i in range(image.shape[0]):
        image[i, :, :] = (image[i, :, :] - mean_vec[i]) / (std_vec[i])

    image = np.stack([image]*1)

    return image



#LOAD MODEL & PREDICT
def predict(input_img):
    model_onnx = './checkpoints/brain-mri-unet.onnx'

    session = onnxruntime.InferenceSession(model_onnx, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: input_img})
    
    return result



def inference(filepath):

    input_batch = load_sample(filepath)
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
examples = [['test-samples/TCGA_CS_4942_19970222_10.png'], ['test-samples/TCGA_CS_4942_19970222_11.png'], ['test-samples/TCGA_CS_4942_19970222_12.png'], ['test-samples/TCGA_CS_4941_19960909_15.png']]  

gr.Interface(inference, inputs=gr.inputs.Image(type="filepath"), outputs=gr.outputs.Image('plot'), title=title,
    description=description,
    # article=article,
    examples=examples).launch(debug=False, enable_queue=True)