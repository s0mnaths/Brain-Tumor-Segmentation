import onnxruntime

def predict(input_img):
    model_onnx = './checkpoints/brain-mri-unet.onnx'

    session = onnxruntime.InferenceSession(model_onnx, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: input_img})
    
    return result