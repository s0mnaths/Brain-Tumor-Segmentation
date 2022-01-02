import cv2
import numpy as np
import albumentations as A

def preprocess(img_filepath):
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