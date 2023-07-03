from fastapi import FastAPI, UploadFile, File
import torch
import monai
from monai.networks.nets import BasicUNetPlusPlus, DenseNet
from monai.data import MetaTensor
from monai.transforms import AsDiscrete, Activations
monai.utils.set_determinism(17)
import numpy as np
import tifffile as tf
import io
from skimage.io import imread
app = FastAPI()

def class_decoding(index):
    index_to_label = {
        0 : 'PDAC',
        1 : 'Melanoma',
        2 : 'CTCL',
        3 : 'Basal Cell'
    }

    return index_to_label[torch.argmax(index).item()]

def scale_metatensor(metatensor):
    min_value = np.min(metatensor)
    max_value = np.max(metatensor)
    range_value = max_value - min_value
    
    shifted_metatensor = metatensor - min_value
    normalized_metatensor = shifted_metatensor / range_value
    scaled_metatensor = normalized_metatensor * 255
    rounded_metatensor = np.round(scaled_metatensor)
    
    rounded_metatensor[rounded_metatensor >= 128] = 1.0
    rounded_metatensor[rounded_metatensor < 128] = 0.0
    
    return MetaTensor(rounded_metatensor)

@app.post("/segmentation")
async def segmentation(image: UploadFile):
    contents = await image.read()
    with io.BytesIO(contents) as f:
        image_array = tf.imread(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    #Segmentation
    segmentation_model = BasicUNetPlusPlus(
        spatial_dims=2,
        in_channels=8,
        out_channels=1,
        features=(16, 32, 64, 128, 64, 32),
    )
    segmentation_state_dict = './models/segmentation_model/segmentation_model_monai_unetpp_2023-07-01_23-25-36.pth'
    segmentation_model.load_state_dict(torch.load(segmentation_state_dict, map_location=device))
    segmentation_model.to(device)

    image = MetaTensor(image_array)
    image = image.to(device)
    with torch.no_grad():
        output = segmentation_model(image.unsqueeze(0))[0]
    output = np.array(output.detach().cpu().numpy()).tolist()
    return {"result": output}

@app.post("/classification")
async def classification(image: UploadFile):
    contents = await image.read()
    with io.BytesIO(contents) as f:
        image_array = imread(f)
        
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Classification
    classification_model = DenseNet(
        spatial_dims=2, 
        in_channels=1, 
        out_channels=4, 
        init_features=64, 
        growth_rate=32, 
        block_config=(6, 12, 24, 16), 
    )
    classification_state_dict = './models/classification_model/classification_model_monai_densenet_2023-07-02_03-54-13.pth'
    classification_model.load_state_dict(torch.load(classification_state_dict, map_location=device))
    classification_model.to(device)

    image = MetaTensor(image_array)/255
    image = image.to(device)
    threshold_fn = AsDiscrete(threshold=0.5)
    with torch.no_grad():
        output = classification_model(image.unsqueeze(0).unsqueeze(0))
    output = output.clone().detach()
    output = class_decoding(output)
    return {"result": output}

@app.post("/combined")
async def combined(image: UploadFile):
    contents = await image.read()
    with io.BytesIO(contents) as f:
        image_array = tf.imread(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    #Segmentation
    segmentation_model = BasicUNetPlusPlus(
        spatial_dims=2,
        in_channels=8,
        out_channels=1,
        features=(16, 32, 64, 128, 64, 32),
    )
    segmentation_state_dict = './models/segmentation_model/segmentation_model_monai_unetpp_2023-07-01_23-25-36.pth'
    segmentation_model.load_state_dict(torch.load(segmentation_state_dict, map_location=device))
    segmentation_model.to(device)

    image = MetaTensor(image_array)
    image = image.to(device)
    with torch.no_grad():
        output = segmentation_model(image.unsqueeze(0))[0]
    output = (scale_metatensor(output))
    
    #Classification
    classification_model = DenseNet(
        spatial_dims=2, 
        in_channels=1, 
        out_channels=4, 
        init_features=64, 
        growth_rate=32, 
        block_config=(6, 12, 24, 16), 
    )
    classification_state_dict = './models/classification_model/classification_model_monai_densenet_2023-07-02_03-54-13.pth'
    classification_model.load_state_dict(torch.load(classification_state_dict, map_location=device))
    classification_model.to(device)

    image = output.to(device)
    threshold_fn = AsDiscrete(threshold=0.5)
    with torch.no_grad():
        output = classification_model(image)
    output = output.clone().detach()
    output = class_decoding(output)
    return {"result": output}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)