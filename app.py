from fastapi import FastAPI, UploadFile, File
import torch
from monai.networks.nets import BasicUNetPlusPlus, DenseNet
from monai.data import MetaTensor
import numpy as np
import tifffile as tf
import io
from skimage.io import imread
app = FastAPI()

#Segmentation
segmentation_model = BasicUNetPlusPlus(
    spatial_dims=2,
    in_channels=8,
    out_channels=1,
    features=(16, 32, 64, 128, 64, 32),
)
segmentation_state_dict = './models/segmentation_model/segmentation_model_monai_unetpp_2023-07-01_23-25-36.pth'
segmentation_model.load_state_dict(torch.load(segmentation_state_dict))

#Classification
classification_model = model = DenseNet(
    spatial_dims=2, 
    in_channels=1, 
    out_channels=4, 
    init_features=64, 
    growth_rate=32, 
    block_config=(6, 12, 24, 16), 
)
classification_state_dict = './models/classification_model/classification_model_monai_densenet_2023-07-02_03-47-04.pth'
classification_model.load_state_dict(torch.load(classification_state_dict))

@app.post("/segmentation")
async def segmentation(image: UploadFile):
    contents = await image.read()
    with io.BytesIO(contents) as f:
        image_array = tf.imread(f)
    image = MetaTensor(image_array)
    output = segmentation_model(image.unsqueeze(0))[0]
    output = np.array(output.detach().numpy()).tolist()
    return {"result": output}

@app.post("/classification")
async def classification(image: UploadFile):
    contents = await image.read()
    with io.BytesIO(contents) as f:
        image_array = imread(f)
    image = MetaTensor(image_array)/255
    output = classification_model(image.unsqueeze(0).unsqueeze(0))
    print(output)
    output = np.array(output.detach().numpy()).tolist()
    return {"result": output}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)