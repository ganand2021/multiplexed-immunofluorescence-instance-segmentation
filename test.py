import torch
from monai.networks.nets import BasicUNetPlusPlus, DenseNet
from monai.data import MetaTensor
from skimage.io import imread

#Segmentation
segmentation_model = BasicUNetPlusPlus(
    spatial_dims=2,
    in_channels=8,
    out_channels=1,
    features=(16, 32, 64, 128, 64, 32),
)
segmentation_state_dict = './models/segmentation_model/segmentation_model_monai_unetpp_2023-07-01_23-25-36.pth'
segmentation_model.load_state_dict(torch.load(segmentation_state_dict))

segmentation_sample_path = './data/segmentation_dataset/image/p01-9999(41323.14003)500,100-crop_cell_tif.tif'
sample_image = MetaTensor(imread(segmentation_sample_path))
try:
    output = segmentation_model(sample_image.unsqueeze(0))[0]
    print("Segmentation Model: Test PASSED ")
except:
    print("Segmentation Model: Test Failed, Implementation Error")


#Classification
classification_model = model = DenseNet(
    spatial_dims=2, 
    in_channels=1, 
    out_channels=4, 
    init_features=64, 
    growth_rate=32, 
    block_config=(6, 12, 24, 16), 
)

classification_state_dict = './models/classification_model/classification_model_monai_densenet_2023-07-01_19-06-55.pth'
classification_model.load_state_dict(torch.load(classification_state_dict))

classification_sample_path = './data/classification_dataset/test/Melanoma/1_crop_1.png'
sample_image = MetaTensor(imread(classification_sample_path))/255

try:
    output = classification_model(sample_image.unsqueeze(0).unsqueeze(0))
    print("Classification Model: Test PASSED ")
except:
    print("Classification Model: Test Failed, Implementation Error")
    
    
combined_path = './data/segmentation_dataset/image/p01-9999(41323.14003)500,100-crop_cell_tif.tif'
sample_image = MetaTensor(imread(combined_path))

try:
    output = segmentation_model(sample_image.unsqueeze(0))[0]
    output = classification_model(output)
    print("Combined Model: Test PASSED ")
except:
    print("Combined Model: Test Failed, Implementation Error")