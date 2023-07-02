import requests
import matplotlib.pyplot as plt
import numpy as np
import torch

URL_ENDPOINT = 'http://127.0.0.1:8000'
segmentation_sample_path = './data/segmentation_dataset/image/p01-9999(41323.14003)500,100-crop_cell_tif.tif'
classification_sample_path = './data/classification_dataset/train/CTCL/0_crop_3_Horizontal Flip.png'

# response = requests.post("http://localhost:8000/segmentation", files = {'image': open(segmentation_sample_path, 'rb')})
response = requests.post("http://localhost:8000/classification", files = {'image': open(classification_sample_path, 'rb')})

result = response.json()
# print(result)
result = np.array(result['result']).squeeze()
result = torch.sigmoid(torch.tensor(result))
print(result)
# plt.imshow(result, cmap='gray')
# plt.axis('off')
# plt.show()