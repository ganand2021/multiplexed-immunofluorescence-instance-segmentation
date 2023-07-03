import requests
import matplotlib.pyplot as plt
import numpy as np

URL_ENDPOINT = 'http://127.0.0.1:8000'
segmentation_sample_path = './data/segmentation_dataset/image/p01-10005(54191.8632)600,100-crop_tif.tif'

response = requests.post("http://localhost:8000/segmentation", files = {'image': open(segmentation_sample_path, 'rb')})

result = np.array(response.json()['result']).squeeze()
plt.imshow(result, cmap='gray')
plt.axis('off')
plt.show()