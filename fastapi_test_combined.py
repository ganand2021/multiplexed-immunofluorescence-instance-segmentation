import requests
import matplotlib.pyplot as plt
import numpy as np

URL_ENDPOINT = 'http://127.0.0.1:8000'
combined_sample_path = './data/segmentation_dataset/image/p01-10005(54191.8632)600,100-crop_tif.tif'

response = requests.post("http://localhost:8000/combined", files = {'image': open(combined_sample_path, 'rb')})

result = response.json()
print(result)