import requests

URL_ENDPOINT = 'http://127.0.0.1:8000'
classification_sample_path = './data/classification_dataset/test/CTCL/1_crop_3.png'

response = requests.post("http://localhost:8000/classification", files = {'image': open(classification_sample_path, 'rb')})

result = response.json()
print(result)