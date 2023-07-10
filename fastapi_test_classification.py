import requests

URL_ENDPOINT = 'http://127.0.0.1:8000'
# URL_ENDPOINT = 'https://simbiosys-390618.ue.r.appspot.com'
classification_sample_path = './data/classification_dataset/test/CTCL/1_crop_3.png'

response = requests.post(URL_ENDPOINT+"/classification", files = {'image': open(classification_sample_path, 'rb')})

result = response.json()
print(result)