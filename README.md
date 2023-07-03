# multiplexed-immunofluorescence-instance-segmentation

### Installation

1. Clone the Repository
2. CD into the directory
3. Run the following code in the command line
\
```
pip install -r requirements.txt
```
\
*If the packages need to removed, run the following code in the command line*
\
```
pip uninstall -r requirements.txt -y
```
---
### Usage
Presently, the trained models are provided as an API using FastAPI and can be used for inference after locally deploying the server by following code in the terminal:
\
```
uvicorn app:app --reload
```

### Use Cases

##### Segmentation Model
```
    import requests
    import matplotlib.pyplot as plt
    import numpy as np

    URL_ENDPOINT = 'http://127.0.0.1:8000'
    segmentation_sample_path = 'path_to_tiff_image'

    response = requests.post("http://localhost:8000/segmentation", files = {'image': open(segmentation_sample_path, 'rb')})

    result = np.array(response.json()['result']).squeeze()
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.show()
```

##### Classification Model
```
    import requests

    URL_ENDPOINT = 'http://127.0.0.1:8000'
    classification_sample_path = 'path_to_png_image'

    response = requests.post("http://localhost:8000/classification", files = {'image': open(classification_sample_path, 'rb')})

    result = response.json()
    print(result)
```

##### Classification Model
```
    import requests
    import matplotlib.pyplot as plt
    import numpy as np

    URL_ENDPOINT = 'http://127.0.0.1:8000'
    combined_sample_path = 'path_to_tiff_image'

    response = requests.post("http://localhost:8000/combined", files = {'image': open(combined_sample_path, 'rb')})

    result = response.json()
    print(result)
```
---
### Data

All the data used for this project has been sourced from the following paper and it's associated dataset link.

[Cross-platform dataset of multiplex fluorescent cellular object image annotations](https://www.nature.com/articles/s41597-023-02108-z)
\
[Dataset](https://www.synapse.org/#!Synapse:syn27624812/files/)
---