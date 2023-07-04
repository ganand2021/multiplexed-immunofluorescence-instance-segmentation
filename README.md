# multiplexed-immunofluorescence-instance-segmentation

### Description

Obtaining cell annotations of tumor cells present several challenges and difficulties like Heterogeneity, Ambiguous cell boundaries, Lack of definitive markers, Evolution of tumor cells and Technical limitations to name a few. Addressing these difficulties requires advancements in imaging technologies, development of standardized annotation guidelines, improved automation and artificial intelligence (AI) algorithms for cell identification, and collaboration among experts to establish consensus in cell annotation practices.

By utilizing advanced deep learning algorithms, this project aims to provide accurate annotations for tumor cells within the images, enabling researchers and medical practitioners to gain valuable insights into the growth patterns and characteristics of tumors. The project's underlying motivation lies in hope that we're able to create product that can accelerate the diagnostic process, reduce human error, and enhancing the overall efficiency of healthcare process while treating such diseases.

### Tech Stack

- Python
- PyTorch
- NVIDIA MONAI
- FastAPI

### Installation

To begin the installation process, please follow these steps:

1. Clone the repository containing the required code.
2. Change your working directory to the location where the repository has been cloned.
3. Execute the provided code in the command line interface.  
```
pip install -r requirements.txt
```  
*If you need to uninstall the packages, please execute the following code in the command line interface.*  
```
pip uninstall -r requirements.txt -y
```
---
### Usage
Currently, our deployed models are accessible through an API built with FastAPI. To utilize them for inference, you can locally deploy the server by executing the following code in the terminal:

```
uvicorn app:app --reload
```

### Use Cases

#### Segmentation Model
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

#### Classification Model
```
    import requests

    URL_ENDPOINT = 'http://127.0.0.1:8000'
    classification_sample_path = 'path_to_png_image'

    response = requests.post("http://localhost:8000/classification", files = {'image': open(classification_sample_path, 'rb')})

    result = response.json()
    print(result)
```

#### Classification Model
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

The data utilized for this project has been acquired from the following research paper, along with its associated dataset link.

_[Cross-platform dataset of multiplex fluorescent cellular object image annotations](https://www.nature.com/articles/s41597-023-02108-z)_  
_[Dataset](https://www.synapse.org/#!Synapse:syn27624812/files/)_  
---