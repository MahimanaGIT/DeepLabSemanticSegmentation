# Deeplab Semantic Segmentation

This tutorial is a simple implementation from [Deep Lab](https://github.com/tensorflow/models/tree/master/research/deeplab)

### Requirements:
1. Tensorflow 1.15
2. Python 3

<img src="https://github.com/MahimanaGIT/DeepLabSemanticSegmentation/blob/master/images/semantic.png" />

Clone the repository for object_detection_api in any location you feel comfortable: [Tensorflow Model](https://github.com/tensorflow/models) from tensorflow

The repository should be of the following for:
> Deep Lab Repostiory:
>   - models
>        - pretrained_model
>   - All the codes in the repository    

Step 1: Exporting the python path to add the object detection from tensorflow object detection API: 

    "export PYTHONPATH=$PYTHONPATH:~/repo/object_detection/object_detection/models/research/:~/repo/object_detection/object_detection/models/research/slim/"

Step 2: Run the following command in terminal from the directory "object_detection/models/research/" for compiling the proto buffers:

>   protoc object_detection/protos/*.proto --python_out=."

Step 3: Run the script to check if everything is OK in the "research" folder of the [object_detection_api](https://github.com/tensorflow/models) from tensorflow

>   python3 object_detection/builders/model_builder_test.py

Step 4: Select and download the model using the script which will save the pretrained model in the folder "./models/pretrained_model" and extract it:

>   python3 SelectingAndDownloadingModel.py

Step 5: Renaming the downloaded model folder to "pretrained_model"

Step 6: Use the frozen path from "./models/pretrained_model/frozen_inference_graph.pb" with the file name "frozen_inference_graph.pb"and run the script:

>   python3 model_main.py
