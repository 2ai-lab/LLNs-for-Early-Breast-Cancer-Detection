## Early Breast Cancer Detection with Liquid Neural Networks (LNN)

This repository contains the implementation of a early breast cancer detection model using Liquid Neural Networks (LNNs) with Neural Circuit Policies (NCPs). The model processes ultrasound breast images from the BreastMNIST dataset to classify them as benign or malignant.
This model objectives is not just detecting the cancer classification but also doing it with great computational efficiancy with reduced parameter count by leveraging the concept of LLNs.

### Repository Contents
- **LNN_BreastMnist_Classification.ipynb**: Jupyter Notebook for training the LNN model on the BreastMNIST dataset.
- **models.py**: Python module containing the implementation of the LNN model, and also CNN and DNN for comparision purposes.

### Dataset : BreastMNIST from MedMNIST
The BreastMNIST is based on a dataset25 of 780 breast ultrasound images. It is categorized into 3 classes: normal, benign, and malignant. As we use low-resolution images, we simplify the task into binary classification by combining normal and benign as positive and classifying them against malignant as negative.The source images of 1 × 500 × 500 are resized into 1 × 28 × 28, for more info please follow the website provided. [BreastMNIST Dataset](https://medmnist.com/)

### Prerequisites to run the program 
Here is the list of all the packages you need to install to run the program

* `torch`: PyTorch is used for building and training the neural network models.
* `torchvision`: This package provides datasets, model architectures, and image transformations for computer vision tasks.
* `tqdm`: Used for displaying progress bars during training and data loading, helping to monitor the progress of the model training process.
* `medmnist`: Provides easy access to the MedMNIST dataset, including the BreastMNIST dataset used in this project.
* `matplotlib`: Essential for plotting graphs and visualizing the training and validation results.
* `seaborn`: Enhances data visualization and aesthetics in plots, making them easier to interpret.
* `scikit-learn`: Provides additional machine learning utilities and metrics, such as evaluation metrics and data preprocessing tools.
* `opencv-python`: Used for image processing tasks, such as reading and transforming images.
* `ncps`: Implements Neural Circuit Policies (NCPs) used in the Liquid Neural Networks, which are crucial for the dynamic synaptic behavior in the model.
You can install these libraries using pip:
```bash
pip install torch torchvision tqdm medmnist matplotlib seaborn scikit-learn opencv-python ncps
```

## Steps to Run the Program

1. __Clone the repository__:
    ```bash
    git clone https://github.com/2ai-lab/LLNs-for-Early-Breast-Cancer-Detection
    ```

2. __Install the required packages__:
    ```bash
    pip install torch torchvision tqdm medmnist matplotlib seaborn scikit-learn opencv-python ncps
    ```
3. __Create a folder__ named `saved_models`, this should be in same directory along with the `models.py` and `LNN_BreastMnist_Classification.ipynb`. The trained model will be saved in the saved_models folder.
    
3. __Run the training script__:
   Open the Jupyter Notebook `LNN_BreastMnist_Classification.ipynb` and run the cells to train the model and it will train the model on the BreastMNIST dataset. Make sure you have the `models.py` and `LNN_BreastMnist_Classification.ipynb` in the same directory. 
