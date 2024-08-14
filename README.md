## Early Breast Cancer Detection with Liquid Neural Networks (LNN)

This repository contains the implementation of an early breast cancer detection model using Liquid Neural Networks (LNNs) with Neural Circuit Policies (NCPs). The model processes ultrasound breast images from the BreastMNIST dataset to classify them as benign or malignant. The objective of this model is not just to detect cancer but also to achieve this with great computational efficiency, reducing parameter count by leveraging the concept of LNNs.

### Repository Contents
- **A_LNN_BreastMNIST.ipynb**: Jupyter Notebook for training the LNN model on the BreastMNIST dataset.
- **B_DNN_BreastMNIST.ipynb**: Jupyter Notebook for training the DNN model on the BreastMNIST dataset.
- **C_CNN_BreastMNIST.ipynb**: Jupyter Notebook for training the CNN model on the BreastMNIST dataset.
- **D_Experimentation_pneumonia.ipynb**: Jupyter Notebook for experimenting with the LNN model on the PneumoniaMNIST dataset.
- **LNN_model.py**: Python module containing the implementation of the LNN model.
- **CNN_model.py**: Python module containing the implementation of the CNN model.
- **DNN_model.py**: Python module containing the implementation of the DNN model.

### Dataset: BreastMNIST and PneumoniaMNIST from MedMNIST
- **BreastMNIST**: Based on a dataset of 780 breast ultrasound images, categorized into 3 classes: normal, benign, and malignant. We simplify the task into binary classification by combining normal and benign as positive and classifying them against malignant as negative. The source images of 1 × 500 × 500 are resized into 1 × 28 × 28. [BreastMNIST Dataset](https://medmnist.com/)

- **PneumoniaMNIST**: A subset of the MedMNIST dataset used for pneumonia detection, consisting of chest X-ray images labeled as normal or pneumonia. The dataset is formatted for binary classification. [PneumoniaMNIST Dataset](https://medmnist.com/) 'This is for just experimentation purpose only'

### Prerequisites to Run the Program
Here is the list of all the packages you need to install to run the program:

* `torch`: PyTorch is used for building and training the neural network models.
* `torchvision`: This package provides datasets, model architectures, and image transformations for computer vision tasks.
* `tqdm`: Used for displaying progress bars during training and data loading, helping to monitor the progress of the model training process.
* `medmnist`: Provides easy access to the MedMNIST datasets, including BreastMNIST and PneumoniaMNIST.
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
1. __Activate conda environment__:
    ```bash
    source <your-virtual-environment>/bin/activate
    ```
2. __Clone the repository__:
    ```bash
    git clone https://github.com/2ai-lab/LLNs-for-Early-Breast-Cancer-Detection
    ```
3. __Change directory to LLNs-for-Early-Breast-Cancer-Detection__
```bash
    cd LLNs-for-Early-Breast-Cancer-Detection
```
4. __Install the required packages__:
    ```bash
    pip install torch torchvision tqdm medmnist matplotlib seaborn scikit-learn opencv-python ncps
    ```
5. __Install the jupyterlab__:
    ```bash
    pip install jupyterlab
    ```
6. __Create the folders to save traned models__:
    ```bash
    mkdir saved_models experimented_models
    ```
7. __Now start the Notebook__ :
   ```bash
    jupyter lab
    ```
   * Open the Jupyter Notebooks A_LNN_BreastMNIST.ipynb, B_DNN_BreastMNIST.ipynb, or C_CNN_BreastMNIST.ipynb to train the respective models on the BreastMNIST dataset.
   * For experimenting with the PneumoniaMNIST dataset, open and run D_Experimentation_pneumonia.ipynb.
   * Make sure that the corresponding model files (e.g., LNN_model.py, DNN_model.py, CNN_model.py) are in the same directory as the notebooks.
    

