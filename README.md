# Image Classification with CNNs

This repository contains an image classification project using Convolutional Neural Networks (CNNs) implemented with TensorFlow and Keras. The goal of this project is to classify images into predefined categories. The dataset used for training and evaluation is organized into separate directories for each class.

## Project Structure

```
├── data
│   └── Train
│       ├── class1
│       ├── class2
│       └── ...
├── New_model
│   └── new.h5
├── README.md
└── main.ipynb
```

## Dependencies

To run this project, you need the following libraries:

- TensorFlow
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib

You can install the required libraries using the following command:

```bash
pip install tensorflow numpy opencv-python scikit-learn matplotlib
```

## Dataset

The dataset should be organized in a directory named `data/Train/` with subdirectories for each class. Each subdirectory contains the images belonging to that class. For example:

```
data/Train/
├── class1
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Loading the Dataset

`tf.keras.utils.image_dataset_from_directory` is used to load images from a directory into a TensorFlow dataset. This function creates a dataset of images and their corresponding labels directly from the directory structure.

### Example Usage

```python
data = tf.keras.utils.image_dataset_from_directory(
    'data/Train/',
    image_size=(256, 256),
    batch_size=32
)
```

**Parameters:**

- `directory`: Path to the directory where the dataset is located.
- `image_size`: Size to which all images found will be resized.
- `batch_size`: Size of the batches of data.

This function will automatically label the images based on the subdirectory names and split them into batches.

## Training the Model

The model is defined using TensorFlow and Keras and consists of multiple convolutional and pooling layers followed by dense layers. The dataset is split into training, validation, and test sets, and the model is trained on the training set.

To train the model, run the `main.ipynb` script:

```bash
python main.ipynb
```

The script will:

1. Load and preprocess the dataset.
2. Split the data into training, validation, and test sets.
3. Define and compile the CNN model.
4. Train the model on the training set.
5. Evaluate the model on the test set.
6. Save the trained model to `New_model/new.h5`.

## Cross-Validation

Stratified K-Fold cross-validation is used to evaluate the model's performance across multiple folds. The average accuracy across all folds is calculated and printed.

## Results

The model's performance is evaluated using the following metrics:

- Training and validation accuracy and loss during training.
- Test accuracy and loss after training.
- Confusion matrix to visualize the model's performance on the test set.

## Usage

To use the trained model for inference or further analysis, load the model from the saved file:

```python
from tensorflow.keras.models import load_model

model = load_model('New_model/new.h5')
```

## Visualization

The script includes functions to plot sample images from the training and test sets with their corresponding labels. Additionally, a confusion matrix is plotted to visualize the model's performance on the test set.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenCV](https://opencv.org/)

