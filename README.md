# Nihongo Hiragana Ricognition
The goal of the project is to train a model that can recognize various Hiragana handwritings

In this project, the training set of 1405 images are split into *training data* and *testing data* in the ratio of *85:15*. The traing data is further split into *trainging set* and *validation set*. A CNN model is constructed to train on the *training set* and be validated by the *validation set*. The trained model is then evaluated by the *testing data*. A 5-fold cross validation test is further constructed from the *training data* to test the integrity of the model. The f1 score of five trained cross validation models is $0.96 \pm 0.02$, proving the good integrity of the model. Please refer to `hiragana_recognition.ipynb` for the demonstration.


## Results
The final trained model is selected by the model of the five validation models with the lowest cost. It is then tested by *testing data* (total 211 images split from the data), obtaining 97% accuracy.

The model is then tested by 92 handwritten Hiraganas of my own, reaching 77% accuracy. Please take a look at `./test_my_handwriting`.


## Modules used
External libraries used:
* `numpy`: Calculates and manipulates matrices (images)
* `pandas`: Constructs a datatable to organize and manipulate all data
* `cv2`: Reads images into 2-D arrays
* `matplotlib`: Plots images with `pyplot`
* `sklearn`: Splits and shuffles data into training and testing by `model_selection.train_test_split`
* `tensorflow.keras`: Augments training data by `preprocessing.image.ImageDataGenerator`. Categorizes data in one-hot encoding by `utils.np_utils`. Constructs a CNN model by `models`, `layers`.

Internal modules used:
* `os`: Finds paths and files
* `re`: Extracts "Romanji" (pronunciation of Hiragana) from each filename


## Programs  included
* `hiragana_recognition.ipynb`: 
    * `organized_data.py`, `label_process.py`: Preprocesses data and extracts "Romanji labels" using `os`, `re`, `numpy`, `pandas`, `cv2`, `sklearn`.
    * `image_process.py`: Preprocesses images, eliminates noises using `cv2`.
    * Augments training data via `ImageDataGenerator`.
    * `ml_model.py`: Trains the CNN model by `keras`. 
        * The CNN model consists of two convolution blocks and one dense layer, each convolution block containing two convolution layers and a maxpooling layer.
    * `demonstration.py`: Plots the confusion matrix and the wrong predictions with `pyplot`.
* `model_hiragana_recognition_cnn.h5`: Saved model trained in `hiragana_recognition.ipynb`.
* `test_loadmodel.ipynb`: Loads `model_hiragana_recognition_cnn.h5` and certifies that it can be used on recognizing handwritten Hiraganas.


## Data sources
The 1000 images of the dataset come from Matheus Inoue's [hirgana-dataset](https://github.com/inoueMashuu/hiragana-dataset) on his Github repository.
The other 405 images come from my friend Wishyut. Please take a look at `./handwriting_wishyut`.

The dataset contains 1405 images, each of them having a handwritten Hiragana character. There are total 50 Hiragana characters, each character corresponding to 20-29 images.
