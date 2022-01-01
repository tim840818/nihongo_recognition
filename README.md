# Nihongo Hiragana Ricognition
In this project, the training set of 1405 images are split into 1124 (training data) and 281 (testing data). A CNN model is constructed to train on the 800 images, and the trained model is then validated by the testing data. The trained model is saved as `model_hiragana_recognition_cnn.h5`, obtaining 97% accuracy for 281 testing data.

The trained model is then tested by 92 handwritings of my own, reaching 83% accuracy. Please take a look at `./test_my_handwriting`.

The 1000 images of the dataset come from Matheus Inoue's [hirgana-dataset](https://github.com/inoueMashuu/hiragana-dataset) on his Github repository.
The other 405 images come from my friend Wishyut Pitawanik. Please take a look at `./handwriting_wishyut`.

The dataset contains 1405 images, each of them having a handwritten Hiragana character. There are total 50 Hiragana characters, each character corresponding to 20-29 images.

## Modules used

External libraries used:
* `numpy`: Calculates and manipulates matrices (images)
* `pandas`: Constructs a datatable to organize and manipulate all data
* `cv2`: Reads images into 2-D arrays
* `matplotlib`: Plots images with `pyplot`
* `sklearn`: Splits and shuffles data into training and testing by `model_selection.train_test_split`
* `keras`: Augments training data by `preprocessing.image.ImageDataGenerator`. Categorizes data in one-hot encoding by `utils.np_utils`. Condtructs a CNN model by `models`, `layers`.

Internal modules used:
* `os`: Finds paths and files
* `re`: Extracts "Romanji" (pronunciation of Hiragana) from each filename

## Programs  included
* `hiragana_recognition.ipynb`: 
    * Preprocesses data and extracts "Romanji labels" using `os`, `re`, `numpy`, `pandas`, `cv2`, `sklearn`.
    * Preprocesses images, eliminates noises using `cv2`.
    * Augments training data via `ImageDataGenerator`
    * Trains the CNN model by `keras`. 
        * The CNN model consists of two convolution blocks and two dense layers, each convolution block containing two convolution layers and a maxpooling layer.
    * Plots the confusion matrix and the wrong predictions with `pyplot`.
* `model_hiragana_recognition_cnn.h5`: Saved model trained in `hiragana_recognition.ipynb`.
* `test_loadmodel.ipynb`: Loads `model_hiragana_recognition_cnn.h5` and certifies that it can be used on recognizing Hiragana handwritings.
