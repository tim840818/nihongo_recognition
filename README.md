# Nihongo Hiragana Ricognition
In this project, the training set of 1000 images are split into 800 (training data) and 200 (testing data). A CNN model is constructed to train on the 800 images, and the trained model is then validated by the testing data. The trained model is saved as `model_hiragana_recognition_cnn.h5`, obtaining 99.5% accuracy for 200 testing data.


The dataset comes from Matheus Inoue's [hirgana-dataset](https://github.com/inoueMashuu/hiragana-dataset) on his Github repository. All the data trained in this model are from his dataset.
The dataset contains 1000 images, each of them having a handwritten Hiragana character. There are total 50 Hiragana characters, each character corresponding to 20 images.

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
    * Augments training data via `ImageDataGenerator` and trains the CNN model by `keras`. 
    * Plots the results with `pyplot`.
* `hiragana_recognition.py`: Same code as `hiragana_recognition.ipynb` but re-organized.
* `model_hiragana_recognition_cnn.h5`: Saved model trained in `hiragana_recognition.ipynb` and compressed as `model_hiragana_recognition_cnn.zip`.
