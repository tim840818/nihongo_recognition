import os, requests, zipfile
import numpy as np
import re
import pandas as pd
import cv2 as cv

from image_process import im_bin, im_reshape, IMG_SIZE

def download_hiragana_dataset():
    directory = "hiragana-dataset-master/hiragana_images"

    if not os.path.exists(directory):
    # download from https://github.com/inoueMashuu/hiragana-dataset/archive/refs/heads/master.zip
        url = "https://github.com/inoueMashuu/hiragana-dataset/archive/refs/heads/master.zip"
        filename = "hiragana-dataset-master.zip"

        response = requests.get(url)
        with open(filename, "wb") as file:
            file.write(response.content)

        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(".")

        os.remove(filename)

def get_romanji(filename): ## extract the romanji from a filename
    keyword = re.search(r'kana(\w+?)\d+', filename).group(1)
    return keyword.lower()

def create_handwriting_table():
    handwriting_list = []
    dir_name = "hiragana-dataset-master/hiragana_images"
    # os.listdir(dir_name)
    for filename in os.listdir(dir_name):
    # print(f"{dir_name}/f{filename}.jpg")
        picture_label_list = [cv.imread(f"{dir_name}/{filename}", cv.IMREAD_GRAYSCALE), filename, get_romanji(filename)]
        handwriting_list.append(picture_label_list)
    # print(handwriting_list)

    handwriting_table = pd.DataFrame(handwriting_list, columns=["Handwriting", "Filename", "Romanji"])
    return handwriting_table

def get_handwriting_wishyut():
    handwriting_list_wishyut = []
    dir_name_wishyut = "handwriting_wishyut"
    for filename in os.listdir(dir_name_wishyut):
        picture_label_list = [cv.imread(f"{dir_name_wishyut}/{filename}", cv.IMREAD_GRAYSCALE), filename, re.search(r'_(\w+?)_', filename).group(1)]
        handwriting_list_wishyut.append(picture_label_list)

    handwriting_table_wishyut = pd.DataFrame(handwriting_list_wishyut, columns=["Handwriting", "Filename", "Romanji"])
    handwriting_table_wishyut.Handwriting = handwriting_table_wishyut.Handwriting.map(lambda img: im_reshape(im_bin(img), IMG_SIZE[0], IMG_SIZE[1]))

    return handwriting_table_wishyut

def std_X(X):
    X = X.astype('float32') / 255
    return X

def load_my_handwritings(img_rows=IMG_SIZE[0], img_cols=IMG_SIZE[1]):
    my_handwritings = []
    answers_myhw = []
    img_rows, img_cols = IMG_SIZE

    for file in os.listdir("test_my_handwriting"):
        key = re.search(r'_(\w+?)_', file).group(1)
        answers_myhw.append(key)
        
        img = cv.imread(f"test_my_handwriting/{file}", cv.IMREAD_GRAYSCALE)
        img = 255 - img
        img = im_reshape(im_bin(img), img_rows, img_cols)
        my_handwritings.append(img)

    my_handwritings = np.array(my_handwritings)
    return my_handwritings, answers_myhw
