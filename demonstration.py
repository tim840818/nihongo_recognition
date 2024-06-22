import itertools
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_train_history(train_history):
    plt.figure(figsize=(18,4))
    plt.suptitle('CNN train history', fontsize=18)

    ax1 = plt.subplot(1, 2, 1)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Number of iterations', fontsize=12)
    plt.plot(train_history.history['accuracy'], color='b', label='Training Accuracy')
    plt.plot(train_history.history['val_accuracy'], color='r', label='Validation Accuracy')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='lower right')

    ax2 = plt.subplot(1, 2, 2)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Number of iterations', fontsize=12)
    plt.plot(train_history.history['loss'], color='b', label='Training Loss')
    plt.plot(train_history.history['val_loss'], color='r', label='Validation Loss')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='upper right')
    plt.show()

def get_predictions(X_test, model):
    prob = model.predict(X_test)
    predictions = np.argmax(prob,axis=1)
    return predictions

def show_errs(X_test, predictions, ground_truth, category_dict):
    inv_category_dict = {v:k for k,v in category_dict.items()} # invert the category dictionary (index -> readable label)

    nVal = len(predictions)
    print(f"total testing data: {nVal}")

    ## print errors
    errors = np.where(predictions != ground_truth)[0]
    num_err = len(errors)
    print("Number of errors = {}/{}".format(num_err,nVal))
    print("Accuracy: {:.2f}".format(1-num_err/nVal))

    num_eachrow = 4
    rows_err = 2 #math.ceil(num_err / num_eachrow)
    fig, axes = plt.subplots(rows_err, num_eachrow, figsize=(8*num_eachrow,8*rows_err))

    for r in range(rows_err):
        for c in range(num_eachrow):
            n_err = r * num_eachrow + c
            if n_err >= num_err: break
            pos_err = errors[n_err]
            # plt.title(f"predicted: {prediction_romanji_myhw[pos_err]}, answer: {answers_myhw[pos_err]}")
            axes[r,c].set_title(f"predicted: {inv_category_dict[predictions[pos_err]]}, answer: {inv_category_dict[ground_truth[pos_err]]}", fontsize=18)
            axes[r,c].imshow(X_test[pos_err], cmap=plt.get_cmap('gray'))

def plot_confusion_matrix(M, category_dict, title='Confusion matrix', show_all=False, normalize=False, cmap=plt.cm.Blues, figsize=(20,20)):
    ## normalize option
    if normalize:
        M = M.astype('float') / M.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    ## delete rows/columns with only diagonal element (Since they are correct and thus not informative for the confusion matrix)
    if not show_all:
        false_array = ~np.all((M - np.diag(np.diag(M))) == 0, axis=0) | ~np.all((M - np.diag(np.diag(M))) == 0, axis=1) # get the index of the rows/columns with at least one non-diagonal element
        false_elements = np.where(false_array)[0] # get the index of the false elements
        false_dict = {v:i for i, v in enumerate(false_elements)} # map the false elements to the new index

        M = M[false_elements][:, false_elements] # get the new confusion matrix
        category_dict = {k:false_dict[v] for k,v in category_dict.items() if v in false_elements} # get the new category dictionary

    inv_category_dict = {v:k for k,v in category_dict.items()} # invert the category dictionary (index -> readable label)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=18)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")

    tick_marks = np.arange(len(category_dict))
    tick_labels = [inv_category_dict[i] for i in tick_marks]

    ax.xaxis.set_ticks(tick_marks) # rotation=45
    ax.xaxis.set_ticklabels(tick_labels)
    ax.yaxis.set_ticks(tick_marks)
    ax.yaxis.set_ticklabels(tick_labels)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.xaxis.set_label_position('top')

    # fmt = '{:.2f}' if normalize else '{:d}'
    thresh = M.max() / 2.
    for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
        value = M[i,j]
        if not normalize:
            value_str = "{:d}".format(value)
        else:
            if value == 0:
                value_str = "0"
            else:
                value_str = "{:.2f}".format(value)
        ax.text(j, i, value_str,
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if M[i, j] > thresh else "black")
