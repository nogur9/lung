import os

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, io, color, exposure, img_as_float, transform
from matplotlib import pyplot as plt
import cv2
def loadDataGeneral(df, path, im_shape):
    X = []
    directory = '/home/martin/Python'
    for filename in os.listdir(directory):
        print(filename)
        img = img_as_float(io.imread(filename))
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        X.append(img)
    X = np.array(X)
    X -= X.mean()
    X /= X.std()

    print ('### Dataset loaded')
    print ('\t{}'.format(path))
    print ('\t{}\t{}'.format(X.shape, y.shape))
    print ('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print ('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))

    return X

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)


def masked2(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask == 1] = [0, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = 'idx.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path = 'data2/'

    df = pd.read_csv(csv_path)

    # Load test data
    im_shape = (256, 256)
    X = loadDataGeneral(df, path, im_shape)

    n_test = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    model_name = '../trained_model.hdf5'
    UNet = load_model(model_name)

    # For inference standard keras ImageGenerator can be used.
    test_gen = ImageDataGenerator(rescale=1.)

    ious = np.zeros(n_test)
    dices = np.zeros(n_test)

    gts, prs = [], []
    i = 0
    plt.figure(figsize=(10, 10))
    for x in X:
        img = exposure.rescale_intensity(np.squeeze(x), out_range=(0,1))
        pred = UNet.predict(x[np.newaxis,:,:,:])[..., 0].reshape(inp_shape[:2])

        #Parameters
        pr = pred > 0.5

        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))

        prs.append(pr)
        print(df.iloc[i][0], ious[i], dices[i])
        plt.axis('off')
        plt.imsave(cv2.bitwise_and(img, img, mask=pred.astype(np.uint8)), cmap=plt.cm.bone)


    print('Mean IoU:', ious.mean())
    print('Mean Dice:', dices.mean())
    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()
