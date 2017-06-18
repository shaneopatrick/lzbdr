import numpy as np
import cv2
import matplotlib.pyplot as plt
import boto
import os
import pickle
# from scipy.misc import imresize, imread

access_key = os.environ['AWS_ACCESS_KEY_ID']
sec_access_key = os.environ['AWS_SECRET_ACCESS_KEY']



def load_npz(x, y):
    X = x['arr_0']
    Y = y['arr_0']
    return X, Y


def retrieve_from_S3(path):
    """
    Download npz's from S3 bucket.
    """
    conn = boto.connect_s3(access_key, sec_access_key)
    bucket = conn.get_bucket('lazybirder')
    file_key = bucket.get_key(path)
    h5 = file_key.get_contents_to_filename(path)
    # X = np.load(path)
    return h5

def crop_square(img, fit_for_test=False):
    if fit_for_test:
        target_size = (224, 224)
    else:
        target_size = (256, 256)
    if abs(img.shape[0] - img.shape[1]) > 5:
        min_dim = min(img.shape)
        buff = int(abs(img.shape[0] - img.shape[1])/2)
        if img.shape[0] < img.shape[1]: # W greater than H
            output = img[:, buff:-buff]
        else: # H greater than W
            output = img[buff:-buff,:]
        output_final = cv2.resize(output, target_size)
    else:
        output = img
        output_final = cv2.resize(output, target_size)
    return output_final

### Make squares and keep aspect ratio
def square_em_up(img, black=True):
    target_size = (224, 224)
    if img.shape[0] != img.shape[1]:
        max_dim = max(img.shape)
        buff = int(abs(img.shape[0] - img.shape[1])/2)
        if black:
            if img.shape[0] < img.shape[1]: # W greater than H
                up_buff = np.zeros((buff, max_dim, 3), dtype=np.int)
                low_buff = np.zeros((buff, max_dim, 3), dtype=np.int)
                temp = np.vstack((up_buff, img))
                output = np.vstack((temp, low_buff))
            else: # H greater than W
                l_buff = np.zeros((max_dim, buff, 3), dtype=np.int)
                r_buff = np.zeros((max_dim, buff, 3), dtype=np.int)
                temp = np.hstack((l_buff, img))
                output = np.hstack((temp, r_buff))
        else:
            if img.shape[0] < img.shape[1]: # W greater than H
                up_buff = np.zeros((buff, max_dim, 3), dtype=np.int) + 255
                low_buff = np.zeros((buff, max_dim, 3), dtype=np.int) + 255
                temp = np.vstack((up_buff, img))
                output = np.vstack((temp, low_buff))
            else: # H greater than W
                l_buff = np.zeros((max_dim, buff, 3), dtype=np.int) + 255
                r_buff = np.zeros((max_dim, buff, 3), dtype=np.int) + 255
                temp = np.hstack((l_buff, img))
                output = np.hstack((temp, r_buff))
    else:
        output = img
    output_final = imresize(output, target_size)
    return output_final
    #cv2.imwrite(outpath, output_final)

def load_dict():
    with open('../data/top200_dict.pkl', 'rb') as f:
        b_dict = pickle.load(f)
        return b_dict

def preprocess_image(filepath):
    img_full_path = '../app/uploads/{}'.format(filepath)
    image = cv2.imread(img_full_path)
    image = crop_square(image, True)
    image = image/255
    image = image.reshape((1,) + image.shape)
    return image


def return_top_n(pred_arr, n=3):
    '''
    IN: flattened prediction array
    OUT: top n indices (dict keys) and percentages
    '''
    sorts = np.argsort(-pred_arr)
    topn_keys = sorts[:n]
    percents = pred_arr[topn_keys]
    return topn_keys, percents


### Measure Accuracy

def score_top5(y_test, probs):
    sorts = np.argsort(probs, axis=1)
    top_5 = sorts[:,-5:]
    zip_list = list(zip(y_test, top_5))
    results = []
    for z in zip_list:
        if z[0] in z[1]:
            results.append(1)
        else:
            results.append(0)
    score =  np.mean(results)
    print('{} percent accuracy for top 5 predictions...'.format(score * 100))

def score_top3(y_test, probs):
    sorts = np.argsort(probs, axis=1)
    top_3 = sorts[:,-3:]
    zip_list = list(zip(y_test, top_3))
    results = []
    for z in zip_list:
        if z[0] in z[1]:
            results.append(1)
        else:
            results.append(0)
    score =  np.mean(results)
    print('{} percent accuracy for top 3 predictions...'.format(score * 100))

def accuracy_topn(y_test, probs, n=5):
    if y_test.ndim == 2:
        y_test = np.argmax(y_test, axis=1)

    predictions = np.argsort(probs, axis=1)[:, -n:]

    accs = np.any(predictions == y_test[:, None], axis=1)

    return np.mean(accs)


### Plotting Model Performance

def model_summary_plots(history, name):
    plt.figure(figsize=(10,10))
    # summarize history for accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.savefig('plots/{}_model_accuracy_loss.png'.format(name))
