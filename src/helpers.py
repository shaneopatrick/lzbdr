import numpy as np
import cv2
import matplotlib.pyplot as plt
import boto
import os
import pickle
import tensorflow as tf

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
    file_key.get_contents_to_filename(path)
    X = np.load(path)
    return X

def _center_image(img, new_size=[224, 224]):
    '''
    Helper function. Takes rectangular image resized to be max length on at least one side and centers it in a black square.
    Input: Image (usually rectangular - if square, this function is not needed).
    Output: Image, centered in square of given size with black empty space (if rectangular).
    '''
    row_buffer = (new_size[0] - img.shape[0]) // 2
    col_buffer = (new_size[1] - img.shape[1]) // 2
    centered = np.zeros((224, 224, 3), dtype=np.uint8)
    centered[row_buffer:(row_buffer + img.shape[0]), col_buffer:(col_buffer + img.shape[1])] = img
    return centered

def resize_image_to_square(img, new_size=((224, 224))):
    '''
    Resizes images without changing aspect ratio. Centers image in square black box.
    Input: Image, desired new size (new_size = [height, width]))
    Output: Resized image, centered in black box with dimensions new_size
    '''
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*new_size[1]/img.shape[0]),new_size[1])
    else:
        tile_size = (new_size[1], int(img.shape[0]*new_size[1]/img.shape[1]))
    # print(cv2.resize(img, dsize=tile_size))
    return _center_image(cv2.resize(img, dsize=tile_size), new_size)


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


def preprocess_image(filepath):
    img_full_path = '../app/uploads/{}'.format(filepath)
    image = cv2.imread(img_full_path)
    image = crop_square(image, True)
    image = image/255
    image = image.reshape((1,) + image.shape)
    return image


def crop_image(filepath, detection_graph):
    img_full_path = '../app/uploads/{}'.format(filepath)
    image_np = cv2.imread(img_full_path)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
            # Set mask for only bird objects
            mask = np.argwhere(classes[0] == 16.)
            if len(mask) < 1:
                return None, None, None
            score1 =  scores[0][mask[0]].ravel()
            classes1 = classes[0][mask[0]].ravel()
            box1 = boxes[0][mask[0]].ravel()
            # print('Box1: ', box1)
            y_min = box1[0]
            x_min = box1[1]
            y_max = box1[2]
            x_max = box1[3]

            im_height, im_width,  _ = image_np.shape
            coords = (abs(int(y_min * im_height)-20), int(y_max * im_height)+20, abs(int(x_min * im_width)-20), int(x_max * im_width)+20)

            # print('\nCoords: ', coords)

            # cv2.imshow('test', image_np)
            # cv2.waitKey(3000)

            new_img = image_np[coords[0]:coords[1], coords[2]:coords[3]]
            # cv2.imwrite('vis_util.png', image_np)
            # cv2.imwrite('cropped.png', new_img)
            # cv2.imshow('cropped', new_img)
            # cv2.waitKey(3000)
            return new_img, score1, classes1


def load_dict():
    with open('../data/top200_dict.pkl', 'rb') as f:
        b_dict = pickle.load(f)
        return b_dict

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
