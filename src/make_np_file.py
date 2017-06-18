import cv2
import pandas as pd
import numpy as np
from helpers import *
from sklearn.preprocessing import LabelEncoder

def make_npy(df, bird1, bird2, y_path, X_path):
    # filter DataFrame
    df = df[(df['image_name'].str.contains(bird1)) | (df['image_name'].str.contains(bird2))]
    # set first row class_id to arbitrary variable
    var = df.iloc[0]['class_id']
    df['class_id'] = df['class_id'].apply(lambda x: 1 if x == var else 0)

    print('Bird class = 1: ', df.iloc[0]['image_name'])
    print('Bird class = 0: ', df.iloc[-1]['image_name'])

    target = df['class_id'].values
    np.save(y_path, target)

    zipper = zip(df.image_name, df.y, df.y_end, df.x, df.x_end)

    new_x = []
    for i, z in enumerate(zipper):
        image = cv2.imread('cub200/CUB_200_2011/images/' + z[0])
        cropped = image[z[1]:z[2], z[3]:z[4]]
        new_x.append(square_em_up(cropped))
        print(i)

    X = np.array(new_x)

    np.save(X_path, X)

def make_npy_multi(dataframe, bird_list, y_path, X_path, augment=False):
    # filter DataFrame
    if augment:
        df = dataframe[dataframe['image_name'].str.contains(bird_list[0])]
        for bird in bird_list[1:]:
            temp = dataframe[dataframe['image_name'].str.contains(bird)]
            df = pd.concat([df, temp])

        u_labels = list(df['class_id'].unique())
        labels = df['class_id']
        le = LabelEncoder().fit(u_labels)
        df['class_id'] = le.transform(labels)

        target = df['class_id'].values
        y_out = []
        for val in target:
            y_out.append(val)
            y_out.append(val)
        np.save(y_path, np.array(y_out))

        zipper = zip(df.image_name, df.y, df.y_end, df.x, df.x_end)

        new_x = []
        for i, z in enumerate(zipper):
            image = cv2.imread('cub200/CUB_200_2011/images/' + z[0])
            cropped = image[z[1]:z[2], z[3]:z[4]]
            squared = square_em_up(cropped)
            flipped = cv2.flip(squared, 1)
            new_x.append(squared)
            new_x.append(flipped)
            if i % 50 == 0:
                print('Processed {} images...'.format(i))
        X = np.array(new_x)

        np.save(X_path, X)
    else:
        df = dataframe[dataframe['image_name'].str.contains(bird_list[0])]
        for bird in bird_list[1:]:
            temp = dataframe[dataframe['image_name'].str.contains(bird)]
            df = pd.concat([df, temp])

        u_labels = list(df['class_id'].unique())
        labels = df['class_id']
        le = LabelEncoder().fit(u_labels)
        df['class_id'] = le.transform(labels)

        target = df['class_id'].values
        np.save(y_path, target)

        zipper = zip(df.image_name, df.y, df.y_end, df.x, df.x_end)

        new_x = []
        for i, z in enumerate(zipper):
            image = cv2.imread('cub200/CUB_200_2011/images/' + z[0])
            cropped = image[z[1]:z[2], z[3]:z[4]]
            new_x.append(square_em_up(cropped))
            if i % 50 == 0:
                print('Processed {} images...'.format(i))
        X = np.array(new_x)

        np.save(X_path, X)

    for label in df['class_id'].unique():
        names = df[df['class_id'] == label]['image_name'].tolist()
        name = names[0]
        print('Bird with class {}: {}'.format(label, name))

def make_npz_full(df, y_path, X_path):

    target = df['label'].values
    np.savez(y_path, target)

    zipper = zip(df.image_name, df.y, df.y_end, df.x, df.x_end)

    new_x = []
    for i, z in enumerate(zipper):
        image = cv2.imread('images/' + z[0])
        cropped = image[z[1]:z[2], z[3]:z[4]]
        new_x.append(square_em_up(cropped))
        print(i)
    X = np.array(new_x)

    np.savez(X_path, X)

def make_npz_full2(df, X_patha, X_pathb):

    # target = df['label'].values
    # np.savez(y_path, target)

    zipper = zip(df.image_name, df.y, df.y_end, df.x, df.x_end)

    new_x = []
    for i, z in enumerate(zipper):
        image = cv2.imread('images/' + z[0])
        cropped = image[z[1]:z[2], z[3]:z[4]]
        new_x.append(square_em_up(cropped))
        print(i)
    Xa = np.array(new_x[:-24281])
    Xb = np.array(new_x[-24281:])

    np.savez(X_patha, Xa)
    np.savez(X_pathb, Xb)


def make_npz_full_lz_crop(df, y_path, X_path):

    target = df['label'].values
    np.savez(y_path, target)

    zipper = zip(df.image_name, df.y, df.y_end, df.x, df.x_end)

    new_x = []
    for i, z in enumerate(zipper):
        image = cv2.imread('images/' + z[0])
        new_x.append(crop_square(image, True))
        print(i)
    X = np.array(new_x)

    np.savez(X_path, X)


def make_npy_full_enc(dataframe, y_path, X_path):

    u_labels = list(df['class_id'].unique())
    labels = df['class_id']
    le = LabelEncoder().fit(u_labels)
    df['class_id'] = le.transform(labels)

    target = df['class_id'].values
    np.save(y_path, target)

    zipper = zip(df.image_name, df.y, df.y_end, df.x, df.x_end)

    new_x = []
    for i, z in enumerate(zipper):
        image = cv2.imread('cub200/CUB_200_2011/images/' + z[0])
        cropped = image[z[1]:z[2], z[3]:z[4]]
        new_x.append(square_em_up(cropped))
        print(i)
    X = np.array(new_x)

    np.save(X_path, X)

def make_canvas_npz(dataframe, bird_list, X_path,  y_path, canvas=False):
    df = dataframe[dataframe['image_name'].str.contains(bird_list[0])]
    for bird in bird_list[1:]:
        temp = dataframe[dataframe['image_name'].str.contains(bird)]
        df = pd.concat([df, temp])

    u_labels = list(df['class_id'].unique())
    labels = df['class_id']
    le = LabelEncoder().fit(u_labels)
    df['class_id'] = le.transform(labels)

    zipper = zip(df.image_name, df.class_id)

    new_x = []
    new_y = []

    if canvas:
        for i, z in enumerate(zipper):
            image = cv2.imread('../cub200/CUB_200_2011/images/' + z[0])
            cropped = crop_square(image)
            img_lst = attention_canvas(cropped)
            for img in img_lst:
                new_x.append(img)
                new_y.append(z[1])
            if i % 50 == 0:
                print('Processed {} images...'.format(i))

    for i, z in enumerate(zipper):
        image = cv2.imread('images/' + z[0])
        cropped = crop_square(image, True)
        new_x.append(cropped)
        new_y.append(z[1])
        if i % 50 == 0:
            print('Processed {} images...'.format(i))

    X, y = np.array(new_x), np.array(new_y)
    np.save(X_path, X)
    np.save(y_path, y)


if __name__ == '__main__':
    df = pd.read_csv('full_flags.csv')
    # train = pd.read_csv('train_200.csv')
    # test = pd.read_csv('test_200.csv')



    make_npz_full2(df, 'data/X_404_a.npz', 'data/X_404_b.npz')
    # make_npz_full_lz_crop(df, 'data/y_200_lz.npz', 'data/X_200_lz.npz')

    #make_npy(df, 'Black_Throated_Sparrow', 'Harris_Sparrow', 'data/y_harris_black-throated.npy', 'data/X_harris_black-throated.npy')

    # birdies = ['Blue_Grosbeak', 'Red_Headed_Woodpecker', 'Grasshopper_Sparrow', 'Pine_Warbler', 'Acadian_Flycatcher', 'American_Pipit', 'Common_Raven', 'Dark_Eyed_Junco', 'Horned_Lark', 'Mallard']
    #make_npy_multi(train, birdies, 'data/y_trainAUG_10.npy', 'data/X_trainAUG_10.npy', True)
    #make_npy_multi(test, birdies, 'data/y_test_10.npy', 'data/X_test_10.npy')
    #make_npy_full(df, 'data/y_full.npy', 'data/X_full.npy')
    #make_npy_full_enc(df, 'data/y_full_enc.npy', 'data/X_full_enc.npy')
    # make_canvas_npz(train, birdies, '../data_ec2/X_train10_canv.npy', '../data_ec2/y_train10_canv.npy', True)
    #make_canvas_npz(test, birdies, '../data_ec2/X_test10_nocanv.npy', '../data_ec2/y_test10_nocanv.npy')
