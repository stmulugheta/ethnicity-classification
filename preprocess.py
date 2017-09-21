import tensorflow as tf
import numpy as np
import os
import glob
import scipy.misc
import scipy.ndimage
import h5py
import math

import cv2
import sys

def prepare_data(sess, folderpath):
    """
    Args:
    dataset: choose train dataset or test dataset
    """
    filetype = 'jpg'
    # filenames = os.listdir(folderpath)
    # print filenames
    print folderpath
    data = glob.glob(os.path.join(folderpath, "*.%s" % (filetype)))
    print data
    return data

def preprocess(path, config, race):
    """
    """
    
    p = "Cropped/Data/Images/Test/"+race
    if not os.path.exists(p):
        os.makedirs(p)

    # Get user supplied values
    imagePath = path
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if len(faces) == 1:
        crop_img = image[y: y + h, x: x + w]
        newPath = os.path.join(os.path.join(os.getcwd(), "Cropped"), path)
        cv2.imwrite(newPath, crop_img)
        
        newImage = scipy.misc.imread(newPath, flatten=True, mode='L').astype(np.float)
        dsize = 64
        size = newImage.shape[0]
        to_crop = int(math.floor(np.mod(size, dsize) / 2.0))
        new_size = newImage[to_crop:size-to_crop, to_crop:size-to_crop].shape[0]            
        scale = newImage[to_crop:size-to_crop, to_crop:size-to_crop].shape[0] / dsize
        if new_size == 64:
            _input= scipy.ndimage.interpolation.zoom(newImage[to_crop:size-to_crop, to_crop:size-to_crop], (1./scale), prefilter=False)
        else:
            _input= scipy.ndimage.interpolation.zoom(newImage[to_crop+1:size-to_crop, to_crop+1:size-to_crop], (1./scale), prefilter=False)
        print newPath
        cv2.imwrite(newPath, _input)
        
        s = newPath.split('.')[0]
        # Randomly flip the image
        _input= cv2.flip(_input,1)
        newPath = "{}-flipped.jpg".format(s)
        print newPath
        cv2.imwrite(newPath, _input)
        
        if race == 'Asian':
            _label = np.array([1, 0, 0])
        elif race == 'Black':
            _label = np.array([0, 1, 0])
        else:
            _label = np.array([0, 0, 1])
    
        return _input, _label
    else:
        return 0, 0
  
def read_data(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label
    
def input_setup(sess, config):
    if config.is_train:
        train_input_array = []
        train_label_array = []
        
        data = prepare_data(sess, 'Data/Images/CUHK')
        for i in xrange(len(data)):
            input_, label_ = preprocess(data[i], config, 'Asian')
            if type(input_) != int:
                inp = input_.reshape([config.image_size, config.image_size, 1])
                lab = label_
                train_input_array.append(inp)
                train_label_array.append(lab)

        data = prepare_data(sess, 'Data/Images/Caucasian/Front')
        for i in xrange(len(data)):
            input_, label_ = preprocess(data[i], config, 'Caucasian')
            if type(input_) != int:
                inp = input_.reshape([config.image_size, config.image_size, 1])
                lab = label_
                train_input_array.append(inp)
                train_label_array.append(lab)
        data = prepare_data(sess, 'Data/Images/Asian/Front')
        for i in xrange(len(data)):
            input_, label_ = preprocess(data[i], config, 'Asian')
            if type(input_) != int:
                inp = input_.reshape([config.image_size, config.image_size, 1])
                lab = label_
                train_input_array.append(inp)
                train_label_array.append(lab)
        data = prepare_data(sess, 'Data/Images/Black/Front')
        for i in xrange(len(data)):
            input_, label_ = preprocess(data[i], config, 'Black')
            if type(input_) != int:
                inp = input_.reshape([config.image_size, config.image_size, 1])
                lab = label_
                train_input_array.append(inp)
                train_label_array.append(lab)
        #mean_image = np.mean(input_array, axis=0)
        train_final_array = []
        for img in train_input_array:
            #img -= mean_image
            train_final_array.append(img)

        # cv2.imwrite('test.jpg', final_array[0])
        # cv2.imwrite('test2.jpg', final_array[20])
        # cv2.imwrite('test3.jpg', final_array[-1])

        print "The size of training data is: %s" % len(train_final_array)
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset('data', data=train_final_array)
            hf.create_dataset('label', data=train_label_array)

        valid_input_array = []
        valid_label_array = []
        data = prepare_data(sess, 'Data/Images/Test/Asian/')
        for i in xrange(len(data)):
            input_, label_ = preprocess(data[i], config, 'Asian')
            if type(input_) != int:
                inp = input_.reshape([config.image_size, config.image_size, 1])
                lab = label_
                valid_input_array.append(inp)
                valid_label_array.append(lab)
        data = prepare_data(sess, 'Data/Images/Test/Black/')
        for i in xrange(len(data)):
            input_, label_ = preprocess(data[i], config, 'Black')
            if type(input_) != int:
                inp = input_.reshape([config.image_size, config.image_size, 1])
                lab = label_
                valid_input_array.append(inp)
                valid_label_array.append(lab)
        data = prepare_data(sess, 'Data/Images/Test/Caucasian/')
        for i in xrange(len(data)):
            input_, label_ = preprocess(data[i], config, 'Caucasian')
            if type(input_) != int:
                inp = input_.reshape([config.image_size, config.image_size, 1])
                lab = label_
                valid_input_array.append(inp)
                valid_label_array.append(lab)
        valid_final_array = []
        for img in valid_input_array:
            #img -= mean_image
            valid_final_array.append(img)

        print "The size of validation data is: %s" % len(valid_final_array)
        savepath = os.path.join(os.getcwd(), 'checkpoint/valid.h5')
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset('data', data=valid_final_array)
            hf.create_dataset('label', data=valid_label_array)
    else:
        test_input_array = []
        test_label_array = []
        data = prepare_data(sess, 'Data/Images/Test/Instagram/Asian')
        for i in xrange(len(data)):
            input_, label_ = preprocess(data[i], config, 'Asian')
            if type(input_) != int:
                inp = input_.reshape([config.image_size, config.image_size, 1])
                lab = label_
                test_input_array.append(inp)
                test_label_array.append(lab)
        # data = prepare_data(sess, 'Data/Images/Test/Instagram/Black')
        # for i in xrange(len(data)):
        #     input_, label_ = preprocess(data[i], config, 'Black')
        #     if type(input_) != int:
        #         inp = input_.reshape([config.image_size, config.image_size, 1])
        #         lab = label_
        #         test_input_array.append(inp)
        #         test_label_array.append(lab)
        # data = prepare_data(sess, 'Data/Images/Test/Instagram/Caucasian')
        # for i in xrange(len(data)):
        #     input_, label_ = preprocess(data[i], config, 'Caucasian')
        #     if type(input_) != int:
        #         inp = input_.reshape([config.image_size, config.image_size, 1])
        #         lab = label_
        #         test_input_array.append(inp)
        #         test_label_array.append(lab)
                
        test_final_array = []
        for img in test_input_array:
            #img -= mean_image
            test_final_array.append(img)  
        print "The size of testing data is: %s" % len(test_final_array)
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset('data', data=test_final_array)
            hf.create_dataset('label', data=test_label_array)
    