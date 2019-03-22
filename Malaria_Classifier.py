import pandas as pd
import tensorflow as tf
import numpy as np
import math
import glob
import random
import VGG_16
import os
import sklearn.model_selection as sklearn
import CNN_TF
from typing import List, Dict

tf.logging.set_verbosity(tf.logging.INFO)

#Using the Estimator approach following the standard Tensorflow tutorial
#https://www.tensorflow.org/tutorials/estimators/cnn
#The tuorial is integrated with some image augmentation and usage of Dataset API.
#Also usage of summary and Tensorboard.

#FLAGS are defined globally. They are hadnled thanks to the tf.app.run() wrapper.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("width", 96, "The width of the imamges fed to the CNN.")
flags.DEFINE_integer("height", 96, "The heigth of the imamges fed to the CNN.")
flags.DEFINE_integer("batch_size", 32, "Size of the shuffling buffer.")
flags.DEFINE_integer("shuffle_buffer_size", 15000, "Size of the shuffling buffer.")
flags.DEFINE_integer("prefetch_buffer_size", 1, "Size of the prefetching buffer.")
flags.DEFINE_integer("classes", 2, "Number of classes for this task.")
flags.DEFINE_integer("dataset_size", 27560, "Dimension of the total Dataset.")
#Difference between epoch and num_steps. Epoch is when all the samples divided in
#batches are considered. Since we use 70% of them, we get 19292
flags.DEFINE_integer("num_examples_epoch", 19292, "Number of samples in one epoch" )
#The number of epochs can be set here, it's the number for which we repeat the
#Dataset.
flags.DEFINE_integer("num_epochs", 10, "Times to go through the all training dataset.")

def main(_):
    total_num_training_steps = (FLAGS.num_examples_epoch / FLAGS.batch_size) * FLAGS.num_epochs
    training_set, test_set = split_training_test_sets()
    config = {"image_width" : FLAGS.width, "image_height" : FLAGS.height,
                    "classes" : FLAGS.classes, "batch_size" : FLAGS.batch_size}
    #Create the Estimator
    cnn_model = CNN_TF.CNN_TensorFlow(config)
    #The Estimator API builds the model's graph when calling this function.
    #When train is first called, checkpoints and other files are added in the directory
    #in which we have saved the model, along with all Event files to be visualized in
    #the Tensorboard. You can add tf.summary around in the code, but be sure to call
    #the TensorBoard on the model directory.
    cnn_estimator = tf.estimator.Estimator(
        model_fn=cnn_model.model, model_dir="./model")

    #We use a lambda since we need to pass a callable function and we have
    #argumentes in the input_fn functions.
    cnn_estimator.train(
        input_fn= lambda: training_input_fn(training_set),
        steps=total_num_training_steps
    )

    #We then evaluate the Estimator(again, the CNN model defined before).
    #We need an input functions that provides batches of the test set.
    #No argument step is provided, so it will proceed unitl Dataset is finished.
    eval_results = cnn_estimator.evaluate(
        input_fn= lambda: testing_input_fn(test_set)
    )

    #Print the final results of the evaluation.
    print(eval_results)

#Tensorflow does not provide a native function for dividing a Dataset in training
#and test sets (except a non-automatic solution with take and skip). Not only that, but Estimator
#train/evaluate/predict creates a graph when called, so we can't define a training/test
#Dataset outside these functions, becuase they would be in different graphs.
#So we get all the files' names and divide them in training and test sets along
#with their labels, and we will pass them to the functions to create the Datasets.
def split_training_test_sets():
    #We operate on the names of the files in both folders
    classes = sorted(glob.glob("./cell-images-for-detecting-malaria/cell_images" + '/*/'))
    num_classes = len(classes)
    files_dict = {}
    for counter, single_class in enumerate(classes):
        for file_name in glob.glob(single_class + "/*.png"):
            files_dict[file_name] = counter
    items = np.array(list(files_dict.items()))
    X_train, X_test = sklearn.train_test_split(items, test_size=0.3, shuffle=True)
    return X_train, X_test


#We need to define an input_fn for the Estimator (the CNN). We also uses Datasets,
#in order to perform an optimized input pipeline.
#The argument is a Tensorflow dataset.
def training_input_fn(training_dataset):
    dataset = tf.data.Dataset.from_tensor_slices(training_dataset)
    #To preserve the ordering, shuffle before repeating.
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    #Since inputs are independent, the mapping can be performed parallely,
    #so the parameter equals the numbers of cores on the machine.
    dataset = dataset.map(_parse_images, num_parallel_calls=8)
    #Perform data augmentation only on the training dataset
    dataset = dataset.map(_apply_random_augmentation, num_parallel_calls=8)
    #Define the number of epochs: basically the iterators will
    #iterate on the dataset n times before stopping.
    dataset = dataset.repeat(FLAGS.num_epochs)
    #Batching operation, always performed after shuffling/repeating.
    dataset = dataset.batch(batch_size=FLAGS.batch_size)
    #Optimization through pipelining
    #Prefetching allows to perform the preparation of the batch N+1 while
    #batch N is being currently used for training. Usually prefecth is set to
    #1, but it can vary.
    dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
    return dataset

#We need to define an input_fn for the Estimator (the CNN). We also uses Datasets,
#in order to perform an optimized input pipeline.
#The argument is a Tensorflow dataset.
def testing_input_fn(test_dataset):
    dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(test_dataset))
    #Since inputs are independent, the mapping can be performed parallely,
    #so the parameter equals the numbers of cores on the machine.
    dataset = dataset.map(_parse_images, num_parallel_calls=8)
    #Batching operation, always performed after shuffling/repeating.
    dataset = dataset.batch(batch_size=FLAGS.batch_size)
    #Optimization through pipelining
    #Prefetching allows to perform the preparation of the batch N+1 while
    #batch N is being currently used for training. Usually prefecth is set to
    #1, but it can vary.
    dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
    return  dataset

#record is an list containing the file path in the first position and ots label on
#the second position
def _parse_images(record: List[str]):
    image_string = tf.read_file(record[0])
    #Since some of the images are grayscale, we decode them to RGB, since
    #for CNNs all images must have same dimension
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_decoded = tf.cast(image_decoded, tf.float32)
    image_decoded = image_decoded/np.float(255)
    #Resizing the images, since we use Convolutional + Full dense layers
    image_resized = tf.image.resize_images(image_decoded, (FLAGS.width, FLAGS.height),
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                    align_corners=True)
    feature_dict = {"image" : image_resized}
    #In order to work with Estimators, the dataset must return a dictionary of
    #features and a tensor indicating the label.
    return feature_dict, tf.string_to_number(record[1], tf.int32)

#Performing online augmentation on the dataset by applying random transformations
#to the images. Online augmentation, the files are modified directly, not added
#as new samples.
def _apply_random_augmentation(feature_dict, label):
    image = feature_dict["image"]
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    degrees = random.randint(0, 360)
    tf.contrib.image.rotate(image, degrees * math.pi / 180, interpolation='BILINEAR')
    #Should also add random translation and cropping (with padding)
    new_feature_dict = {"image" : image}
    return feature_dict, label

if __name__ == "__main__":
    tf.app.run()
