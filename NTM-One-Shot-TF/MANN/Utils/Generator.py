import tensorflow as tf
import numpy as np

import os
import random
import pickle
#from Images import get_shuffled_images, time_offset_label, load_transform

class OmniglotGenerator(object):
    """Docstring for OmniglotGenerator"""
    def __init__(self, data_folder, batch_size=1, nb_classes=5, nb_samples_per_class=10, max_rotation=-np.pi/6, max_shift=10, img_size=(20,20), max_iter=None):
        super(OmniglotGenerator, self).__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_rotation = max_rotation
        self.max_shift = max_shift
        self.img_size = img_size
        self.max_iter = max_iter
        self.num_iter = 0
        self.character_folders = [os.path.join(self.data_folder, family, character) \
                                  for family in os.listdir(self.data_folder) \
                                  if os.path.isdir(os.path.join(self.data_folder, family)) \
                                  for character in os.listdir(os.path.join(self.data_folder, family))]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            return (self.num_iter - 1), self.sample(self.nb_classes)
        else:
            raise StopIteration

    def sample(self, nb_classes):
        sampled_character_folders = random.sample(self.character_folders, nb_classes)
        random.shuffle(sampled_character_folders)

        example_inputs = np.zeros((self.batch_size, nb_classes * self.nb_samples_per_class, np.prod(self.img_size)), dtype=np.float32)
        example_outputs = np.zeros((self.batch_size, nb_classes * self.nb_samples_per_class), dtype=np.float32)     #notice hardcoded np.float32 here and above, change it to something else in tf

        for i in range(self.batch_size):
            labels_and_images = get_shuffled_images(sampled_character_folders, range(nb_classes), nb_classes=self.nb_samples_per_class)
            sequence_length = len(labels_and_images)
            labels, image_files = zip(*labels_and_images)

            angles = np.random.uniform(-self.max_rotation, self.max_rotation, size=sequence_length)
            shifts = np.random.uniform(-self.max_shift, self.max_shift, size=sequence_length)

            example_inputs[i] = np.asarray([load_transform(filename, angle=angle, s=shift, size=self.img_size).flatten() \
                                            for (filename, angle, shift) in zip(image_files, angles, shifts)], dtype=np.float32)
            example_outputs[i] = np.asarray(labels, dtype=np.int32)

        return example_inputs, example_outputs


class CifarGenerator(object):
    """Docstring for Cifar10Generator"""
    def __init__(self, data_folder, batch_size=1, nb_classes=1, _class=0, nb_samples_per_class=10, img_size=(32,32,3), max_iter=None):
        super(CifarGenerator, self).__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.img_size = img_size
        self.max_iter = max_iter
        self.num_iter = 0
        self.data_batch_file = 1
        self.which_class = _class
        self.file_name = "{}/data_batch_{}".format(self.data_folder, self.data_batch_file)
        self.full_data_images, self.full_data_labels = None, None

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


    def load_images(self):
        with open(self.file_name, 'rb') as file:
            unpickler = pickle._Unpickler(file)
            unpickler.encoding = 'latin1'
            contents = unpickler.load()
            self.full_data_images, self.full_data_labels = np.asarray(contents['data'], dtype=np.float32), np.asarray(contents['labels'])

    def next(self):
        if(self.num_iter == 0):
            self.load_images()

        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            return (self.num_iter - 1), self.sample(self.nb_classes)
        else:
            raise StopIteration

    def sample(self, nb_classes):

        example_inputs = np.zeros((self.batch_size, nb_classes * self.nb_samples_per_class, np.prod(self.img_size)), dtype=np.float32)
        example_outputs = np.zeros((self.batch_size, nb_classes * self.nb_samples_per_class), dtype=np.float32)     #notice hardcoded np.float32 here and above, change it to something else in tf

        for i in range(self.batch_size):

            indices = np.where(self.full_data_labels == self.which_class)
            images = self.full_data_images[indices][(self.num_iter - 1) * self.nb_samples_per_class : self.num_iter * self.nb_samples_per_class]
            labels = self.full_data_labels[indices][(self.num_iter - 1) * self.nb_samples_per_class : self.num_iter * self.nb_samples_per_class]

            example_inputs[i] = np.asarray(images, dtype=np.float32)
            example_outputs[i] = np.asarray(labels, dtype=np.int32)

        return example_inputs, example_outputs