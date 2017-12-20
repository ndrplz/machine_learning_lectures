import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.io import imread
import matplotlib.pyplot as plt


class Car:

    car_image   = 'img/car.png'

    def __init__(self, start_pos):
        self.image = imread(Circuit.car_image, mode='RGB')
        self.start



class Circuit:

    train_image = 'img/circuit_train.png'
    test_image  = 'img/circuit_test.png'

    start_pos   = (75, 260)  # row, col

    def __init__(self, mode):

        if mode not in ['train', 'test']:
            raise ValueError('Mode "{}" not valid.'.format(mode))
        image = Circuit.train_image if mode == 'train' else Circuit.test_image

        self.image = imread(image, mode='RGB')
        self.image_gray = imread(image, mode='L')

        self.reward_map = distance_transform_edt(input=np.uint8(self.image_gray == 0))

        self.car = Car()

    def start_new_episode(self):
        self.car.position = Circuit.start_position

    def show(self):
        plt.imshow(self.image)
        plt.show()


if __name__ == '__main__':

    circuit = Circuit(mode='train')
    circuit.show()
