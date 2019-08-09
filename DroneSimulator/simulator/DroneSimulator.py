from __future__ import division
from pyqtgraph.Qt import QtCore, QtGui
from PIL import Image
import numpy as np
import pyqtgraph as pg
import sys
import random
import threading

np.set_printoptions(threshold=sys.maxsize)


# Given the position i of the only set bit in a 24-bit string, the function returns the RGB colour.
def getColour(i):
    colour = np.zeros(shape=3)

    # Inverting the order of bit
    i = 24 - i - 1

    # Red must be the first
    colour[2 - i // 8] = 2 ** (i % 8)
    return colour


class DroneSimulator:

    def __init__(self, bitmap, batch_size, observation_range, amount_of_drones, stigmation_evaporation_speed, inertia,
                 collision_detection, max_steps, render_allowed=False, drone_colour=[255, 255, 255]):

        self.__init_simulator_parameters(bitmap, batch_size, observation_range, amount_of_drones, stigmation_evaporation_speed,
                                         inertia, collision_detection, max_steps, render_allowed, drone_colour)

        self.__init_environment_parameters()
        self.__parse_bitmap()

        self.__init_drones_parameters()
        self.__init_drones()

        self.__init_render_parameters()

    def __init_simulator_parameters(self, bitmap, batch_size, observation_range, amount_of_drones, stigmation_evaporation_speed,
                                    inertia, collision_detection, max_steps, render_allowed, drone_colour):

        self.__bitmap = bitmap
        self.__batch_size = batch_size
        self.__observation_range = observation_range
        self.__amount_of_drones = amount_of_drones
        self.__stigmation_evaporation_speed = stigmation_evaporation_speed  # TODO: non viene utilizzato per ora
        self.__inertia = inertia
        self.__collision_detection = collision_detection
        self.__max_steps = max_steps
        self.__render_allowed = render_allowed
        self.__drone_colour = drone_colour

    def __init_environment_parameters(self):
        # __environment_bitmap contains the initial bitmap, with all the fixed information (ground, targets, obstacles)
        self.__environment_bitmap = None
        self.__targets = np.array([])
        self.__collision = np.array([])
        self.__no_collision = np.array([])  # Initialized in parse_bitmap(), but currently not used

    def __parse_bitmap(self):
        input_array = np.asarray(Image.open(self.__bitmap))
        rgb_bit_array = np.unpackbits(input_array, axis=2)
        # rgb_bit_array is a matrix of pixels, where each cell (each pixel) is a 24-bit array

        # The transpose() is needed by PyQtGraph to draw the map properly
        env = []
        no_collision = []
        level_founded = 0

        for i in range(0, 24):
            level = rgb_bit_array[:, :, i]
            # Only levels with at least 1 item are inserted in the environment
            if np.any(level):
                if level_founded == 0:
                    # First level is composed of targets
                    self.__targets = np.asarray(level).transpose()
                    self.__environment_bitmap = np.full((self.__targets.shape[0], self.__targets.shape[1], 3), 0)
                else:
                    if self.__collision_detection[level_founded - 1]:
                        env.append(level.transpose())
                    else:
                        no_collision.append(level.transpose())

                level_founded += 1
                self.__environment_bitmap[level.transpose() == 1, :] = self.__environment_bitmap[level.transpose() == 1,
                                                                       :] + getColour(i)

        if not env:
            env = np.zeros((1, self.__targets.shape[0], self.__targets.shape[1]))
        else:
            env = np.asarray(env)

        if not no_collision:
            self.__no_collision = np.zeros((1, self.__targets.shape[0], self.__targets.shape[1]))
        else:
            self.__no_collision = np.asarray(no_collision)

        self.__collision = np.sum(env, axis=0)
        self.__collision[self.__collision > 0] = 1

    def __init_drones_parameters(self):
        # A drone is not positioned if its position is equal to -1
        self.__drones_position_float = None  # It contains the not approximated position. Not used for drawing
        self.__drones_position = np.full((self.__amount_of_drones, 2), -1)
        self.__drones_velocity = np.zeros((self.__amount_of_drones, 2))
        self.__drawn_drones = np.zeros((self.__amount_of_drones, self.__targets.shape[0], self.__targets.shape[1]))

        self.__stigmergy_space = np.zeros((self.__stigmation_evaporation_speed.shape[0], self.__targets.shape[0],
                                           self.__targets.shape[1]))  # TODO: Non viene utilizzato per ora
        self.__stigmergy_colours = np.array([])  # TODO: Non viene utilizzato per ora

        # The batch dimension will be added only for stigmergy_space and drones' attributes, because
        # they are the only two levels that can change between different parallel simulations.
        self.__stigmergy_space = self.__add_batch_dimension(self.__stigmergy_space)
        self.__drones_velocity = self.__add_batch_dimension(self.__drones_velocity)
        self.__drones_position = self.__add_batch_dimension(self.__drones_position)
        self.__drawn_drones = self.__add_batch_dimension(self.__drawn_drones)

    def __add_batch_dimension(self, matrix):
        matrix = matrix[np.newaxis, ...]
        matrix = np.repeat(matrix, self.__batch_size, axis=0)
        return matrix

    def __init_drones(self):
        for batchIndex in range(self.__batch_size):
            droneIndex = 0
            while droneIndex < len(self.__drones_position[batchIndex]):
                drone_level = np.zeros((self.__targets.shape[0], self.__targets.shape[1]))
                self.__drones_position[batchIndex][droneIndex] = np.asarray([random.randint(0, self.__targets.shape[0] - 1),
                                                                             random.randint(0, self.__targets.shape[1] - 1)])

                drone_level[self.__drones_position[batchIndex][droneIndex][0] - 1: self.__drones_position[batchIndex][droneIndex][0] + 2,
                            self.__drones_position[batchIndex][droneIndex][1] - 1: self.__drones_position[batchIndex][droneIndex][1] + 2] = 1

                self.__drawn_drones[batchIndex][droneIndex] = drone_level

                # A drone is correctly positioned if it's rendered completely inside the map and
                # it doesn't collides with environment or other drones
                if not self.__detect_collision(batchIndex) and not self.__out_of_map(batchIndex, droneIndex):
                    droneIndex += 1

        self.__drones_position_float = np.copy(self.__drones_position).astype(float)

    def __detect_collision(self, batchIndex):
        collision_level = self.__collision[np.newaxis, ...]
        collision_detection = np.append(self.__drawn_drones[batchIndex], collision_level, axis=0)
        collision_detection = np.sum(collision_detection, axis=0)

        if np.any(collision_detection > 1):
            return True

        return False

    def __out_of_map(self, batchIndex, droneIndex):
        # The drone for now it's a simple 3x3 square (this explains the +2 margin used in the check)

        if (self.__drones_position[batchIndex][droneIndex][0] - 1 < 0 or
                self.__drones_position[batchIndex][droneIndex][0] + 2 > self.__targets.shape[0]):
            return True

        if (self.__drones_position[batchIndex][droneIndex][1] - 1 < 0 or
                self.__drones_position[batchIndex][droneIndex][1] + 2 > self.__targets.shape[0]):
            return True

        return False

    def __init_render_parameters(self):
        self.__image = np.full((self.__targets.shape[0], self.__targets.shape[1], 3), 0)
        self.__image_semaphore = None

        if self.__batch_size == 1 and self.__render_allowed:
            self.__image_semaphore = threading.Lock()
            rendering = threading.Thread(target=self.__init_render)
            rendering.start()

    def __init_render(self):
        self.__image_semaphore.acquire()
        app = QtGui.QApplication([])

        # Create window with GraphicsView widget
        w = pg.GraphicsView()
        w.show()
        w.showMaximized()  #w.resize(self.__targets.shape[0], self.__targets.shape[1])
        w.setWindowTitle('Drone simulator')

        view = pg.ViewBox()
        view.invertY()
        w.setCentralItem(view)

        # Lock the aspect ratio
        view.setAspectLocked(True)

        # Create image item
        img = pg.ImageItem(self.__image)
        self.__image_semaphore.release()
        view.addItem(img)

        # Start Qt event loop unless running in interactive mode or using pyside.
        QtGui.QApplication.instance().exec_()





    def render(self):
        if self.__batch_size > 1 and self.__render_allowed:
            raise Exception("Render is allowed only when batch_size is equal to 1")

        drones = np.sum(self.__drawn_drones[0], axis=0)
        self.__environment_bitmap[drones == 1, :] = self.__environment_bitmap[drones == 1, :] + self.__drone_colour
        self.__image_semaphore.acquire()
        np.copyto(self.__image, self.__environment_bitmap)
        self.__image_semaphore.release()

    '''
    def step(self, actions):
        self.__drones_velocity = actions[:, :, 0:2] * self.__inertia + (1 - self.__inertia) * self.__drones_velocity
        # with t = 1
        self.__drones_position_float = self.__drones_position_float + self.__drones_velocity
        self.__drones_position = np.copy(self.__drones_position_float).astype(int)
        self.__render()
        observations = []
        for index, drones in enumerate(self.__drawn_drones):
            observations.append(self.__get_observation(drones, index))

    def __get_observation(self, drones, batchIndex):
        drones_observation = []
        for droneIndex in range(len(drones)):
            # all drones except the one we are working with
            other_drones = [drone for index, drone in enumerate(drones) if index != droneIndex]
            drone_observation = np.array(other_drones)
            drone_position = self.__drones_position[batchIndex][droneIndex]
            drone_observation = drone_observation[:,
                                drone_position[0] - self.__observation_range:
                                drone_position[0] + self.__observation_range + 1,
                                drone_position[1] - self.__observation_range:
                                drone_position[1] + self.__observation_range + 1
                                ]
            print(drone_position)
            print(drone_observation)

        print("stop")
        exit()

    '''