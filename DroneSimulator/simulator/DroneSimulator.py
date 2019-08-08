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
    colour[2 - i//8] = 2**(i % 8)
    return colour


class DroneSimulator:

    def __init__(self, bitmap, batch_size, observation_range, amount_of_drones, stigmation_evaporation_speed, inertia,
                 collision_detection, reward_function, max_steps, render=False, drone_colour=[255, 255, 255]):

        self.__batch_size = batch_size
        self.__observation_range = observation_range
        self.__amount_of_drones = amount_of_drones
        self.__stigmation_evaporation_speed = stigmation_evaporation_speed
        self.__inertia = inertia
        self.__reward_function = reward_function   # DA AGGIUNGERE REWARD FUNCTION
        self.__max_steps = max_steps
        self.__drone_colour = drone_colour

        # __environment_bitmap contains the initial bitmap, with all the fixed information (ground, targets, obstacles);
        # drones and stigmergy space will be added in the render() method
        self.__environment_bitmap = None

        self.__targets = np.array([])
        self.__collision = np.array([])      # "Schiaccia" tutti gli ostacoli su un solo livello bidimensionale?
        self.__no_collision = np.array([])   # Variabile non usata in nessun metodo, a cosa serve?
        self.__image = np.array([])
        self.__image_semaphore = None

        self.__parse_bitmap(bitmap, collision_detection)

        # A drone is not positioned if its position is equal to -1
        # drones_position_float contains the true position of drones. Not used for drawing.
        self.__drones_position_float = None
        self.__drones_position = np.full((amount_of_drones, 2), -1)
        self.__drones_velocity = np.zeros((amount_of_drones, 2))
        self.__drawn_drones = np.zeros((amount_of_drones, self.__targets.shape[0], self.__targets.shape[1]))

        self.__stigmergy_space = np.zeros((
                self.__stigmation_evaporation_speed.shape[0],
                self.__targets.shape[0],
                self.__targets.shape[1]
            ))
        self.__stigmergy_colours = np.array([])  # Non viene utilizzato per ora

        # Only for stigmergy_space and drones the batch dimension will be added
        # because is the only 2 levels that can change beetween batch the other
        # will be fixed
        self.__stigmergy_space = self.__add_batch_dimension(self.__stigmergy_space)
        self.__drones_velocity = self.__add_batch_dimension(self.__drones_velocity)
        self.__drones_position = self.__add_batch_dimension(self.__drones_position)
        self.__drawn_drones = self.__add_batch_dimension(self.__drawn_drones)
        self.__init_drones()

        if batch_size == 1 and render:
            self.__image_semaphore = threading.Lock()
            rendering = threading.Thread(target=self.__init_render)
            rendering.start()

    def __parse_bitmap(self, bitmap, collision_detection):
        input_array = np.asarray(Image.open(bitmap))
        rgb_bit_array = np.unpackbits(input_array, axis=2)
        # rgb_bit_array is a matrix of pixels, where each cell (each pixel) is a 24-bit array

        env = []
        no_collision = []
        level_founded = 0

        for i in range(0, 24):
            level = rgb_bit_array[:, :, i]
            if np.any(level):             # Only levels with at least 1 item are inserted in the environment
                if level_founded == 0:    # First level is composed of targets
                    self.__targets = np.asarray(level).transpose() # The transpose() is needed by PyQtGraph to draw the map properly
                    self.__environment_bitmap = np.full((self.__targets.shape[0], self.__targets.shape[1], 3), 0)

                else:
                    if collision_detection[level_founded - 1]:
                        env.append(level.transpose())
                    else:
                        no_collision.append(level.transpose())

                level_founded += 1
                self.__environment_bitmap[level.transpose() == 1, :] = self.__environment_bitmap[level.transpose() == 1, :] + getColour(i)

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

        self.__image = np.full((self.__targets.shape[0], self.__targets.shape[1], 3), 0)

    def __add_batch_dimension(self, matrix):
        matrix = matrix[np.newaxis, ...]
        matrix = np.repeat(matrix, self.__batch_size, axis=0)
        return matrix

    def __init_drones(self):
        for batchIndex in range(self.__batch_size):
            droneIndex = 0
            while droneIndex < len(self.__drones_position[batchIndex]):
                self.__drones_position[batchIndex][droneIndex] =  np.asarray([
                    random.randint(0, self.__targets.shape[0] - 1),
                    random.randint(0, self.__targets.shape[1] - 1)
                ])
                self.__render(batchIndex)

                # A drone is correctly positioned if it was rendered in a level and it doesn't collides with environment
                # or other drones. A drone is not rendered if it leaves the map.
                if np.any(self.__drones_position[batchIndex][droneIndex]) and not self.__detect_collision(batchIndex):
                    droneIndex += 1

        self.__drones_position_float = np.copy(self.__drones_position).astype(float)

    def __render(self, batchIndex = None):
        dronePositionVelocity = np.concatenate((self.__drones_position, self.__drones_velocity), 2)

        if batchIndex is None:
            for i in range(self.__batch_size):
                self.__drawn_drones[i] = self.__render_batch(dronePositionVelocity[i])
        else:
            self.__drawn_drones[batchIndex] = self.__render_batch(dronePositionVelocity[batchIndex])

    def __render_batch(self, batch_drone_level):  # DA RIVEDERE
        return np.apply_along_axis(self.__draw_drone, 1, batch_drone_level)

    # DA RIVEDERE draw_drone(), viene chiamata troppe volte (forse inutilmente)
    def __draw_drone(self, positionVelocity):
        level = np.zeros((self.__targets.shape[0], self.__targets.shape[1]))

        # This is for not positioned drones
        if positionVelocity[0] < 0 or positionVelocity[1] < 0:
            return level

        # Velocity is provided for future better representation of drone
        # If a drone will go outside of the map, it will be considered as fallen to the ground
        # The drone for now it's a simple 3x3 square (this explains the +2 margin used in the check)
        if positionVelocity[0] - 1 < 0 or positionVelocity[0] + 2 > self.__targets.shape[0]:
            return level

        if positionVelocity[1] - 1 < 0 or positionVelocity[1] + 2 > self.__targets.shape[1]:
            return level

        level[int(positionVelocity[0]) - 1 : int(positionVelocity[0]) + 2,
              int(positionVelocity[1]) - 1 : int(positionVelocity[1]) + 2] = 1
        return level

    def __detect_collision(self, batchIndex):
        # tmp is useful to test collisions (drone-drone and drone-obstacle collisions)
        tmp = self.__collision[np.newaxis, ...]

        # Detection of drones that are gone out of the map
        for i in range(self.__drawn_drones.shape[1]):
            if not np.any(self.__drawn_drones[batchIndex][i]) and not np.array_equal(self.__drones_position[batchIndex][i], [-1, -1]):
                return True

        tmp = np.append(self.__drawn_drones[batchIndex], tmp, axis = 0)
        tmp = np.sum(tmp, axis=0)
        if np.any(tmp > 1):
            return True

        return False

    def __init_render(self):
        self.__image_semaphore.acquire()
        app = QtGui.QApplication([])

        # Create window with GraphicsView widget
        w = pg.GraphicsView()
        w.show()
        w.resize(self.__targets.shape[0], self.__targets.shape[1])
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

    def render(self, render):
        if self.__batch_size > 1 and render:
            raise Exception("render is allowed only with batch_size equal to 1")

        drones = np.sum(self.__drawn_drones[0], axis=0)
        self.__environment_bitmap[drones == 1, :] = self.__environment_bitmap[drones == 1, :] + self.__drone_colour
        self.__image_semaphore.acquire()
        np.copyto(self.__image, self.__environment_bitmap)
        self.__image_semaphore.release()

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