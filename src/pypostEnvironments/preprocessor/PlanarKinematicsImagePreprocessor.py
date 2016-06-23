import numpy as np
from PIL import Image
from PIL import ImageDraw
from pypostEnvironments.preprocessor.Preprocessor import Preprocessor
from pypost.data.DataManipulator import DataManipulator

class PlanarKinematicsImagePreprocessor(Preprocessor, DataManipulator):

    def __init__(self, dataManager, nrJoints):

        Preprocessor.__init__(self)
        DataManipulator.__init__(self, dataManager)

        self.nrJoints = nrJoints

        self.imgSize = 48
        self.nrJoints = 1
        self.lineWidth = 5
        self.encoding = 'F' # Floating point gray scale encoding (between 0 and 1). use 'L' for 8 bit int

        self.linkProperty('imgSize')
        self.linkProperty('lineWidth')

        self.renderer = Renderer(self.imgSize, self.nrJoints, self.lineWidth, self.encoding)
        self.dataManager.addDataEntry('f_images', self.imgSize ** 2)
        self.addDataManipulationFunction(self.renderer.generateFlattenedImagesFromStates,
                                         ['states'], ['f_images'], name='createImages')

    def preprocessData(self, data, *args):
        self.callDataFunction('createImages', data)


class Renderer:
    # create image that is twice as large as desired image, scales down for aliasing
    def __init__(self, img_size, nr_joints, line_width=-1, encoding='F'):

        self.encoding = encoding
        self.nr_joints = nr_joints
        self.scale = 2
        self.img_size_actual = img_size
        self.img_size_internal = self.img_size_actual * self.scale
        self.x0 = self.y0 = self.img_size_internal / 2
        self.length = self.img_size_internal / 2
        self.length = self.length // nr_joints
        if line_width == -1:
            self.line_width = self.img_size_internal // 15
        else:
            self.line_width = line_width

    def _generateLines(self, points):
        img = Image.new(self.encoding, (self.img_size_internal, self.img_size_internal), 0)
        draw = ImageDraw.Draw(img)
        for i in range(self.nr_joints):
            draw.line([(points[i, 0], self.img_size_internal - points[i, 1]),
                       (points[i + 1, 0],  self.img_size_internal - points[i + 1, 1])],
                      fill=1.0, width=self.line_width)
        img = img.resize((self.img_size_internal // self.scale, self.img_size_internal // self.scale), resample=Image.ANTIALIAS)
        img_as_array = np.asarray(img)
        # somehow values outside the interval [0,1] are created. Currently just clipped. Map into interval instead?
        if self.encoding == 'F':
            img_as_array = np.clip(img_as_array, 0, 1)
        return img_as_array

    def _generatePointsFromAngles(self, angles):
        points = np.zeros((self.nr_joints + 1 , 2))
        points[0, 0] = self.x0
        points[0, 1] = self.y0
        for i in range(1, self.nr_joints + 1):
            points[i, 0] = points[i - 1, 0] + np.sin(angles[i - 1]) * self.length
            points[i, 1] = points[i - 1, 1] + np.cos(angles[i - 1]) * self.length
        return points

    def generateImageFromState(self, state):
        angles = np.zeros(self.nr_joints)
        for i in range(0, self.nr_joints):
            angles[i] = state[2 * i]
        points = self._generatePointsFromAngles(angles)
        return self._generateLines(points)

    def generateImagesFromStates(self, states):
        images = np.zeros((len(states), self.img_size_actual, self.img_size_actual))
        for i, state in enumerate(states):
            images[i] = self.generateImageFromState(state)
        return images

    def generateFlattenedImagesFromStates(self, states):
        return np.reshape(self.generateImagesFromStates(states), (len(states), -1))
