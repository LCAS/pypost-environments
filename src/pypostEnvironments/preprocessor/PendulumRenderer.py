import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from pypost.data.Data import Data
from pypost.data.DataManager import DataManager
from pypostEnvironments.preprocessor.Preprocessor import Preprocessor



# create image that is twice as large desired image, scales down for aliasing
class PendulumImageGenerator(Preprocessor):

    def __init__(self, img_size, line_width=-1, encoding='F'):

        self.encoding = encoding
        self.scale = 2
        self.img_size_actual = img_size
        self.img_size_internal = self.img_size_actual * self.scale
        self.x0 = self.y0 = self.img_size_internal / 2
        self.length = self.img_size_internal / 2 - self.img_size_internal / 10
        if line_width == -1:
            self.line_width = self.img_size_internal // 15
        else:
            self.line_width = line_width

    def _generateLine(self, x1, y1):
        img = Image.new(self.encoding, (self.img_size_internal, self.img_size_internal), 0)
        draw = ImageDraw.Draw(img)
        draw.line([(self.x0, self.y0), (x1, y1)], fill=1.0, width=self.line_width, )
        img = img.resize((self.img_size_internal // self.scale, self.img_size_internal // self.scale), resample=Image.ANTIALIAS)
        return np.asarray(img)

    def generateImageFromState(self, state):
        pos = state[0]
        x_end = np.sin(pos) * self.length + self.x0
        y_end = np.cos(pos) * self.length + self.y0
        return self._generateLine(x_end, y_end)

    def generateFlattenedImageFromState(self, state):
        return np.reshape(self.generateImageFromState(state), (self.img_size_actual ** 2))

    def plotImage(self, img):
        reshaped_img = np.reshape(img, (self.img_size_actual, self.img_size_actual))
        plt.imshow(reshaped_img, cmap='gray')

    #todo implement
    def preprocessData(self, data, flat=False):
        if not isinstance(data, Data):
            raise TypeError('argument data needs to be of type pypost.data.Data not ' + str(type(data)))

        states = data.getDataEntry('states')

        numElements = data.getNumElements('states')

        imgManager = DataManager('images')
        imgManager.addDataEntry('rawImages', self.img_size_actual ** 2)

        img_data = imgManager.getDataObject(numElements)


        images = np.zeros((len(states), self.img_size_actual ** 2))
        for i, state in enumerate(states):
            img_data.setDataEntry('rawImages', i, self.generateFlattenedImageFromState(state))

        print('bla')



        return data
