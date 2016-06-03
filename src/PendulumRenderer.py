from src.SimplePendulum import Pendulum
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw


# create image that is twice as large desired image, scales down for aliasing
# Todo create 'preprocessor' interface and connection to toolbox
class PendulumImageGenerator:

    def __init__(self, img_size, line_width=-1):

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
        #'F' is float gray scale (between 0 and 1) , Todo: make setable
        img = Image.new('F', (self.img_size_internal, self.img_size_internal), 0)
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

