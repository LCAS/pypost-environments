import numpy as np
import matplotlib.pyplot as plt
import pypost.common.SettingsManager as SettingsManager
from pypost.sampler.EpisodeWithStepsSampler import EpisodeWithStepsSampler
from pypost.sampler.initialSampler.InitialStateSamplerStandard import InitialStateSamplerStandard
from pypost.sampler.isActiveSampler.IsActiveNumSteps import IsActiveNumSteps
from pypostEnvironments.preprocessor.PlanarKinematicsImagePreprocessor import PlanarKinematicsImagePreprocessor as ImgPreprocessor
from pypostEnvironments.dynamicalSystem.Pendulum import Pendulum


# pendulum
defaultSettings = SettingsManager.getDefaultSettings()
defaultSettings.setProperty('noiseStd', 1.0)
defaultSettings.setProperty('initialStateDistributionMinRange', np.asarray([np.pi - np.pi, -2]))
defaultSettings.setProperty('initialStateDistributionMaxRange', np.asarray([np.pi + np.pi,  2]))
defaultSettings.setProperty('initialStateDistributionType', 'Uniform')
defaultSettings.setProperty('dt', 0.025)
defaultSettings.setProperty('initSigmaActions', 1.0)
defaultSettings.setProperty('initialStateDistributionMinRange', np.asarray([np.pi - np.pi, -2]))
defaultSettings.setProperty('initialStateDistributionMaxRange', np.asarray([np.pi + np.pi,  2]))

sampler = EpisodeWithStepsSampler()
pendulum = Pendulum(sampler)

img_size = 48

dataManager = sampler.getEpisodeDataManager()
stepManager = dataManager.subDataManager
stepSampler = sampler.stepSampler
stepSampler.setIsActiveSampler(IsActiveNumSteps(dataManager, 'steps', 40))

img_pre = ImgPreprocessor(stepManager)

initialStateSampler = InitialStateSamplerStandard(sampler)
actionCost = 0.001
stateCost = np.asarray([[10, 0], [0, 0]])

sampler.setTransitionFunction(pendulum)
sampler.setInitialStateSampler(initialStateSampler)
sampler.setActionPolicy(pendulum)
sampler.setRewardFunction(pendulum)
sampler.setReturnFunction(pendulum)

sampler.finalizeSampler(True)
data = dataManager.getDataObject(10)
sampler.numSamples = 100
sampler.setParallelSampling(True)
sampler.createSamples(data)
img_pre.preprocessData(data)
print('done - generating')

f_images_0 = data.getDataEntry('f_images', 0)
f_images_1 = data.getDataEntry('f_images', 1)
f_images_2 = data.getDataEntry('f_images', 2)

n = 20
plt.figure()
for i in range(0, n):
    # display original
    ax = plt.subplot(3, n, i +1)
    plt.imshow(np.reshape(f_images_0[i], (img_size, img_size)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(np.reshape(f_images_1[i], (img_size, img_size)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i+1 + 2 * n)
    plt.imshow(np.reshape(f_images_2[i], (img_size, img_size)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()