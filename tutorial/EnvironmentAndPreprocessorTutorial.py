import numpy as np
import matplotlib.pyplot as plt
import pypost.common.SettingsManager as SettingsManager
from pypost.sampler.EpisodeWithStepsSampler import EpisodeWithStepsSampler
from pypost.sampler.initialSampler.InitialStateSamplerStandard import InitialStateSamplerStandard
from pypost.sampler.isActiveSampler.IsActiveNumSteps import IsActiveNumSteps
from pypostEnvironments.preprocessor.PlanarKinematicsImagePreprocessor import PlanarKinematicsImagePreprocessor as ImgPreprocessor
from pypostEnvironments.dynamicalSystem.Pendulum import Pendulum
from pypostEnvironments.dynamicalSystem.DoubleLink import DoubleLink
from pypostEnvironments.dynamicalSystem.QuadLink import QuadLink
from test.DummyActionAndReward import DummyActionAndReward
# Tutorial: How to initialize an environment with a preprocessor:

# First we get the default settings Object from the Settings Manager
defaultSettings = SettingsManager.getDefaultSettings()
# Afterwards we can set those settings. First for the pendulum itself...
defaultSettings.setProperty('noiseStd', 1e-10)
defaultSettings.setProperty('dt', 0.025)
defaultSettings.setProperty('initSigmaActions', 1e-10)
# ... to sample the initial states ...
defaultSettings.setProperty('initialStateDistributionType', 'Uniform')
#defaultSettings.setProperty('initialStateDistributionMinRange', np.asarray([np.pi - np.pi, -2]))
#defaultSettings.setProperty('initialStateDistributionMaxRange', np.asarray([np.pi + np.pi,  2]))
# ... and for the image creating preprocessor. Note that the same settings object is used here which is then
# distributed by the setting manager to all its clients. (The Pendulum, the InitalStateSampler as well as the
# Preprocessor going to be setting clients

img_size = 48 # pixels, height and width. (currently only squared images possible)
defaultSettings.setProperty('imgSize', img_size)
defaultSettings.setProperty('lineWidth', 3)  # again pixel

# Next we can create a "Sampler". It will later sample the values from the desired environment...
sampler = EpisodeWithStepsSampler()
# ... which we create next!
# This will load the settings specified above from the setting manager and add all needed entries to the samplers
# data manager.
pendulum = DoubleLink(sampler)
initialStateSampler = InitialStateSamplerStandard(sampler)

# Next we can get the Data Manager and its sub manager from the sampler. We will need them later
episodeManager = sampler.getEpisodeDataManager()
stepManager = episodeManager.subDataManager

# Dummy Action and Reward functions (return always 0):
action_and_reward = DummyActionAndReward(stepManager, 2)

# The isActiveSampler specifies how long the step sampler should sample each epoch.
#  For this tutorial we just use a fixed number.
steps_per_epoch = 100
stepSampler = sampler.stepSampler
stepSampler.setIsActiveSampler(IsActiveNumSteps(episodeManager, 'steps', steps_per_epoch))

# Afterwards we tell the sampler where to find the functions it should sample from
sampler.setTransitionFunction(pendulum)
sampler.setInitialStateSampler(initialStateSampler)
# The next three are currently just dummies (all returning just zeros) and can be replaced with your own functions
sampler.setActionPolicy(action_and_reward)
sampler.setRewardFunction(action_and_reward)
sampler.setReturnFunction(action_and_reward)

# We initialize the preprocessor. It will add an entry to hold the (flattened) images to the step manager.
number_of_joints = 2  # one joint since we use the pendulum
img_pre = ImgPreprocessor(stepManager, number_of_joints)

# Finally we are going to sample the data. In order to do this we first need to get a data object of the
# desired size from the data manager. In this case the "size" is the number of epochs we want to sample.
nr_of_epochs = 10
data = episodeManager.getDataObject(nr_of_epochs)

# We change the number once more, because we can...
nr_of_epochs = 100
sampler.numSamples = nr_of_epochs
# ... and tell the sampler that there are no dependencies between the episodes and sampling can be performed parallel
# (Note that the Steps of course depend on each other and hence can not be sampled parallel)
sampler.setParallelSampling(True)

# We first sample the data ...
sampler.createSamples(data)
# ... and then preprocess them
img_pre.preprocessData(data)

print('done - generating')

# We get the first three sampled episodes and plot the first 20 images from each
f_images_0 = data.getDataEntry('f_images', 0)
f_images_1 = data.getDataEntry('f_images', 1)
f_images_2 = data.getDataEntry('f_images', 2)

n = 20
plt.figure(figsize=(20,4))
for i in range(0, n):

    ax = plt.subplot(3, n, i +1)
    plt.imshow(np.reshape(f_images_0[i], (img_size, img_size)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

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