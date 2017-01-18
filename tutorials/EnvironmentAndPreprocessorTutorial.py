import matplotlib.pyplot as plt
import numpy as np
import pypost.common.SettingsManager as SettingsManager
from pypost.dynamicalSystem import DoubleLink
from pypost.envs.ReturnSummedReward import ReturnSummedReward
from pypost.sampler import EpisodeWithStepsSampler
from pypost.initialSampler import InitialStateSamplerStandard

from pypost.preprocessor import PlanarKinematicsImagePreprocessor as ImgPreprocessor
from tests.DummyActionAndReward import DummyActionAndReward

# Tutorial: How to initialize an environment with a preprocessor:

steps_per_epoch = 100

# First we get the default settings Object from the Settings Manager
defaultSettings = SettingsManager.getDefaultSettings()

# Afterwards we can set those settings. First for the pendulum itself...
defaultSettings.setProperty('noiseStd', 1e-10)
defaultSettings.setProperty('dt', 0.025)
defaultSettings.setProperty('initSigmaActions', 1e-10)

# ... to sample the initial states ...
defaultSettings.setProperty('initialStateDistributionType', 'Uniform')
defaultSettings.setProperty('numTimeSteps', steps_per_epoch)

# ... and for the image creating preprocessor. Note that the same settings object is used here which is then
# distributed by the setting manager to all its clients. (The Pendulum, the InitalStateSampler as well as the
# Preprocessor going to be setting clients
img_size = 48 # pixels, height and width. (currently only squared images possible)
defaultSettings.setProperty('imgSize', img_size)
defaultSettings.setProperty('lineWidth', 3)  # again pixel

# Next we can create a "Sampler". It will later sample the values from the desired environment
sampler = EpisodeWithStepsSampler()

# We also need the data managers for the episode and step layer - we get them from the sampler
episodeManager = sampler.getEpisodeDataManager()

# Finally, we can create our environment. Try exchanging the DoubleLink with the QuadLink or the Pendulum...
n_link_pendulum = DoubleLink(episodeManager)
# ... make sure you change the number of joints too
number_of_joints = 2

# We can still add further settings, e.g. for the initialStateSampler created next
defaultSettings.setProperty('initialStateDistributionMinRange', np.tile(np.asarray([np.pi - np.pi, -2]), [number_of_joints]))
defaultSettings.setProperty('initialStateDistributionMaxRange', np.tile(np.asarray([np.pi + np.pi,  2]), [number_of_joints]))
initialStateSampler = InitialStateSamplerStandard(episodeManager)

# Dummy Action and Reward Functions: Reward is always zero. Actions are sampled random if generateActions=True, else
# all actions are 0
action_and_reward = DummyActionAndReward(episodeManager, number_of_joints, generateActions=True)
returnFunction = ReturnSummedReward(episodeManager)
# The isActiveSampler specifies how long the step sampler should sample each epoch.
#  For this tutorials we just use a fixed number.

# Afterwards we tell the sampler where to find the functions it should sample from
sampler.setTransitionFunction(n_link_pendulum)
sampler.setInitStateSampler(initialStateSampler)
sampler.setActionPolicy(action_and_reward.sampleAction)
# The next two are currently just dummies (all returning just zeros) and can be replaced with your own functions
sampler.setRewardFunction(action_and_reward.sampleReward)
sampler.setReturnFunction(returnFunction)

# We initialize the preprocessor. It will add an entry to hold the (flattened) images to the step manager.
img_pre = ImgPreprocessor(episodeManager, number_of_joints)

# Finally we are going to sample the data. In order to do this we first need to get a data object of the
# desired size from the data manager. In this case the "size" is the number of epochs we want to sample.
nr_of_epochs = 10
data = episodeManager.getDataObject([nr_of_epochs, steps_per_epoch])

# We change the number once more, because we can...
nr_of_epochs = 10
sampler.numSamples = nr_of_epochs
# ... and tell the sampler that there are no dependencies between the episodes and sampling can be performed parallel
# (Note that the Steps of course depend on each other and hence can not be sampled parallel)
sampler.setParallelSampling(True)

# We first sample the data ...
data[...] >> sampler
print('done - generating trajectories')
# ... and then preprocess it
data[...] >> img_pre.preprocessStates
print('done - generating images')

# We get the first three sampled episodes and plot every 5th image from each
f_images_0 = data[0, ...].flatImages
f_images_1 = data[1, ...].flatImages
f_images_2 = data[2, ...].flatImages

n = 20
plt.figure(figsize=(20, 4))
for i in range(0, n):

    ax = plt.subplot(3, n, i +1)
    plt.imshow(np.reshape(f_images_0[i * 5], (img_size, img_size)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(np.reshape(f_images_1[i * 5], (img_size, img_size)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i+1 + 2 * n)
    plt.imshow(np.reshape(f_images_2[i * 5], (img_size, img_size)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()