import numpy as np
import pypost.common.SettingsManager as SettingsManager
from pypost.sampler.EpisodeWithStepsSampler import EpisodeWithStepsSampler
from pypost.sampler.initialSampler.InitialStateSamplerStandard import InitialStateSamplerStandard
from pypost.sampler.isActiveSampler.IsActiveNumSteps import IsActiveNumSteps

from pypostEnvironments.dynamicalSystem.Pendulum import Pendulum

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


dataManager = sampler.getEpisodeDataManager()
stepSampler = sampler.stepSampler
stepSampler.setIsActiveSampler(IsActiveNumSteps(dataManager, 'steps', 40))

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
print('done - generating')
#
#pendulum_img_gen = PendulumRenderer.PendulumImageGenerator(48)
#pendulum_img_gen.preprocessData(data, flat=True)


