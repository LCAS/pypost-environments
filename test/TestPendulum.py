import numpy as np
import pypost.common.SettingsManager as SettingsManager
from pypost.sampler.EpisodeWithStepsSampler import EpisodeWithStepsSampler
from pypost.sampler.initialSampler.InitialStateSamplerStandard import InitialStateSamplerStandard
from pypost.sampler.isActiveSampler.IsActiveNumSteps import IsActiveNumSteps

from pypostEnvironments.dynamicalSystem.Pendulum import Pendulum
from test.DummyActionAndReward import DummyActionAndReward

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

dummyActionAndReward = DummyActionAndReward(dataManager.subDataManager, 1, True)

sampler.setTransitionFunction(pendulum)
sampler.setInitialStateSampler(initialStateSampler)
sampler.setActionPolicy(dummyActionAndReward)
sampler.setRewardFunction(dummyActionAndReward)
sampler.setReturnFunction(dummyActionAndReward)

sampler.finalizeSampler(True)
data = dataManager.getDataObject(10)
sampler.numSamples = 100
sampler.setParallelSampling(True)
sampler.createSamples(data)
print('done - generating')



