import numpy as np
from pypost.sampler.EpisodeWithStepsSampler import EpisodeWithStepsSampler
from pypost.sampler.initialSampler.InitialStateSamplerStandard import InitialStateSamplerStandard
from pypost.sampler.isActiveSampler.IsActiveNumSteps import IsActiveNumSteps

from src.pypostEnvironments.dynamicalSystem.DoubleLink import DoubleLink

#defaultSettings = SettingsManager.getDefaultSettings()
#defaultSettings.setProperty('noiseStd', 1.0)
#defaultSettings.setProperty('initialStateDistributionMinRange', np.asarray([np.pi - np.pi, -2]))
#defaultSettings.setProperty('initialStateDistributionMaxRange', np.asarray([np.pi + np.pi,  2]))
#defaultSettings.setProperty('initialStateDistributionType', 'Uniform')
#defaultSettings.setProperty('dt', 0.025)
#defaultSettings.setProperty('initSigmaActions', 1.0)
#defaultSettings.setProperty('initialStateDistributionMinRange', np.asarray([np.pi - np.pi, -2]))
#defaultSettings.setProperty('initialStateDistributionMaxRange', np.asarray([np.pi + np.pi,  2]))

sampler = EpisodeWithStepsSampler()
double_link = DoubleLink(sampler)


dataManager = sampler.getEpisodeDataManager()
stepSampler = sampler.stepSampler
stepSampler.setIsActiveSampler(IsActiveNumSteps(dataManager, 'steps', 40))

initialStateSampler = InitialStateSamplerStandard(sampler)
actionCost = 0.001
stateCost = np.asarray([[10, 0], [0, 0]])

sampler.setTransitionFunction(double_link)
sampler.setInitialStateSampler(initialStateSampler)
sampler.setActionPolicy(double_link)
sampler.setRewardFunction(double_link)
sampler.setReturnFunction(double_link)

sampler.finalizeSampler(True)
data = dataManager.getDataObject(10)
sampler.numSamples = 100
sampler.setParallelSampling(True)
sampler.createSamples(data)
print('done - generating')


