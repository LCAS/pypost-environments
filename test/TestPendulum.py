import src.Pendulum as pend
from pypost.sampler.EpisodeWithStepsSampler import  EpisodeWithStepsSampler
from pypost.sampler.isActiveSampler.IsActiveNumSteps import IsActiveNumSteps

sampler = EpisodeWithStepsSampler()
dataManager = sampler.getEpisodeDataManager()
sampler.stepSampler.setIsActiveSampler(IsActiveNumSteps(dataManager, "step", numTimeSteps=60))

environment = pend.Pendulum(sampler)

sampler.setContextSampler(environment)
sampler.setActionPolicy(environment)
sampler.setTransitionFunction(environment)
sampler.setRewardFunction(environment)
sampler.setInitialStateSampler(environment)

newData = dataManager.getDataObject(10)
sampler.numSamples=100
sampler.setParallelSampling(True)
sampler.createSamples(newData)
print('finished')



