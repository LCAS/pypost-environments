from pypost.sampler.EpisodeWithStepsSampler import  EpisodeWithStepsSampler
from pypost.sampler.isActiveSampler.IsActiveNumSteps import IsActiveNumSteps
from src.MountainCar import MountainCar

sampler = EpisodeWithStepsSampler()
dataManager = sampler.getEpisodeDataManager()
sampler.stepSampler.setIsActiveSampler(IsActiveNumSteps(dataManager, "step", numTimeSteps=60))

environment = MountainCar(sampler)

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
