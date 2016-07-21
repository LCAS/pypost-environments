import numpy as np
from pypost.sampler.EpisodeWithStepsSampler import EpisodeWithStepsSampler
from pypost.sampler.initialSampler.InitialStateSamplerStandard import InitialStateSamplerStandard
from pypost.sampler.isActiveSampler.IsActiveNumSteps import IsActiveNumSteps
from pypostEnvironments.dynamicalSystem.QuadLink import QuadLink
from test.DummyActionAndReward import DummyActionAndReward

sampler = EpisodeWithStepsSampler()
quad_link = QuadLink(sampler)


dataManager = sampler.getEpisodeDataManager()
stepSampler = sampler.stepSampler
stepSampler.setIsActiveSampler(IsActiveNumSteps(dataManager, 'steps', 1000))

initialStateSampler = InitialStateSamplerStandard(sampler)

dummyActionAndReward = DummyActionAndReward(dataManager, 4, generateActions=True)

sampler.setTransitionFunction(quad_link)
sampler.setInitialStateSampler(initialStateSampler)
sampler.setActionPolicy(dummyActionAndReward)
sampler.setRewardFunction(dummyActionAndReward)
sampler.setReturnFunction(dummyActionAndReward)

sampler.finalizeSampler(True)
data = dataManager.getDataObject(10)
sampler.numSamples = 1000
sampler.setParallelSampling(True)
state = np.random.uniform(-1, 1, 8)
action = np.random.uniform(-1, 1, 4)
#quad_link.getLinearizedContinuousTimeDynamics(state, action)
sampler.createSamples(data)
print('done - generating')