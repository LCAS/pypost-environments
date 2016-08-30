import unittest

from pypost.sampler.EpisodeWithStepsSampler import EpisodeWithStepsSampler
from pypost.sampler.initialSampler.InitialStateSamplerStandard import InitialStateSamplerStandard
from pypost.sampler.isActiveSampler.IsActiveNumSteps import IsActiveNumSteps

from pypost.dynamicalSystem.DoubleLink import DoubleLink
from tests.DummyActionAndReward import DummyActionAndReward


class Test(unittest.TestCase):

    def setUp(self):
        self.sampler = EpisodeWithStepsSampler()
        self.episodeManager = self.sampler.getEpisodeDataManager()
        double_link = DoubleLink(self.episodeManager)
        self.sampler.stepSampler.setIsActiveSampler(IsActiveNumSteps(self.episodeManager, 'steps', 40))

        initialStateSampler = InitialStateSamplerStandard(self.episodeManager)
        dummyActionAndReward = DummyActionAndReward(self.episodeManager.subDataManager, 2, True)

        self.sampler.setTransitionFunction(double_link.getExpectedNextState)
        self.sampler.setInitStateSampler(initialStateSampler.sampleInitState)
        self.sampler.setActionPolicy(dummyActionAndReward.sampleAction)
        self.sampler.setRewardFunction(dummyActionAndReward.sampleReward)
        self.sampler.setReturnFunction(dummyActionAndReward.sampleReward)

    def testGenerating(self):
        data = self.episodeManager.getDataObject(10)
        self.sampler.numSamples = 100
        self.sampler.setParallelSampling(True)
        self.sampler.createSamples(data)
        self.assertEqual(data[:, 1].states.shape, (100, 4))
        self.assertEqual(data[1, :].states.shape, (40, 4))


