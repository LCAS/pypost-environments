import unittest

from pypost.sampler import EpisodeWithStepsSampler, StepBasedEpisodeTerminationSampler
from pypost.initialSampler import InitialStateSamplerStandard
from pypost.dynamicalSystem.QuadLink import QuadLink
from tests.DummyActionAndReward import DummyActionAndReward


class Test(unittest.TestCase):

    def setUp(self):
        self.sampler = EpisodeWithStepsSampler()
        self.episodeManager = self.sampler.getEpisodeDataManager()
        quad_link = QuadLink(self.episodeManager)
        self.sampler.stepSampler.setIsActiveSampler(StepBasedEpisodeTerminationSampler(
            self.episodeManager, 'steps', 40))

        initialStateSampler = InitialStateSamplerStandard(self.episodeManager)
        dummyActionAndReward = DummyActionAndReward(self.episodeManager.subDataManager, 4, True)

        self.sampler.setTransitionFunction(quad_link.getExpectedNextState)
        self.sampler.setInitStateSampler(initialStateSampler.sampleInitState)
        self.sampler.setActionPolicy(dummyActionAndReward.sampleAction)
        self.sampler.setRewardFunction(dummyActionAndReward.sampleReward)
        self.sampler.setReturnFunction(dummyActionAndReward.sampleReward)

    def testGenerating(self):
        data = self.episodeManager.getDataObject(10)
        self.sampler.numSamples = 100
        self.sampler.setParallelSampling(True)
        data[...] >> self.sampler
        t1 = data[:, 1].states
        t2 = data[1, :].states
        self.assertEqual(t1.shape, (100, 8))
        self.assertEqual(t2.shape, (40, 8))
