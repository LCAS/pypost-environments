import unittest

import numpy as np
from pypost.common import SettingsManager
from pypost.sampler import EpisodeWithStepsSampler, StepBasedEpisodeTerminationSampler
from pypost.initialSampler import InitialStateSamplerStandard

from pypost.dynamicalSystem.Pendulum import Pendulum
from tests.DummyActionAndReward import DummyActionAndReward


class Test(unittest.TestCase):

    def setUp(self):
        defaultSettings = SettingsManager.getDefaultSettings()
        defaultSettings.setProperty('noiseStd', 1.0)
        defaultSettings.setProperty('initialStateDistributionMinRange', np.asarray([np.pi - np.pi, -2]))
        defaultSettings.setProperty('initialStateDistributionMaxRange', np.asarray([np.pi + np.pi,  2]))
        defaultSettings.setProperty('initialStateDistributionType', 'Uniform')
        defaultSettings.setProperty('dt', 0.025)
        defaultSettings.setProperty('initSigmaActions', 1.0)
        defaultSettings.setProperty('initialStateDistributionMinRange', np.asarray([np.pi - np.pi, -2]))
        defaultSettings.setProperty('initialStateDistributionMaxRange', np.asarray([np.pi + np.pi,  2]))

        self.sampler = EpisodeWithStepsSampler()
        self.episodeManager = self.sampler.getEpisodeDataManager()
        self.stepManager = self.episodeManager.subDataManager
        self.pendulum = Pendulum(self.episodeManager)

        self.sampler.stepSampler.setIsActiveSampler(StepBasedEpisodeTerminationSampler(
            self.episodeManager, 'bla', numTimeSteps=40))

        initialStateSampler = InitialStateSamplerStandard(self.episodeManager)

        dummyActionAndReward = DummyActionAndReward(self.episodeManager.subDataManager, 1)

        self.sampler.setTransitionFunction(self.pendulum.getExpectedNextState)
        self.sampler.setInitStateSampler(initialStateSampler.sampleInitState)
        self.sampler.setActionPolicy(dummyActionAndReward.sampleAction)
        self.sampler.setRewardFunction(dummyActionAndReward.sampleReward)
        self.sampler.setReturnFunction(dummyActionAndReward.sampleReward)


    def testGenerating(self):
        data = self.episodeManager.getDataObject(10)
        self.sampler.numSamples = 100
        self.sampler.setParallelSampling(True)
        data >> self.sampler
        self.assertEqual(data[:, 1].states.shape, (100, 2))
        self.assertEqual(data[1, :].states.shape, (40, 2))


