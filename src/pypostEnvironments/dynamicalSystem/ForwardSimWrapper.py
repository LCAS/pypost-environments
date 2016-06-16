import doublePendulumForwardModel
import numpy as np

def simulate_double_pendulum(states ,actions, lengths, masses, inertias, g, friction, dt, dst,
                             use_pd=0, pdSetPoints=np.zeros((4)), pdGain=np.zeros((4))):
    return doublePendulumForwardModel.simulate(states[0], states[1], states[2], states[3],
                                               actions[0], actions[1],
                                               lengths[0], lengths[1],
                                               masses[0], masses[1],
                                               inertias[0], inertias[1],
                                               g,
                                               friction[0], friction[1],
                                               dt,
                                               dst,
                                               use_pd,
                                               pdSetPoints[0], pdSetPoints[1], pdSetPoints[2], pdSetPoints[3],
                                               pdGain[0], pdGain[1], pdGain[2], pdGain[3])
