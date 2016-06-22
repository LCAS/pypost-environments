import doublePendulumForwardModel
import quadPendulumForwardModel
import numpy as np

def simulate_double_pendulum(states, actions, lengths, masses, inertias, g, friction, dt, dst,
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

def simulate_quad_pendulum(states, actions, lengths, masses, inertias, g, friction, dt, dst):
    return quadPendulumForwardModel.simulate(states[0], states[1], states[2], states[3],
                                             states[4], states[5], states[6], states[7],
                                             actions[0], actions[1], actions[2], actions[3],
                                             lengths[0], lengths[1], lengths[2], lengths[3],
                                             masses[0], masses[1], masses[2], masses[3],
                                             inertias[0], inertias[1], inertias[2], inertias[3],
                                             g,
                                             friction[0], friction[1], friction[2], friction[3],
                                             dt, dst)
