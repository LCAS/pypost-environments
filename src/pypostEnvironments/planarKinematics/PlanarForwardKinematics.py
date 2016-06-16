from pypost.data.DataManipulator import DataManipulator
import numpy as np

class PlanarForwardKinematics(DataManipulator):

    def __init__(self, dataManager, dimensions):
        super().__init__(dataManager)
        self.lengths = np.ones((1, dimensions))
        self.numJoints = dimensions
        self.offSet = np.asarray([0, 0])


    #def registerPlanarKinematicsFunctions(self):
      #  self.addDataManipulationFunction(self.getForwardKinematics, self.getN)



    def getForwardKinematics(self, theta, numLink=None):
        if numLink == None:
            numLink = self.numJoints

        y = np.zeros(np.shape(theta)[0], 2)
        for i in range(0, numLink):
            y += self._fuu(theta, i)
        return  y + self.offSet

    def getTaskSpaceVelocity(self, jointPositions, jointVelocities):
        taskSpaceVelocity = np.zeros(np.shape(jointVelocities)[0], 2)
        for i in range(0, jointVelocities):
            J, _ = self.getJacobian(jointPositions[i, :])
            taskSpaceVelocity[i,:] = np.matmul(J, np.transpose(jointVelocities[i, :]))

    # Todo check what is going on here, cleaner code
    def getJacobian(self, theta, numLink=None):
        if numLink == None:
            numLink = self.numJoints

        si = self.getForwardKinematics(theta, numLink)
        J = np.zeros(2, self.numJoints)

        for j in range(0, numLink-1):
            pj = [0, 0]
            for i in range(0, j):
                pj += self._fuu(theta, i)
            pj = -(si - pj)
            J[0:1, j+1] = np.asarray([-pj(1), pj(0)])
        return J, si

    #Todo check an rename
    def _fuu(self, theta, i):
        accumulatedTheta = np.sum(theta[0:i])
        return np.asarray([np.sin(accumulatedTheta), np.cos(accumulatedTheta)]) * self.lengths[i]