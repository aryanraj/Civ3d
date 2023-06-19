MAX_LOADCASE = 501
MAX_LOAD_DISTANCE = 50
N_MODES = 4

import os
os.environ["MAX_DOF"] = str(30)
os.environ["MAX_LOADCASE"] = str(MAX_LOADCASE)

from Models import DOFClass, Node, BeamSection, Beam
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,MAX_LOAD_DISTANCE, MAX_LOADCASE)

n0 = Node([ 0.0,0,0], [1, 1, 1, 1, 0, 1])
n2 = Node([ 5.0,0,0], [1, 1, 0, 1, 0, 1])
n4 = Node([10.0,0,0], [1, 1, 1, 1, 0, 1])

section = BeamSection()
b1 = Beam([n0, n2, n4], section)

Beam.addMovingPointLoadsToBeamList([b1], 2, [-1], [0], MAX_LOAD_DISTANCE/(MAX_LOADCASE-1), list(range(MAX_LOADCASE)))
staticDisplacementVector = DOFClass.analyse()
midSpanMaxDisplacement = np.min(staticDisplacementVector[n2.DOF[2].id,:])
print(f"Maximum displacement @ midspan for static case is {midSpanMaxDisplacement}")
plt.plot(x, staticDisplacementVector[n2.DOF[2].id,:], label="static")

eigenValues, eigenVectors, effectiveMass, massParticipationFactor = DOFClass.eig(N_MODES)
dampingRatios = np.ones((N_MODES,), np.float64) * 0.02

def performTimeHistoryAnalysis(speed_kmph):
  speed_mps = speed_kmph/3.6/350
  dt = MAX_LOAD_DISTANCE/speed_mps/(MAX_LOADCASE-1)
  timeHistoryDisplacementVector, timeHistoryVelocityVector, timeHistoryAccelerationVector = DOFClass.analyseTimeHistoryNewmark(dt, eigenValues, eigenVectors, dampingRatios)
  midSpanMaxDisplacement = np.min(timeHistoryDisplacementVector[n2.DOF[2].id,:])
  print(f"Maximum displacement @ midspan for {speed_kmph}kmph speed case is {midSpanMaxDisplacement}")
  plt.plot(x, timeHistoryDisplacementVector[n2.DOF[2].id,:], label=f"{speed_kmph} kmph")

for id, distance in enumerate(np.linspace(0,MAX_LOAD_DISTANCE,MAX_LOADCASE).tolist()):
    b1.addPointLoad(2, -1, distance, [id])
performTimeHistoryAnalysis(0.0001)
performTimeHistoryAnalysis(0.001)
performTimeHistoryAnalysis(0.01)
performTimeHistoryAnalysis(20)
performTimeHistoryAnalysis(200)
performTimeHistoryAnalysis(600)
performTimeHistoryAnalysis(2000)
plt.legend()
plt.show()