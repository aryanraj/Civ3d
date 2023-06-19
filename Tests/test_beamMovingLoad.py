MAX_LOADCASE = 501
MAX_LOAD_DISTANCE = 150
N_MODES = 10

import os
os.environ["MAX_DOF"] = str(13*6)
os.environ["MAX_LOADCASE"] = str(MAX_LOADCASE)

from Models import DOFClass, Node, BeamSection, Beam
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

x = np.linspace(0,MAX_LOAD_DISTANCE, MAX_LOADCASE)

print("** Defining structure **")
n0 = Node([ 0.0,0,0], [1, 1, 1, 1, 0, 1])
n1 = [Node([ __,0,0], [1, 1, 0, 1, 0, 1]) for __ in [1.,2.,3.,4.]]
n2 = Node([ 5.0,0,0], [1, 1, 0, 1, 0, 1])
n3 = [Node([ __,0,0], [1, 1, 0, 1, 0, 1]) for __ in [6.,7.,8.,9.]]
n4 = Node([10.0,0,0], [1, 1, 1, 1, 0, 1])
section = BeamSection.Rectangle(1., 0.25, 205_000_000_000, 0.3, 7850, 7850*9.806)
b1 = Beam([n0, *n1, n2, *n3, n4], section)

print("** Adding Vehicle Load **")
vehicleAxleLoadDistance:list[tuple[float,float]] = [
  (191295, 0.0),
  (191295, 1.9),
  (191295, 3.8),
  (191295, 10.39),
  (191295, 12.29),
  (191295, 14.19),
  (199192.05, 17.923),
  (199192.05, 19.923),
  (199192.05, 25.393),
  (199192.05, 27.393),
  (199388.25, 29.923),
  (199388.25, 31.923),
  (199388.25, 37.393),
  (199388.25, 39.393),
  (199339.2, 41.923),
  (199339.2, 43.923),
  (199339.2, 49.393),
  (199339.2, 51.393),
  (199388.25, 53.923),
  (199388.25, 55.923),
  (199388.25, 61.393),
  (199388.25, 63.393),
  (199314.675, 65.923),
  (199314.675, 67.923),
  (199314.675, 73.393),
  (199314.675, 75.393),
  (199167.525, 77.923),
  (199167.525, 79.923),
  (199167.525, 85.393),
  (199167.525, 87.393),
]

Beam.addMovingPointLoadsToBeamList(
  [b1],
  2,
  [-_[0]/2/9.806 for _ in vehicleAxleLoadDistance],
  [-_[1] for _ in vehicleAxleLoadDistance],
  MAX_LOAD_DISTANCE/(MAX_LOADCASE-1),
  list(range(MAX_LOADCASE)),
  )

print("** Eigenvalue Analysis **")
eigenValues, eigenVectors, effectiveMass, massParticipationFactor = DOFClass.eig(N_MODES)
dampingRatios = np.ones((N_MODES,), np.float64) * 0.02

def performTimeHistoryAnalysis(speed_kmph):
  speed_mps = speed_kmph/3.6
  dt = MAX_LOAD_DISTANCE/speed_mps/(MAX_LOADCASE-1)
  timeHistoryDisplacementVector, timeHistoryVelocityVector, timeHistoryAccelerationVector = DOFClass.analyseTimeHistory(dt, eigenValues, eigenVectors, dampingRatios)
  midSpanMaxDisplacement = np.min(timeHistoryDisplacementVector[n2.DOF[2].id,:])
  newmark, _, _ = DOFClass.analyseTimeHistoryNewmark(dt, eigenValues, eigenVectors, dampingRatios)
  print(f"Maximum displacement @ midspan for {speed_kmph}kmph speed case is {midSpanMaxDisplacement} : Newmark {np.min(newmark[n2.DOF[2].id,:])}")
  plt.plot(x, timeHistoryDisplacementVector[n2.DOF[2].id,:], label=f"{speed_kmph} kmph")

print("** Time History Analysis **")
performTimeHistoryAnalysis(0.01)
performTimeHistoryAnalysis(20)
performTimeHistoryAnalysis(40)
performTimeHistoryAnalysis(70)
performTimeHistoryAnalysis(100)
performTimeHistoryAnalysis(160)
performTimeHistoryAnalysis(200)
performTimeHistoryAnalysis(400)
plt.legend()
plt.show()

# Calculation using oscilating loads
performTimeHistoryAnalysis(0.01)
performTimeHistoryAnalysis(70)
nonOscillatingLoad = DOFClass.ImbalancedActionVector.copy()
DOFClass.resetAllActionsAndReactions()

def performTimeHistoryAnalysisOscilatingLoad(speed_kmph):
  oscilatingFreqHz = 1
  oscilatingPercentage = 0.1
  speed_mps = speed_kmph/3.6
  dt = MAX_LOAD_DISTANCE/speed_mps/(MAX_LOADCASE-1)
  oscillatingFactor = 1 + oscilatingPercentage * np.sin(2 * np.pi * oscilatingFreqHz * np.arange(MAX_LOADCASE)*dt)
  oscillatingLoad = nonOscillatingLoad.copy() @ sp.diags([oscillatingFactor],[0])
  DOFClass.ImbalancedActionVector = oscillatingLoad
  timeHistoryDisplacementVector, timeHistoryVelocityVector, timeHistoryAccelerationVector = DOFClass.analyseTimeHistory(dt, eigenValues, eigenVectors, dampingRatios)
  midSpanMaxDisplacement = np.min(timeHistoryDisplacementVector[n2.DOF[2].id,:])
  print(f"Maximum displacement @ midspan for {speed_kmph}kmph speed case is {midSpanMaxDisplacement}")
  plt.plot(x, timeHistoryDisplacementVector[n2.DOF[2].id,:], label=f"Os {speed_kmph} kmph")
  DOFClass.resetAllActionsAndReactions()

performTimeHistoryAnalysisOscilatingLoad(70)
plt.legend()
plt.show()