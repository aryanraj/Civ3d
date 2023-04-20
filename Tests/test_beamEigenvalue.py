import os
os.environ["MAX_DOF"] = str(66)

import numpy as np
from Models import DOFClass, Node, BeamSection, Beam
n0 = Node([0,0,0], [1, 1, 1, 1, 1, 1])
n1 = Node([1,0,0])
n21 = Node([2,0,0], [1, 0, 1, 1, 1, 0])
n22 = Node([3,0,0], [1, 0, 1, 1, 1, 0])
n23 = Node([4,0,0], [1, 0, 1, 1, 1, 0])
n24 = Node([5,0,0], [1, 0, 1, 1, 1, 0])
n25 = Node([6,0,0], [1, 0, 1, 1, 1, 0])
n26 = Node([7,0,0], [1, 0, 1, 1, 1, 0])
n27 = Node([8,0,0], [1, 0, 1, 1, 1, 0])
n3 = Node([9,0,0])
n4 = Node([10,0,0], [1, 1, 1, 1, 1, 1])

acceptableError = 0.001

section = BeamSection()
b1 = Beam([n1, n21, n22, n23, n24, n25, n26, n27, n3], section, A=n0, B=n4, constraintsA=[1,1,1,1,1,1], constraintsB=[1,1,1,1,1,1])
D,V,EffectiveMass,MassParticipationFactor = DOFClass.eig(1)
T0 = 2*np.pi/D[0]**0.5
# print(f"Fundamental Time period {T0:.2f=} secs for beam only section")

massPerLength = section.rho * section.Area
b1.addMassUDL(section.rho * section.Area)
D,V,EffectiveMass,MassParticipationFactor = DOFClass.eig(1)
T1 = 2*np.pi/D[0]**0.5
print(f"Fundamental Time period {T1=:.2f} secs for beam with additional mass added as UDL")
assert ((ratio:=T1/T0) - (expectedRatio:=2**0.5)) < acceptableError, f"The Ratio {ratio=:.3f} is expected to be nearly {expectedRatio:.3f}"

for _ in b1.nodes:
  _.addLumpedMass(massPerLength)
D,V,EffectiveMass,MassParticipationFactor = DOFClass.eig(1)
T2 = 2*np.pi/D[0]**0.5
print(f"Fundamental Time period {T2=:.2f} secs for beam with additional lumped mass on Nodes")
assert ((ratio:=T2/T0) - (expectedRatio:=3**0.5)) < acceptableError, f"The Ratio {ratio=:.3f} is expected to be nearly {expectedRatio:.3f}"
