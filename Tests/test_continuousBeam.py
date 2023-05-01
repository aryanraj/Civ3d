import os
os.environ["MAX_DOF"] = str(30)

from Models import DOFClass, Node, BeamSection, Beam
import numpy as np

n0 = Node([0,0,0], [1, 1, 1, 1, 1, 1])
n1 = Node([1,0,0])
n2 = Node([5,0,0])
n3 = Node([9,0,0])
n4 = Node([10,0,0], [1, 1, 1, 1, 1, 1])

section = BeamSection()
b1 = Beam([n1, n2, n3], section, A=n0, B=n4, constraintsA=[1,1,1,1,1,0], constraintsB=[1,1,1,1,1,0])
b1.addUDL(1, -10, [0])
DOFClass.analyse()
midSpanMaxDisplacement = n2.getDisplacement([0]).flatten()[1]
print(f"Maximum displacement @ midspan is {midSpanMaxDisplacement}")

b1.addUDL(1, -10, [0])
b1.addEndStiffness(endStiffnessA=[0,0,0,0,0,1], endStiffnessB=[0,0,0,0,0,1])
DOFClass.analyse()
midSpanMaxDisplacement = n2.getDisplacement([0]).flatten()[1]
print(f"Maximum displacement @ midspan is {midSpanMaxDisplacement}")

n2.addNodalForce(np.array([[0,-10,0,0,0,0]]).T, [0])
DOFClass.analyse()
midSpanMaxDisplacement = n2.getDisplacement([0]).flatten()[1]
print(f"Maximum displacement @ midspan is {midSpanMaxDisplacement}")

for i in range(11):
  b1.addPointLoad(1, -10, i, [0])
DOFClass.analyse()
midSpanMaxDisplacement = n2.getDisplacement([0]).flatten()[1]
print(f"Maximum displacement @ midspan is {midSpanMaxDisplacement}")

print(DOFClass.getDisplacementVector([0]).flatten())
print(DOFClass.getActionVector([0]).flatten())
print(DOFClass.getReactionVector([0]).flatten())

