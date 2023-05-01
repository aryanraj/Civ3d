import os
os.environ["MAX_DOF"] = str(42)

from Models import DOFClass, Node, BeamSection, Beam
n0 = Node([0,0,0], [1, 1, 1, 1, 1, 1])
n1 = Node([1,0,0])
n2 = Node([5,0,0])
n3 = Node([9,0,0])
n4 = Node([10,0,0], [1, 1, 1, 1, 1, 1])

section = BeamSection()
b1 = Beam([n1, n2], section, A=n0, constraintsA=[1,1,1,1,1,0])
b2 = Beam([n2, n3], section, B=n4, constraintsB=[1,1,1,1,1,0])
b1.addUDL(1, -10, [0])
b2.addUDL(1, -10, [0])
DOFClass.analyse()

b1.addUDL(1, -10, [0])
b2.addUDL(1, -10, [0])
b1.addEndStiffness(endStiffnessA=[0,0,0,0,0,1])
b2.addEndStiffness(endStiffnessB=[0,0,0,0,0,1])
DOFClass.analyse()

print(DOFClass.getDisplacementVector([0]).flatten())
print(DOFClass.getActionVector([0]).flatten())
print(DOFClass.getReactionVector([0]).flatten())
