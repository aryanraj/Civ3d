import os
os.environ["MAX_DOF"] = str(30)

from Models import DOFClass, Node, BeamSection, Beam
n0 = Node([0,0,0], [1, 1, 1, 1, 1, 1])
n1 = Node([1,0,0])
n2 = Node([5,0,0])
n3 = Node([9,0,0])
n4 = Node([10,0,0], [1, 1, 1, 1, 1, 1])

section = BeamSection()
b1 = Beam([n1, n2, n3], section, A=n0, B=n4, constraintsA=[1,1,1,1,1,0], constraintsB=[1,1,1,1,1,0])
b1.addSelfWeight(1, -1, [0])
DOFClass.analyse()

b1.addUDL(1, -10, [0])
b1.addEndStiffness(endStiffnessA=[0,0,0,0,0,1], endStiffnessB=[0,0,0,0,0,1])
DOFClass.analyse()

print("Action Vector")
print("n1:", n1.getAction([0]).flatten())
print("n2:", n2.getAction([0]).flatten())
print("n3:", n3.getAction([0]).flatten())
print("Reactions:-")
print("n0:", n0.getReaction([0]).flatten())
print("n4:", n4.getReaction([0]).flatten())
print("Displacement:-")
print("n0:", n0.getDisplacement([0]).flatten())
print("n1:", n1.getDisplacement([0]).flatten())
print("n2:", n2.getDisplacement([0]).flatten())
print("n3:", n3.getDisplacement([0]).flatten())
print("n4:", n4.getDisplacement([0]).flatten())
