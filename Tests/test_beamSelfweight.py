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
b1.addSelfWeight(1)
DOFClass.analyse()

b1.addUDL(1, -10)
b1.addEndStiffness(endStiffnessA=[0,0,0,0,0,1], endStiffnessB=[0,0,0,0,0,1])
DOFClass.analyse()

print("Action Vector")
print("n1:", n1.getAction().tolist())
print("n2:", n2.getAction().tolist())
print("n3:", n3.getAction().tolist())
print("Reactions:-")
print("n0:", n0.getReaction().tolist())
print("n4:", n4.getReaction().tolist())
print("Displacement:-")
print("n0:", n0.getDisplacement().tolist())
print("n1:", n1.getDisplacement().tolist())
print("n2:", n2.getDisplacement().tolist())
print("n3:", n3.getDisplacement().tolist())
print("n4:", n4.getDisplacement().tolist())
