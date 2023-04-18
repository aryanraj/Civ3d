import os
os.environ["MAX_DOF"] = str(30)

from Models import DOFClass, Node, BeamSection, Beam
n0 = Node([0,0,0], [1, 1, 1, 1, 1, 1])
n1 = Node([1,0,0])
n2 = Node([5,0,0])
n3 = Node([9,0,0])
n4 = Node([10,0,0], [1, 1, 1, 1, 1, 1])

section = BeamSection()
b1 = Beam([n1, n2, n3], section, A=n0, B=n4, isConstrainedA=[1,1,1,1,1,0], isConstrainedB=[1,1,1,1,1,0])
b1.addUDL(1, -10)
DOFClass.analyse()
midSpanMaxDisplacement = n2.getDisplacement()[1]
print(f"Maximum displacement @ midspan is {midSpanMaxDisplacement}")

b1.addUDL(1, -10)
b1.addSimpleEndStiffness(endStiffnessA=[0,0,0,0,0,1], endStiffnessB=[0,0,0,0,0,1])
DOFClass.analyse()
midSpanMaxDisplacement = n2.getDisplacement()[1]
print(f"Maximum displacement @ midspan is {midSpanMaxDisplacement}")

n2.addNodalForce([0,-10,0,0,0,0])
DOFClass.analyse()
midSpanMaxDisplacement = n2.getDisplacement()[1]
print(f"Maximum displacement @ midspan is {midSpanMaxDisplacement}")

print(DOFClass.DisplacementVector)
print(DOFClass.ActionVector)
print(DOFClass.ReactionVector)

