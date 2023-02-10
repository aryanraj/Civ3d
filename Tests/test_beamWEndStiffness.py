import os
os.environ["MAX_DOF"] = str(42)

from Models import Beam, Node, DOFClass
n0 = Node([0,0,0], [1, 1, 1, 1, 1, 1])
n1 = Node([1,0,0])
n2 = Node([5,0,0])
n3 = Node([9,0,0])
n4 = Node([10,0,0], [1, 1, 1, 1, 1, 1])

b1 = Beam(n1, n2, A=n0, isConstrainedA=[1,1,1,1,1,0])
b2 = Beam(n2, n3, B=n4, isConstrainedB=[1,1,1,1,1,0])
b1.addUDL(1, -10)
b2.addUDL(1, -10)
DOFClass.analyse()

b1.addUDL(1, -10)
b2.addUDL(1, -10)
b1.addSimpleEndStiffness(endStiffnessA=[0,0,0,0,0,1])
b2.addSimpleEndStiffness(endStiffnessB=[0,0,0,0,0,1])
DOFClass.analyse()

print(DOFClass.DisplacementVector)
print(DOFClass.ActionVector)
print(DOFClass.ReactionVector)
