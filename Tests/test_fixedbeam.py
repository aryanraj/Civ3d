import os
os.environ["MAX_DOF"] = str(24)

from Models import FixedBeam, Node, DOFClass
n1 = Node(1, [0,0,0], [1, 1, 1, 1, 1, 1])
n2 = Node(2, [10,0,0], [1, 1, 1, 1, 1, 1])

b1 = FixedBeam(1, n1, n2)
b1.addUDL(1, -10)
DOFClass.analyse()

b1.addUDL(1, -10)
DOFClass.analyse()

n2.DOF[1].removeRestraint()
n2.DOF[5].removeRestraint()
DOFClass.analyse()

print(DOFClass.DisplacementVector)
print(DOFClass.ActionVector)
print(DOFClass.ReactionVector)

