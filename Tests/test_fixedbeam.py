import os
os.environ["MAX_DOF"] = str(24)

from Models import DOFClass, Node, BeamSection, FixedBeam
n1 = Node([0,0,0], [1, 1, 1, 1, 1, 1])
n2 = Node([10,0,0], [1, 1, 1, 1, 1, 1])

section = BeamSection()
b1 = FixedBeam(n1, n2, section)
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

