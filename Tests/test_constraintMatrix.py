import os
os.environ["MAX_DOF"] = str(18)

from Models import DOFClass, Node

n1 = Node([0,0,0])
n2 = Node([1,0,0])
n3 = Node([1,0,1])

n1.addRestraint([1,1,1,1,1,1])
n1.constrainChildNode(n2, [1,1,1,1,1,1])
n2.constrainChildNode(n3, [1,1,1,1,1,1])

n3.addNodalForce([1,0,0,0,0,0])

DOFClass.analyse()
print(n1.getReaction())
