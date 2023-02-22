import os
os.environ["MAX_DOF"] = str(66)

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

section = BeamSection()
b1 = Beam([n1, n21, n22, n23, n24, n25, n26, n27, n3], section, A=n0, B=n4, isConstrainedA=[1,1,1,1,1,1], isConstrainedB=[1,1,1,1,1,1])
print(DOFClass.eig(1))