import os
import numpy as np
from scipy.linalg import eig
MAX_DOF = (4+4)*6
MAX_LOADCASE = (4+4)*3
os.environ["MAX_DOF"] = str(MAX_DOF)
os.environ["MAX_LOADCASE"] = str(MAX_LOADCASE)

np.set_printoptions(edgeitems=10, linewidth=250, formatter=dict(float=lambda x: " ____ " if x==0 else f"{x:6.3f}"))

from Models import DOFClass, Node, BeamSection, FixedBeam
n1 = Node([ 0, 0,0], [0, 0, 1, 1, 1, 0])
n2 = Node([ 0,10,0], [0, 0, 1, 1, 1, 0])
n3 = Node([10,10,0], [0, 0, 1, 1, 1, 0])
n4 = Node([10, 0,0], [0, 0, 1, 1, 1, 0])

n5 = Node([ 0, 5,0], [0, 1, 1, 1, 1, 0])
n6 = Node([ 5,10,0], [1, 0, 1, 1, 1, 0])
n7 = Node([10, 5,0], [0, 1, 1, 1, 1, 0])
n8 = Node([ 5, 0,0], [1, 0, 1, 1, 1, 0])

section = BeamSection()
b11 = FixedBeam(n1, n5, section)
b12 = FixedBeam(n5, n2, section)
b21 = FixedBeam(n2, n6, section)
b22 = FixedBeam(n6, n3, section)
b31 = FixedBeam(n3, n7, section)
b32 = FixedBeam(n7, n4, section)
b41 = FixedBeam(n1, n8, section)
b42 = FixedBeam(n8, n4, section)

concernedDOFs = [i for i in range(8*6) if i%6 in [0,1,5]]

freeMask = np.zeros((MAX_DOF,), dtype=np.bool_)
freeMask[concernedDOFs] = True

# Eigenvalue analysis
K = DOFClass.StiffnessMatrix[np.ix_(freeMask, freeMask)]
w, vr = eig(K.todense())
vr = vr.real
vr[np.abs(vr) < 0.001] = 0
w = w.real
w[np.abs(w) < 0.001] = 0
print(vr)
print(w)

# Setting it up as loads
loadCases = [*range(MAX_LOADCASE)]
for i, iDOF in enumerate(concernedDOFs):
    DOFClass.DOFList[iDOF].addAction(vr[i,:], loadCases)

DOFClass.analyse()

# print(DOFClass.getDisplacementVector([0]).flatten())
# print(DOFClass.getActionVector([0]).flatten())
# print(DOFClass.getReactionVector([0]).flatten())

from Views.simpleStructure import SimpleView
modeShapes = DOFClass.DisplacementVector.todense()
print(modeShapes)
modeShapeTags = [f"loadCase {i}" for i in loadCases]
view = SimpleView([n1, n2, n3, n4, n5, n6, n7, n8], [b11, b12, b21, b22, b31, b32, b41, b42], modeShapes, modeShapeTags)
view.start()

