from __future__ import annotations
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from . import DOFClass

@dataclass
class Node:
  id: int
  coord: npt.NDArray[np.float64]
  restraint: npt.NDArray[np.bool_] = field(default_factory=lambda:np.zeros((6,1), dtype=np.bool_))
  Kg: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))
  DOF: list[DOFClass] = field(init=False)
  disp: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.zeros((6,1)))
  force: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.zeros((6,1)))

  def __post_init__(self):
    if type(self.coord) is list:
      self.coord = np.array(self.coord, dtype=np.float64)
    if type(self.restraint) is list:
      self.restraint = np.array([self.restraint], dtype=np.bool_).T
    if type(self.Kg) is list:
      self.Kg = np.array(self.Kg, dtype=np.float64)
    self.DOF = [DOFClass(dir, res) for dir, res in zip([[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1]], self.restraint)]

  def addRestraint(self, res: npt.NDArray[np.bool_]) -> None:
    res = np.array(res, dtype=np.bool_)
    for _DOF, _res in zip(self.DOF, res):
      if _res and not _DOF.isRestrained:
        _DOF.addRestraint()
      if not _res and _DOF.isRestrained:
        _DOF.removeRestraint()
  
  def constrainChildNode(self, childNode: Node, isConstrained: npt.NDArray[np.bool_]):
    # Computing Transformation matrix in parent local axis
    TransformationMartix = np.eye(6, dtype=np.float64)
    V = childNode.coord - self.coord
    TransformationMartix[0:3, 3:6] = -computePreCrossProductTransform(V)
    # Change Basis from Parent axis to child axis
    BasisChangeMartix = np.zeros((6,6), dtype=np.float64)
    BasisChangeMartix[0:3, 0:3] = np.array([_.dir for _ in childNode.DOF[0:3]]) @ np.array([_.dir for _ in self.DOF[0:3]]).T
    BasisChangeMartix[3:6, 3:6] = np.array([_.dir for _ in childNode.DOF[3:6]]) @ np.array([_.dir for _ in self.DOF[3:6]]).T
    # Transformation matrix in Child Basis
    Tgl = BasisChangeMartix @ TransformationMartix
    for _isConstrained, _Tgl, _localDOF in zip(isConstrained, Tgl, childNode.DOF):
      if not _isConstrained: continue
      _localDOF.addConstraint(self.DOF,_Tgl)

  def addStiffness(self, arg: npt.NDArray[np.float64]):
    DOFClass.addStiffness(self.DOF, arg)

def computePreCrossProductTransform(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
  return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
