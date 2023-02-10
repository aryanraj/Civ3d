from __future__ import annotations
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field, InitVar
from . import DOFClass

@dataclass
class Node:
  coord: npt.NDArray[np.float64]
  restraint: InitVar[npt.NDArray[np.bool_]] = None
  Kg: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))

  DOF: list[DOFClass] = field(init=False)
  disp: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.zeros((6,1)))
  force: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.zeros((6,1)))

  def __post_init__(self, restraint):
    if not type(restraint) is list and not type(restraint) is np.array:
      restraint = np.zeros((6,), dtype=np.bool_)
    if type(self.coord) is list:
      self.coord = np.array(self.coord, dtype=np.float64)
    if type(restraint) is list:
      restraint = np.array([restraint], dtype=np.bool_).T
    if type(self.Kg) is list:
      self.Kg = np.array(self.Kg, dtype=np.float64)
    self.DOF = [DOFClass(dir, res) for dir, res in zip([[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1]], restraint)]

  def __del__(self):
    raise NotImplementedError(f"Deletion of {type(self).__name__} is not supported")

  @property
  def restraint(self) -> npt.NDArray[np.bool_]:
    return np.array([_DOF.isRestrained for _DOF in self.DOF], dtype=np.bool_)

  def addRestraint(self, res: npt.NDArray[np.bool_]) -> None:
    res = np.array(res, dtype=np.bool_)
    for _DOF, _res in zip(self.DOF, res):
      if _res and not _DOF.isRestrained:
        _DOF.addRestraint()
      if not _res and _DOF.isRestrained:
        _DOF.removeRestraint()
  
  def getConstraintMatrixFor(self, childNode: Node) -> npt.NDArray[np.float64]:
    # Computing Transformation matrix in parent local axis
    TransformationMartix = np.eye(6, dtype=np.float64)
    V = childNode.coord - self.coord
    TransformationMartix[0:3, 3:6] = -computePreCrossProductTransform(V)
    # Change Basis from Parent axis to child axis
    BasisChangeMartix = np.zeros((6,6), dtype=np.float64)
    BasisChangeMartix[0:3, 0:3] = np.array([_.dir for _ in childNode.DOF[0:3]]) @ np.array([_.dir for _ in self.DOF[0:3]]).T
    BasisChangeMartix[3:6, 3:6] = np.array([_.dir for _ in childNode.DOF[3:6]]) @ np.array([_.dir for _ in self.DOF[3:6]]).T
    # Transformation matrix in Child Basis
    return BasisChangeMartix @ TransformationMartix

  def constrainChildNode(self, childNode: Node, constraints: npt.NDArray[np.bool_]):
    constraintMatrix = self.getConstraintMatrixFor(childNode)
    for _constraint, _Tgl, _localDOF in zip(constraints, constraintMatrix, childNode.DOF):
      if not _constraint: continue
      _localDOF.addConstraint(self.DOF,_Tgl)

  def addChildNodeStiffness(self, childNode: Node, K6: npt.NDArray[np.float64]):
    if K6.ndim != 2 or K6.shape != (6,6):
      raise Exception("Stiffness matrix must be 2-Dimensional of size (6,6)")
    C6 = self.getConstraintMatrixFor(childNode)
    K12 = np.vstack((np.hstack((K6, -K6)), np.hstack((-K6, K6))))
    C12 = np.vstack((np.hstack((C6, np.zeros((6,6)))), np.hstack((np.zeros((6,6)),np.eye(6)))))
    DOFClass.addStiffness(self.DOF+childNode.DOF, C12.T @ K12 @ C12)

  def addStiffness(self, K: npt.NDArray[np.float64]):
    DOFClass.addStiffness(self.DOF, K)

  def copy(self):
    cls = type(self)
    return cls(self.coord, self.restraint, self.Kg)

def computePreCrossProductTransform(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
  return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
