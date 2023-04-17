from __future__ import annotations
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field, InitVar
from . import DOFClass
from . import utils
from .types import DOFTypes

@dataclass
class Node:
  coord: npt.NDArray[np.float64]
  restraint: InitVar[npt.NDArray[np.bool_]] = None
  Kg: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))
  axis: InitVar[npt.NDArray[np.float64]] = None

  DOF: list[DOFClass] = field(init=False, default_factory=list)
  disp: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.zeros((6,1)))
  force: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.zeros((6,1)))

  def __post_init__(self, restraint, axis):
    restraint = utils.ensure1DNumpyArray(restraint, np.bool_, np.zeros((6,),dtype=np.bool_))
    axis = utils.ensure2DSquareNumpyArray(axis, np.float64, np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]]))
    self.coord = utils.ensure1DNumpyArray(self.coord, np.float64)
    self.Kg = utils.ensure2DSquareNumpyArray(self.Kg, np.float64)
    self.DOF.extend([DOFClass(DOFTypes.DISPLACEMENT, dir, res) for dir, res in zip(axis, restraint[:3])])
    self.DOF.extend([DOFClass(DOFTypes.ROTATION, dir, res) for dir, res in zip(axis, restraint[3:])])

  def __del__(self):
    raise NotImplementedError(f"Deletion of {type(self).__name__} is not supported")

  @property
  def axis(self) -> npt.NDArray[np.float64]:
    return np.array([_.dir for _ in self.DOF[:3]], dtype=np.float64)

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
    TransformationMartix[0:3, 3:6] = -utils.computePreCrossProductTransform(V)
    # Change Basis from Parent axis to child axis
    BasisChangeMartix = np.zeros((6,6), dtype=np.float64)
    BasisChangeMartix[0:3, 0:3] = utils.globalToLocalBasisChangeMatrix(self.axis, childNode.axis)
    BasisChangeMartix[3:6, 3:6] = utils.globalToLocalBasisChangeMatrix(self.axis, childNode.axis)
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

  def addMass(self, M: npt.NDArray[np.float64]):
    DOFClass.addMass(self.DOF, M)

  def addLumpedMass(self, mass: float):
    self.addMass(np.diag([mass]*3+[0]*3))

  def getAction(self) -> npt.NDArray[np.float64]:
    return np.array([_.action for _ in self.DOF])

  def getReaction(self) -> npt.NDArray[np.float64]:
    return np.array([_.reaction for _ in self.DOF])
  
  def getDisplacement(self) -> npt.NDArray[np.float64]:
    return np.array([_.displacement for _ in self.DOF])
