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
  initialRestraint: InitVar[npt.NDArray[np.bool_]] = None
  Kg: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))
  Mg: npt.NDArray[np.float64]  = field(default_factory=lambda:np.zeros((6,6)))
  axis: InitVar[npt.NDArray[np.float64]] = None

  DOF: list[DOFClass] = field(init=False, default_factory=list)

  def __post_init__(self, initialRestraint, axis):
    initialRestraint = utils.ensure1DNumpyArray(initialRestraint, np.bool_, np.zeros((6,),dtype=np.bool_))
    axis = utils.ensure2DSquareNumpyArray(axis, np.float64, np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]]))
    self.coord = utils.ensure1DNumpyArray(self.coord, np.float64)
    self.Kg = utils.ensure2DSquareNumpyArray(self.Kg, np.float64)
    self.DOF.extend([DOFClass(DOFTypes.DISPLACEMENT, dir, res) for dir, res in zip(axis, initialRestraint[:3])])
    self.DOF.extend([DOFClass(DOFTypes.ROTATION, dir, res) for dir, res in zip(axis, initialRestraint[3:])])

  def __del__(self):
    raise NotImplementedError(f"Deletion of {type(self).__name__} is not supported")

  @property
  def axis(self) -> npt.NDArray[np.float64]:
    return np.array([_.dir for _ in self.DOF[:3]], dtype=np.float64)

  @property
  def restraint(self) -> npt.NDArray[np.bool_]:
    return np.array([_DOF.isRestrained for _DOF in self.DOF], dtype=np.bool_)

  @property
  def constraint(self) -> npt.NDArray[np.bool_]:
    return np.array([_DOF.isConstrained for _DOF in self.DOF], dtype=np.bool_)

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
      if _localDOF.isConstrained:
        _localDOF.removeConstraint()
      if _constraint:
        _localDOF.addConstraint(self.DOF, _Tgl)

  def addChildNodeStiffness(self, childNode: Node, K6: npt.NDArray[np.float64]):
    if K6.ndim != 2 or K6.shape != (6,6):
      raise Exception("Stiffness matrix must be 2-Dimensional of size (6,6)")
    C6 = self.getConstraintMatrixFor(childNode)
    K12 = np.vstack((np.hstack((K6, -K6)), np.hstack((-K6, K6))))
    C12 = np.vstack((np.hstack((C6, np.zeros((6,6)))), np.hstack((np.zeros((6,6)),np.eye(6)))))
    DOFClass.addStiffness(self.DOF+childNode.DOF, C12.T @ K12 @ C12)

  def addStiffness(self, K: npt.NDArray[np.float64]):
    self.Kg += K
    DOFClass.addStiffness(self.DOF, K)
  
  def setStiffness(self, KTarget:npt.NDArray[np.float64]):
    if not type(KTarget) is np.ndarray:
      KTarget = np.array(KTarget, dtype=np.float64)
    if KTarget.ndim == 1:
      KTarget = np.diag(KTarget)
    elif KTarget.ndim != 2:
      raise Exception("The dimension of K should either be 1 or 2")
    self.addStiffness(KTarget - self.Kg)

  def addMass(self, M: npt.NDArray[np.float64]):
    self.Mg += M
    DOFClass.addMass(self.DOF, M)

  def addLumpedMass(self, mass: float):
    self.addMass(np.diag([mass]*3+[0]*3))

  def addNodalForce(self, force:npt.NDArray[np.float64], loadCases:list[int]):
    if not type(force) is np.ndarray or force.ndim != 2:
      raise Exception("The force should be an NDArray with ndim=2")
    if force.shape[1] != 1 and force.shape[1] != len(loadCases):
      raise Exception(f"The force matrix should have number of columns either 1 OR {len(loadCases)=}")
    if force.shape[0] != len(self.DOF):
      raise Exception(f"The force matrix should have number of rows same as {len(self.DOF)=}")
    for _DOF, _force in zip(self.DOF, force):
      _DOF.addAction(np.array([_force]), loadCases)

  def addSelfWeight(self, dir:int, factor:float, loadCases:list[int]):
    self.addNodalForce(np.array([self.Mg[dir]*9.806*factor]).T, loadCases)

  def getAction(self, loadCases:list[int]) -> npt.NDArray[np.float64]:
    return np.vstack([_.action(loadCases) for _ in self.DOF])

  def getReaction(self, loadCases:list[int]) -> npt.NDArray[np.float64]:
    return np.vstack([_.reaction(loadCases) for _ in self.DOF])
  
  def getDisplacement(self, loadCases:list[int]) -> npt.NDArray[np.float64]:
    return np.vstack([_.displacement(loadCases) for _ in self.DOF])
