from __future__ import annotations
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from typing import ClassVar
from dataclasses import dataclass, field
from itertools import count
import os

MAX_DOF: int = int(os.getenv("MAX_DOF", 10000))

@dataclass
class DOFClass():
  id: int = field(init=False, default_factory=count().__next__)
  isRestrained: bool = False
  constraints: tuple[list[DOFClass],list[float]] | None = field(init=False, default=None)

  # Class Variables
  DOFList: ClassVar[list[DOFClass]] = []
  StiffnessMatrix: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,MAX_DOF), dtype=np.float64)
  MassMatrix: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,MAX_DOF), dtype=np.float64)
  ConstraintMatrix: ClassVar[sp.lil_array] = sp.lil_array(sp.eye(MAX_DOF, dtype=np.float64))
  ReactionVector: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,1), dtype=np.float64)
  ActionVector: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,1), dtype=np.float64)
  DisplacementVector: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,1), dtype=np.float64)
  ImbalancedForceVector: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,1), dtype=np.float64)
  ImbalancedDisplacementVector: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,1), dtype=np.float64)
  RestraintVector: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,1), dtype=np.bool_)

  def __post_init__(self):
    if not type(self.isRestrained) is bool:
      self.isRestrained = bool(self.isRestrained)
    cls = type(self)
    cls.RestraintVector[self.id,0] = self.isRestrained
    cls.DOFList.append(self)

  def __del__(self):
    raise NotImplementedError(f"Deletion of {type(self).__name__} is not supported")

  @property
  def displacement(self):
    """Displacement Property"""
    cls = type(self)
    return cls.DisplacementVector[self.id,0]
  
  @property
  def reaction(self):
    """Force Property"""
    cls = type(self)
    return cls.ReactionVector[self.id,0]
  
  @property
  def action(self):
    """Force Property"""
    cls = type(self)
    return cls.ActionVector[self.id,0]
  
  @property
  def isConstrained(self):
    """Checks if DOF constrained or not"""
    return not self.constraints is None

  def addDisplacement(self, val):
    cls = type(self)
    cls.ImbalancedDisplacementVector[self.id,0] += val
  
  def addAction(self, val):
    cls = type(self)
    cls.ActionVector[self.id,0] += val
    cls.ImbalancedForceVector[self.id,0] += val

  def addFixedEndReaction(self, val):
    cls = type(self)
    cls.ActionVector[self.id,0] -= val
    cls.ImbalancedForceVector[self.id,0] -= val

  def addConstraint(self, DOFs:list[DOFClass], factors:list[float]):
    if self.isConstrained:
      raise Exception(f"Already Constrained DOF {self.id}")
    if self.isRestrained:
      raise Exception(f"DOF {self.id} is restrained and cannot be further constrained")
    m = sp.lil_array(sp.eye(MAX_DOF, dtype=np.float64))
    m[self.id, self.id] = 0
    m[self.id, [_.id for _ in DOFs]] = factors
    self.constraints = (DOFs, factors)
    cls = type(self)
    cls.ConstraintMatrix = cls.ConstraintMatrix @ m
  
  def removeConstraint(self):
    if not self.isConstrained:
      raise Exception("No Constraint available to remove")
    raise NotImplementedError("Future feature: remove constraint")
    self.constraints = None
  
  def addRestraint(self):
    if self.isConstrained:
      raise Exception(f"DOF {self.id} is constrained and cannot be further restrained")
    if self.isRestrained:
      raise Exception(f"Already Restrained DOF {self.id}")
    cls = type(self)
    self.isRestrained = True
    cls.RestraintVector[self.id,0] = self.isRestrained

  def removeRestraint(self):
    if not self.isRestrained:
      raise Exception("No Restraint available to remove")
    cls = type(self)
    self.isRestrained = False
    cls.RestraintVector[self.id,0] = self.isRestrained
    cls.ImbalancedForceVector[self.id,0] -= cls.ReactionVector[self.id,0]
    cls.ReactionVector[self.id,0] = 0

  @classmethod
  def addStiffness(self, DOFs:list[DOFClass], K:npt.NDArray[np.float64]):
    self.StiffnessMatrix[np.ix_([_.id for _ in DOFs],[_.id for _ in DOFs])] += K

  @classmethod
  def addMass(self, DOFs:list[DOFClass], M:npt.NDArray[np.float64]):
    self.MassMatrix[np.ix_([_.id for _ in DOFs],[_.id for _ in DOFs])] += M

  @classmethod
  def analyse(cls):
    Kg = cls.ConstraintMatrix.T @ cls.StiffnessMatrix @ cls.ConstraintMatrix
    disp = cls.ConstraintMatrix @ cls.ImbalancedDisplacementVector
    force = cls.ConstraintMatrix.T @ cls.ImbalancedForceVector

    # Mask for all the DOFs which are not restraints
    mask = ~cls.RestraintVector.toarray().flatten()

    # Filter out DOFs which have zero stiffness and zero unbalanced force
    # TODO: Fix this part using LU_factor and LU_solve
    for i, _Kg, _force in zip(range(len(mask)), Kg, force):
      if cls.RestraintVector[i,0]: continue
      if np.all(_Kg.toarray() == 0) and np.all(_force.toarray() == 0):
        mask[i] = False
      if np.all(_Kg.toarray() == 0) and np.any(_force.toarray() != 0):
        raise Exception(f"Instability at DOF {i}")
    
    # Do not proceed if the DOF mask is empty
    if np.any(mask):
      K11 = Kg[np.ix_(mask, mask)]
      K12 = Kg[np.ix_(mask, ~mask)]
      K21 = Kg[np.ix_(~mask, mask)]
      K22 = Kg[np.ix_(~mask, ~mask)]

      Qk = force[mask]
      Dk = disp[~mask]
      Du = sp.lil_matrix([splinalg.spsolve(K11, Qk - K12 @ Dk)]).T # Not sure why is sp solve returning list object
      Qu = K21 @ Du + K22 @ Dk

      disp[mask] = Du
      force[~mask] -= Qu
      force[mask] = 0

    cls.DisplacementVector += cls.ConstraintMatrix @ disp
    cls.ReactionVector -= force
    cls.ImbalancedDisplacementVector[:,0] = 0
    cls.ImbalancedForceVector[:,0] = 0

  @classmethod
  def eig(cls, nModes):
    Kg = cls.ConstraintMatrix.T @ cls.StiffnessMatrix @ cls.ConstraintMatrix
    Mg = cls.ConstraintMatrix.T @ cls.MassMatrix @ cls.ConstraintMatrix

    # Mask for all the DOFs which are not restraints
    mask = ~cls.RestraintVector.toarray().flatten()

    # Filter out DOFs which have zero stiffness
    # TODO: Fix this part using LU_factor and LU_solve
    for i, _Kg in zip(range(len(mask)), Kg, Mg):
      if cls.RestraintVector[i,0]: continue
      if np.all(_Kg.toarray() == 0):
        mask[i] = False
    
    # Do not proceed if the DOF mask is empty
    if np.any(mask):
      K11 = Kg[np.ix_(mask, mask)]
      M11 = Mg[np.ix_(mask, mask)]
      return splinalg.eigsh(K11, nModes, M11, sigma=0)

  @classmethod
  def createCopy(cls, obj:DOFClass) -> DOFClass:
    return cls(obj.isRestrained)