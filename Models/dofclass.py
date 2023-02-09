from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import ClassVar
from dataclasses import dataclass, field
from itertools import count
import os

MAX_DOF: int = int(os.getenv("MAX_DOF", 10000))

@dataclass
class DOFClass():
  id: int = field(init=False, default_factory=count().__next__)
  dir: npt.NDArray[np.float64]
  isRestrained: bool = False
  constraints: tuple[list[DOFClass],list[float]] | None = field(init=False, default=None)

  # Class Variables
  DOFList: ClassVar[list[DOFClass]] = []
  StiffnessMatrix: ClassVar[npt.NDArray[np.float64]] = np.zeros((MAX_DOF,MAX_DOF), dtype=np.float64)
  ConstraintMatrix: ClassVar[npt.NDArray[np.float64]] = np.eye(MAX_DOF, dtype=np.float64)
  ReactionVector: ClassVar[npt.NDArray[np.float64]] = np.zeros((MAX_DOF,1), dtype=np.float64)
  ActionVector: ClassVar[npt.NDArray[np.float64]] = np.zeros((MAX_DOF,1), dtype=np.float64)
  DisplacementVector: ClassVar[npt.NDArray[np.float64]] = np.zeros((MAX_DOF,1), dtype=np.float64)
  ImbalancedForceVector: ClassVar[npt.NDArray[np.float64]] = np.zeros((MAX_DOF,1), dtype=np.float64)
  ImbalancedDisplacementVector: ClassVar[npt.NDArray[np.float64]] = np.zeros((MAX_DOF,1), dtype=np.float64)
  RestraintVector: ClassVar[npt.NDArray[np.bool_]] = np.zeros((MAX_DOF,), dtype=np.bool_)

  def __post_init__(self):
    if type(self.dir) is list:
      self.dir = np.array(self.dir, dtype=np.float64)
    if not type(self.isRestrained) is bool:
      self.isRestrained = bool(self.isRestrained)
    cls = type(self)
    cls.RestraintVector[self.id] = self.isRestrained
    cls.DOFList.append(self)

  @property
  def displacement(self):
    """Displacement Property"""
    cls = type(self)
    return cls.DisplacementVector[self.id]
  
  @property
  def reaction(self):
    """Force Property"""
    cls = type(self)
    return cls.ReactionVector[self.id]
  
  @property
  def action(self):
    """Force Property"""
    cls = type(self)
    return cls.ActionVector[self.id]
  
  @property
  def isConstrained(self):
    """Checks if DOF constrained or not"""
    return not self.constraints is None

  def addDisplacement(self, val):
    cls = type(self)
    cls.ImbalancedDisplacementVector[self.id] += val
  
  def addAction(self, val):
    cls = type(self)
    cls.ActionVector[self.id] += val
    cls.ImbalancedForceVector[self.id] += val

  def addFixedEndReaction(self, val):
    cls = type(self)
    cls.ActionVector[self.id] -= val
    cls.ImbalancedForceVector[self.id] -= val

  def addConstraint(self, DOFs:list[DOFClass], factors:list[float]):
    if self.isConstrained:
      raise Exception(f"Already Constrained DOF {self.id}")
    if self.isRestrained:
      raise Exception(f"DOF {self.id} is restrained and cannot be further constrained")
    m = np.eye(MAX_DOF, dtype=np.float64)
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
    cls.RestraintVector[self.id] = self.isRestrained

  def removeRestraint(self):
    if not self.isRestrained:
      raise Exception("No Restraint available to remove")
    cls = type(self)
    self.isRestrained = False
    cls.RestraintVector[self.id] = self.isRestrained
    cls.ImbalancedForceVector[self.id] -= cls.ReactionVector[self.id]
    cls.ReactionVector[self.id] = 0

  @classmethod
  def addStiffness(self, DOFs:list[DOFClass], K:npt.NDArray[np.float64]):
    self.StiffnessMatrix[np.ix_([_.id for _ in DOFs],[_.id for _ in DOFs])] += K

  @classmethod
  def analyse(cls):
    Kg = cls.ConstraintMatrix.T @ cls.StiffnessMatrix @ cls.ConstraintMatrix
    disp = cls.ConstraintMatrix @ cls.ImbalancedDisplacementVector
    force = cls.ConstraintMatrix.T @ cls.ImbalancedForceVector

    # Mask for all the DOFs which are not restraints
    mask = ~cls.RestraintVector

    # Filter out DOFs which have zero stiffness and zero unbalanced force
    for i, _Kg, _force in zip(range(len(mask)), Kg, force):
      if cls.RestraintVector[i]: continue
      if np.all(_Kg == 0) and np.all(_force == 0):
        mask[i] = False
      if np.all(_Kg == 0) and np.any(_force != 0):
        raise Exception(f"Instability at DOF {i}")
    
    # Do not proceed if the DOF mask is empty
    if np.any(mask):
      K11 = Kg[np.ix_(mask, mask)]
      K12 = Kg[np.ix_(mask, ~mask)]
      K21 = Kg[np.ix_(~mask, mask)]
      K22 = Kg[np.ix_(~mask, ~mask)]

      Qk = force[mask]
      Dk = disp[~mask]
      Du = np.linalg.solve(K11, Qk - K12 @ Dk)
      Qu = K21 @ Du + K22 @ Dk

      disp[mask] = Du
      force[~mask] -= Qu
      force[mask] = 0

    cls.DisplacementVector += cls.ConstraintMatrix @ disp
    cls.ReactionVector -= force
    cls.ImbalancedDisplacementVector[:] = 0
    cls.ImbalancedForceVector[:] = 0

  @classmethod
  def createCopy(cls, obj:DOFClass) -> DOFClass:
    return cls(obj.dir)