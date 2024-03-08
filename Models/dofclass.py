from __future__ import annotations
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from itertools import count
from .types import DOFTypes
from . import utils

@dataclass
class DOFClass():
  id: int = field(init=False, default_factory=count().__next__)
  represents: DOFTypes
  dir: npt.NDArray[np.float64]
  isRestrained: bool = False
  constraints: tuple[list[DOFClass],list[float]] | None = field(init=False, default=None)

  def __post_init__(self):
    if not type(self.isRestrained) is bool:
      self.isRestrained = bool(self.isRestrained)
    self.dir = utils.ensure1DNumpyArray(self.dir, np.float64, np.array([1.,0.,0.]))

  @property
  def isConstrained(self):
    """Checks if DOF constrained or not"""
    return not self.constraints is None

  def addConstraint(self, DOFs:list[DOFClass], factors:list[float]):
    if self.isConstrained:
      raise Exception(f"Already Constrained DOF {self.id}")
    if self.isRestrained:
      raise Exception(f"DOF {self.id} is restrained and cannot be further constrained")
    self.constraints = (DOFs, factors)
  
  def removeConstraint(self):
    if not self.isConstrained:
      raise Exception("No Constraint available to remove")
    self.constraints = None
  
  def addRestraint(self):
    if self.isConstrained:
      raise Exception(f"DOF {self.id} is constrained and cannot be further restrained")
    if self.isRestrained:
      raise Exception(f"Already Restrained DOF {self.id}")
    self.isRestrained = True

  def removeRestraint(self):
    if not self.isRestrained:
      raise Exception("No Restraint available to remove")
    self.isRestrained = False

