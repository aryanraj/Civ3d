import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field

@dataclass
class node:
  id: int
  coord: npt.NDArray[np.float64]
  restraint: npt.NDArray[np.bool8] = field(default_factory=lambda:np.zeros((6,1), dtype=np.bool8))
  Kg: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))
  DOF: list[str] = field(init=False)
  disp: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.zeros((6,1)))
  force: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.zeros((6,1)))

  def __post_init__(self):
    if type(self.coord) is list:
      self.coord = np.array(self.coord, dtype=np.float64)
    if type(self.restraint) is list:
      self.restraint = np.array([self.restraint], dtype=np.bool8).T
    if type(self.Kg) is list:
      self.Kg = np.array(self.Kg, dtype=np.float64)
    self.DOF = [f"R{self.id}{i+1}" for i in range(6)]

  def solveResults(self, force, disp):
    self.force += force
    self.disp += disp
  
  def addRestraint(self, arg: npt.NDArray[np.bool8]):
    arg = np.array(arg, dtype=np.bool8)
    self.restraint = arg

  def addStiffness(self, arg: npt.NDArray[np.float64]):
    raise NotImplementedError("Add node stiffness while creating the object")
