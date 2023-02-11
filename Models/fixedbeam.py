import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from . import Node, DOFClass
from .utils import getAxisFromTwoNodesAndBeta

@dataclass
class FixedBeam:
  i: Node
  j: Node
  EA: float = 1
  EIy: float = 1
  EIz: float = 1
  GJx: float = 0
  beta: float = 0 # Angle in degrees
  
  UDL: list[tuple[float]] = field(init=False, default_factory=list)
  PointLoad: list[tuple[float]] = field(init=False, default_factory=list)
  L: float = field(init=False)
  axis: npt.NDArray[np.float64] = field(init=False)

  def __post_init__(self):
    # Total Length of the beam
    self.L = np.linalg.norm(self.j.coord - self.i.coord)

    # Calculating local directions
    # TODO: Currently it is assumed that both nodes axis are parallel to the beam
    self.axis = getAxisFromTwoNodesAndBeta(self.i.coord, self.j.coord, self.beta)

    # Adding Stiffness to DOFclass
    DOFClass.addStiffness(self.DOF, self.Kl)
  
  def __del__(self):
    raise NotImplementedError(f"Deletion of {type(self).__name__} is not supported")

  @property
  def DOF(self) -> list[DOFClass]:
    return self.i.DOF + self.j.DOF

  @property
  def Kl(self):
    # Special transformation from local axis to reduced stiffness axis
    _L = 1/self.L
    Tlk = np.array([
      [  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
      [  0,  0,-_L,  0,  1,  0,  0,  0, _L,  0,  0,  0],
      [  0, _L,  0,  0,  0,  1,  0,-_L,  0,  0,  0,  0],
      [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
      [  0,  0,-_L,  0,  0,  0,  0,  0, _L,  0,  1,  0],
      [  0, _L,  0,  0,  0,  0,  0,-_L,  0,  0,  0,  1],
    ])

    # Computing local Stiffness matrix for reduced stiffness axis
    EA_L = self.EA/self.L
    EIy_L = self.EIy/self.L
    EIz_L = self.EIz/self.L
    Kk = np.array([
      [ EA_L,      0,      0,-EA_L,      0,      0],
      [    0,4*EIy_L,      0,    0,2*EIy_L,      0],
      [    0,      0,4*EIz_L,    0,      0,2*EIz_L],
      [-EA_L,      0,      0, EA_L,      0,      0],
      [    0,2*EIy_L,      0,    0,4*EIy_L,      0],
      [    0,      0,2*EIz_L,    0,      0,4*EIz_L]
    ])
    return Tlk.T @ Kk @ Tlk

  def addLocalFEForce(self, force: npt.NDArray[np.float64]) -> None:
    for _DOF, _force in zip(self.DOF, force):
      #TODO: Transform local force to global force
      _DOF.addFixedEndReaction(_force)

  def addUDL(self, dir: float, val: float) -> None:
    self.UDL.append((dir, val))
    forcel = np.zeros((12,1))
    if dir == 0:
      forcel[0] -= val*self.L/2
      forcel[6] -= val*self.L/2
    elif dir == 1:
      forcel[1] -= val*self.L/2
      forcel[7] -= val*self.L/2
      forcel[5] -= val*self.L**2/12
      forcel[11] += val*self.L**2/12
    elif dir == 2:
      forcel[2] -= val*self.L/2
      forcel[8] -= val*self.L/2
      forcel[4] += val*self.L**2/12
      forcel[10] -= val*self.L**2/12
    self.addLocalFEForce(forcel)

  def addPointLoad(self, dir:int, val: float, dist: float) -> None:
    self.PointLoad.append((dir, val, dist))
    forcel = np.zeros((12,1))
    if dir == 0:
      forcel[0] -= val*(1 - dist/self.L)
      forcel[6] -= val*dist/self.L
    elif dir == 1:
      forcel[1] -= val*(1 - dist/self.L)
      forcel[7] -= val*dist/self.L
      a, b = dist, self.L - dist
      forcel[5] -= val*a*b**2/self.L**2
      forcel[11] += val*b*a**2/self.L**2
    elif dir == 2:
      forcel[2] -= val*(1 - dist/self.L)
      forcel[8] -= val*dist/self.L
      a, b = dist, self.L - dist
      forcel[4] += val*a*b**2/self.L**2
      forcel[10] -= val*b*a**2/self.L**2
    self.addLocalFEForce(forcel)
