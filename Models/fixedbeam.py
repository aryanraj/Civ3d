import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from . import Node, DOFClass
from .utils import getAxisFromTwoNodesAndBeta, globalToLocalBasisChangeMatrix

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
    self.axis = getAxisFromTwoNodesAndBeta(self.i.coord, self.j.coord, self.beta)

    # Adding Stiffness to DOFclass
    DOFClass.addStiffness(self.DOF, self.Kg)
  
  def __del__(self):
    raise NotImplementedError(f"Deletion of {type(self).__name__} is not supported")

  @property
  def DOF(self) -> list[DOFClass]:
    return self.i.DOF + self.j.DOF

  @property
  def Kl(self) -> npt.NDArray[np.float64]:
    _L = 1/self.L
    EA_L = self.EA/self.L
    EIy_L = self.EIy/self.L
    EIz_L = self.EIz/self.L
    GJx_L = self.GJx/self.L

    # Special transformation from local axis to reduced stiffness axis
    Tlk = np.array([
      [  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
      [  0,  0,-_L,  0,  1,  0,  0,  0, _L,  0,  0,  0],
      [  0, _L,  0,  0,  0,  1,  0,-_L,  0,  0,  0,  0],
      [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
      [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
      [  0,  0,-_L,  0,  0,  0,  0,  0, _L,  0,  1,  0],
      [  0, _L,  0,  0,  0,  0,  0,-_L,  0,  0,  0,  1],
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
    ])

    # Computing local Stiffness matrix for reduced stiffness axis
    Kk = np.array([
      [ EA_L,      0,      0,     0,-EA_L,      0,      0,     0],
      [    0,4*EIy_L,      0,     0,    0,2*EIy_L,      0,     0],
      [    0,      0,4*EIz_L,     0,    0,      0,2*EIz_L,     0],
      [    0,      0,      0, GJx_L,    0,      0,      0,-GJx_L],
      [-EA_L,      0,      0,     0, EA_L,      0,      0,     0],
      [    0,2*EIy_L,      0,     0,    0,4*EIy_L,      0,     0],
      [    0,      0,2*EIz_L,     0,    0,      0,4*EIz_L,     0],
      [    0,      0,      0,-GJx_L,    0,      0,      0, GJx_L],
    ])
    return Tlk.T @ Kk @ Tlk

  @property
  def Tgl(self) -> npt.NDArray[np.float64]:
    _ = np.zeros((12,12))
    _[0:3,0:3] = globalToLocalBasisChangeMatrix(self.i.axis, self.axis)
    _[3:6,3:6] = globalToLocalBasisChangeMatrix(self.i.axis, self.axis)
    _[6:9,6:9] = globalToLocalBasisChangeMatrix(self.j.axis, self.axis)
    _[9:12,9:12] = globalToLocalBasisChangeMatrix(self.j.axis, self.axis)
    return _

  @property
  def Kg(self) -> npt.NDArray[np.float64]:
    return self.Tgl.T @ self.Kl @ self.Tgl
    
  def addLocalFEForce(self, forcel: npt.NDArray[np.float64]) -> None:
    forceg = self.Tgl.T @ forcel
    for _DOF, _force in zip(self.DOF, forceg):
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
