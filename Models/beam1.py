import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from . import node

def computePreCrossProductTransform(vec):
  return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

@dataclass
class beam1:
  id: int
  A: node
  B: node
  i: node
  j: node
  EA: float = 1
  EIy: float = 1
  EIz: float = 1
  beta: float = 0
  L: float = field(init=False)
  xij: npt.NDArray[np.float64] = field(init=False)
  yij: npt.NDArray[np.float64] = field(init=False)
  zij: npt.NDArray[np.float64] = field(init=False)
  Tgl: npt.NDArray[np.float64] = field(init=False)
  Tlk: npt.NDArray[np.float64] = field(init=False)
  Kk: npt.NDArray[np.float64] = field(init=False)
  Kg: npt.NDArray[np.float64] = field(init=False)
  DOF: list[str] = field(init=False, default_factory=list)
  disp: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.empty((12,1)))
  force: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.empty((12,1)))
  UDL: list[tuple[float]] = field(init=False, default_factory=list)
  PointLoad: list[tuple[float]] = field(init=False, default_factory=list)

  def __post_init__(self):
    # Setting up the DOF info
    self.DOF.extend(self.A.DOF)    
    self.DOF.extend(self.B.DOF)    

    # Calculating local directions
    # TODO: Add the effects of beta angle while calculating local Y axis
    self.L = np.linalg.norm(self.j.coord - self.i.coord)
    self.xij = (self.j.coord - self.i.coord)/self.L
    self.yij = self.computeLocalYAxis(self.xij)
    self.zij = np.cross(self.xij, self.yij)
    
    # Computing Transformation matrix from global to local axis
    self.Tgl = np.identity(12)
    r1 = self.i.coord - self.A.coord
    self.Tgl[0:3, 3:6] = -computePreCrossProductTransform(r1)
    r2 = self.j.coord - self.B.coord
    self.Tgl[6:9, 9:12] = -computePreCrossProductTransform(r2)
    # Rotating the matrix accoring to the direction of the local direction
    rMatrix = np.zeros(shape=(12, 12))
    for start, end in zip(range(0, 10, 3), range(3, 13, 3)):
      rMatrix[start:end, start:end] = np.array([self.xij, self.yij, self.zij])
    self.Tgl = rMatrix @ self.Tgl

    # Special transformation from local axis to reduced stiffness axis
    _L = 1/self.L
    self.Tlk = np.array([
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
    self.Kk = np.array([
      [ EA_L,      0,      0,-EA_L,      0,      0],
      [    0,4*EIy_L,      0,    0,2*EIy_L,      0],
      [    0,      0,4*EIz_L,    0,      0,2*EIz_L],
      [-EA_L,      0,      0, EA_L,      0,      0],
      [    0,2*EIy_L,      0,    0,4*EIy_L,      0],
      [    0,      0,2*EIz_L,    0,      0,4*EIz_L]
    ])

    # Computing global stiffness matrix for the element
    self.Kg = self.Tgl.T @ self.Tlk.T @ self.Kk @ self.Tlk @ self.Tgl

  def addUDL(self, dir: float, val: float):
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
    self.force += forcel
    forceg = self.Tgl.T @ forcel
    self.A.force += forceg[0:6]
    self.B.force += forceg[6:12]

  def addPointLoad(self, dir:int, val: float, dist: float):
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
    self.force += forcel
    forceg = self.Tgl.T @ forcel
    self.A.force += forceg[0:6]
    self.B.force += forceg[6:12]

  def solveResults(self, forceg, dispg):
    self.disp += self.Tgl @ dispg
    self.force += self.Tlk.T @ self.Kk @ self.Tlk @ self.Tgl @ dispg

  @staticmethod
  def computeLocalYAxis(dirX: npt.NDArray):
    return np.array([-dirX[1], dirX[0], 0])