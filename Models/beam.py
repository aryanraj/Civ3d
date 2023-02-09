import numpy as np
import numpy.typing as npt
from . import Node, DOFClass
from dataclasses import dataclass, field

@dataclass
class Beam:
  id: int
  i: Node
  j: Node
  EA: float = 1
  EIy: float = 1
  EIz: float = 1
  beta: float = 0 # Angle in degrees
  
  UDL: list[tuple[float]] = field(init=False, default_factory=list)
  PointLoad: list[tuple[float]] = field(init=False, default_factory=list)
  L: float = field(init=False)
  
  localDOF: list[DOFClass] = field(init=False, default_factory=list)
  DOF: list[DOFClass] = field(init=False, default_factory=list)

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

  def __post_init__(self):
    # Setting up the DOF info
    self.DOF.extend(self.i.DOF)
    self.DOF.extend(self.j.DOF)

    # Total Length of the beam
    self.L = np.linalg.norm(self.j.coord - self.i.coord)

    # Calculating local directions
    xij = (self.j.coord - self.i.coord)/self.L
    yij = computeLocalYAxis(xij, self.beta)
    zij = np.cross(xij, yij)
    self.localDOF.extend([DOFClass(_) for _ in [xij, yij, zij, xij, yij, zij]])
    self.localDOF.extend([DOFClass(_) for _ in [xij, yij, zij, xij, yij, zij]])

    # Adding Stiffness to DOFclass
    DOFClass.addStiffness(self.localDOF, self.Kl)

  def addConstraint(self, isConstrained):
    # Computing Transformation matrix from global to local axis
    rCartesianToLocal = np.zeros(shape=(12, 12))
    rGlobalToCartesian = np.zeros(shape=(12, 12))
    for start, end in zip(range(0, 10, 3), range(3, 13, 3)):
      rCartesianToLocal[start:end, start:end] = np.array([_.dir for _ in self.localDOF[start:end]])
      rGlobalToCartesian[start:end, start:end] = np.array([_.dir for _ in self.DOF[start:end]]).T
    Tgl = rCartesianToLocal @ rGlobalToCartesian
    for _isConstrained, _Tgl, _localDOF in zip(isConstrained, Tgl, self.localDOF):
      if not _isConstrained: continue
      _localDOF.addConstraint(self.DOF,_Tgl)

  def addLocalFEForce(self, force: npt.NDArray[np.float64]) -> None:
    for _localDOF, _force in zip(self.localDOF, force):
      _localDOF.addFixedEndReaction(_force)

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



def computeLocalYAxis(dirX: npt.NDArray, beta: float):
  """
  Using Rodrigues' rotation formula for performing a rotation for beta angle
  For more details goto https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
  """
  unrotated = np.array([-dirX[1], dirX[0], 0])
  K = computePreCrossProductTransform(dirX)
  R: npt.NDArray[np.float64] = np.identity(3) + np.sin(beta/(2*np.pi))*K + (1 - np.cos(beta/(2*np.pi))) * K @ K
  return unrotated @ R.T

def computePreCrossProductTransform(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
  return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
