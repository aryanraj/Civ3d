import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from . import DOFClass, Node, BeamSection 
from .utils import getAxisFromTwoNodesAndBeta, globalToLocalBasisChangeMatrix

@dataclass
class FixedBeam:
  i: Node
  j: Node
  section: BeamSection
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

    # Adding Stiffness and Mass to DOFclass
    DOFClass.addStiffness(self.DOF, self.Kg)
    DOFClass.addMass(self.DOF, self.Mg)
  
  def __del__(self):
    raise NotImplementedError(f"Deletion of {type(self).__name__} is not supported")
  
  @property
  def EA(self) -> float:
    return self.section.E * self.section.Area

  @property
  def EIy(self) -> float:
    return self.section.E * self.section.Iyy

  @property
  def EIz(self) -> float:
    return self.section.E * self.section.Izz

  @property
  def GJx(self) -> float:
    return self.section.G * self.section.Ixx

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
  def Ml(self) -> npt.NDArray[np.float64]:
    L = self.L
    L2 = L*L
    rAL = self.section.rho * self.section.Area * L
    rIxxL = self.section.rho * self.section.Ixx * L
    rAL_420 = rAL/420.
    # Mass along local X dir
    Tlk_xd = np.array([
      [  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
      [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
    ], dtype=np.float64)
    Mk_xd = rAL*np.array([
      [1/3, 1/6],
      [1/6, 1/3],
    ], dtype=np.float64)
    Tlk_xr = np.array([
      [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
    ], dtype=np.float64)
    Mk_xr = rIxxL*np.array([
      [1/3, 1/6],
      [1/6, 1/3],
    ], dtype=np.float64)
    #Mass in local Y/Z dir
    Tlk_y = np.array([
      [  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
      [  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
      [  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
    ], dtype=np.float64)
    Tlk_z = np.array([
      [  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
      [  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
      [  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0],
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0],
    ], dtype=np.float64)
    Mk_yz = rAL_420*np.array([
      [  156,   54, 22*L,-13*L],
      [   54,  156, 13*L,-22*L],
      [ 22*L, 13*L, 4*L2,-3*L2],
      [-13*L,-22*L,-3*L2, 4*L2],
    ], dtype=np.float64)
    return Tlk_xd.T @ Mk_xd @ Tlk_xd + Tlk_xr.T @ Mk_xr @ Tlk_xr + Tlk_y.T @ Mk_yz @ Tlk_y + Tlk_z.T @ Mk_yz @ Tlk_z

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
    
  @property
  def Mg(self) -> npt.NDArray[np.float64]:
    return self.Tgl.T @ self.Ml @ self.Tgl

  def addLocalFEForce(self, forcel: npt.NDArray[np.float64]) -> None:
    forceg = self.Tgl.T @ forcel
    for _DOF, _force in zip(self.DOF, forceg):
      _DOF.addFixedEndReaction(_force)

  def addUDL(self, dir: int, val: float) -> None:
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

  def addSelfWeight(self, dir:int, factor:float=-1) -> None:
    udl = self.section.Area * self.section.rhog * factor
    if dir == 0:
      globalDir = np.array([1,0,0])
    elif dir == 1:
      globalDir = np.array([0,1,0])
    else:
      globalDir = np.array([0,0,1])
    self.addUDL(0, udl*np.dot(globalDir, self.axis[0]))
    self.addUDL(1, udl*np.dot(globalDir, self.axis[1]))
    self.addUDL(2, udl*np.dot(globalDir, self.axis[2]))