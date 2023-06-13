from typing import Union
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from . import DOFClass, Node, BeamSection 
from .utils import getAxisFromTwoNodesAndBeta, globalToLocalBasisChangeMatrix, ensure1DNumpyArray

@dataclass
class FixedBeam:
  i: Node
  j: Node
  section: BeamSection
  beta: float = 0 # Angle in degrees
  
  UDL: list[tuple[int, npt.NDArray[np.float64], list[int]]] = field(init=False, default_factory=list)
  PointLoad: list[tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64], list[int]]] = field(init=False, default_factory=list)
  AdditionalMassUDL: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.zeros((3,)))
  L: float = field(init=False)
  axis: npt.NDArray[np.float64] = field(init=False)
  stiffnessFactor: float = field(init=False, default=1.)

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
    rA = self.section.rho * self.section.Area
    rIxx = self.section.rho * self.section.Ixx
    return self.getMassMatrixUDL(L, rA, rA, rA, rIxx)

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

  def addLocalFEForce(self, forcel:npt.NDArray[np.float64], loadCases:list[int]) -> None:
    if not type(forcel) is np.ndarray or forcel.ndim != 2:
      raise Exception("The forcel should be an NDArray with ndim=2")
    if forcel.shape[1] != 1 and forcel.shape[1] != len(loadCases):
      raise Exception(f"The forcel matrix should have number of columns either 1 OR {len(loadCases)=}")
    if forcel.shape[0] != self.Tgl.shape[0]:
      raise Exception(f"The forcel matrix should have number of rows same as {self.Tgl.shape[0]=}")
    forceg = self.Tgl.T @ forcel
    for _DOF, _force in zip(self.DOF, forceg):
      _DOF.addFixedEndReaction(np.array([_force]), loadCases)

  def addUDL(self, dir:int, val:npt.NDArray[np.float64], loadCases:list[int]) -> None:
    if not type(val) is np.ndarray or val.ndim == 0:
      val = np.array([val])
    if val.ndim != 1:
      raise Exception("The dimension of val must be 1")
    self.UDL.append((dir, val, loadCases))
    forcel = np.zeros((12, val.shape[0]))
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
    self.addLocalFEForce(forcel, loadCases)

  def addPointLoad(self, dir:int, val:npt.NDArray[np.float64], dist:npt.NDArray[np.float64], loadCases:list[int]) -> None:
    if not type(val) is np.ndarray or val.ndim == 0:
      val = np.array([val])
    val = np.array(val, dtype=np.float64)
    if not type(dist) is np.ndarray or dist.ndim == 0:
      dist = np.array([dist])
    dist = np.array(dist, dtype=np.float64)
    if dist.ndim != 1 or val.ndim != 1:
      raise Exception("The dimension of both val and dist must be 1")
    self.PointLoad.append((dir, val, dist, loadCases))
    forcel = np.zeros((12,val.shape[0]))
    if dir == 0:
      forcel[0] -= val*(1 - dist/self.L)
      forcel[6] -= val*dist/self.L
    elif dir == 1:
      a, b = dist, self.L - dist
      forcel[1] -= val*b**2*(3*a+b)/self.L**3
      forcel[7] -= val*a**2*(a+3*b)/self.L**3
      forcel[5] -= val*a*b**2/self.L**2
      forcel[11] += val*b*a**2/self.L**2
    elif dir == 2:
      a, b = dist, self.L - dist
      forcel[2] -= val*b**2*(3*a+b)/self.L**3
      forcel[8] -= val*a**2*(a+3*b)/self.L**3
      forcel[4] += val*a*b**2/self.L**2
      forcel[10] -= val*b*a**2/self.L**2
    self.addLocalFEForce(forcel, loadCases)

  def addSelfWeight(self, dir:int, factor:float, loadCases:list[int]) -> None:
    udl = (self.section.Area * self.section.rhog + self.AdditionalMassUDL[dir] * 9.806) * factor
    if dir == 0:
      globalDir = np.array([1,0,0])
    elif dir == 1:
      globalDir = np.array([0,1,0])
    else:
      globalDir = np.array([0,0,1])
    self.addUDL(0, udl*np.dot(globalDir, self.axis[0]), loadCases)
    self.addUDL(1, udl*np.dot(globalDir, self.axis[1]), loadCases)
    self.addUDL(2, udl*np.dot(globalDir, self.axis[2]), loadCases)

  def addMassUDL(self, massUDL:Union[float, npt.NDArray[np.float64]]) -> None:
    if type(massUDL) is float:
      massUDL = np.ones((3,))*massUDL
    massUDL = ensure1DNumpyArray(massUDL, np.float64, np.zeros((3,)))
    self.AdditionalMassUDL += massUDL
    massMatrix = self.getMassMatrixUDL(self.L, *massUDL, 0)
    DOFClass.addMass(self.DOF, massMatrix)

  def setAdditionalMassUDL(self, additionalMassUDL:Union[float, npt.NDArray[np.float64]]) -> None:
    if type(additionalMassUDL) is float:
      additionalMassUDL = np.ones((3,))*additionalMassUDL
    additionalMassUDL = ensure1DNumpyArray(additionalMassUDL, np.float64, np.zeros((3,)))
    self.addMassUDL(additionalMassUDL - self.AdditionalMassUDL)
  
  def setAdditionalMassFactor(self, selfWeightFactor:Union[float, npt.NDArray[np.float64]]) -> None:
    if type(selfWeightFactor) is float:
      selfWeightFactor = np.ones((3,))*selfWeightFactor
    selfWeightFactor = ensure1DNumpyArray(selfWeightFactor, np.float64, np.zeros((3,)))
    additionalMassUDL = selfWeightFactor * self.section.Area * self.section.rho
    self.setAdditionalMassUDL(additionalMassUDL)

  def setStiffnessFactor(self, stiffnessFactor:float) -> None:
    DOFClass.addStiffness(self.DOF, self.Kg*(stiffnessFactor - self.stiffnessFactor))
    self.stiffnessFactor = stiffnessFactor
  
  def getAxialStrainForGlobalDisplacement(self, displacementVector:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    _displacementVector = self.Tgl @ displacementVector[[_.id for _ in self.DOF]]
    return (_displacementVector[6] - _displacementVector[0])/self.L
  
  def getAxialStrainForLoadCases(self, loadCases:list[int]) -> npt.NDArray[np.float64]:
    _displacementVector = self.Tgl @ np.array([_.displacement(loadCases).flatten() for _ in self.DOF])
    return (_displacementVector[6] - _displacementVector[0])/self.L

  @staticmethod
  def getMassMatrixUDL(L:float, rAx:float, rAy:float, rAz:float, rIxx:float=0) -> npt.NDArray[np.float64]:
    L2 = L*L
    rAxL = rAx * L
    rIxxL = rIxx * L
    rAyL_420 = rAy*L/420.
    rAzL_420 = rAz*L/420.
    # Mass along local X dir
    Tlk_xd = np.array([
      [  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
      [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
    ], dtype=np.float64)
    Mk_xd = rAxL*np.array([
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
    Mk_y = rAyL_420*np.array([
      [  156,   54, 22*L,-13*L],
      [   54,  156, 13*L,-22*L],
      [ 22*L, 13*L, 4*L2,-3*L2],
      [-13*L,-22*L,-3*L2, 4*L2],
    ], dtype=np.float64)
    Mk_z = rAzL_420*np.array([
      [  156,   54, 22*L,-13*L],
      [   54,  156, 13*L,-22*L],
      [ 22*L, 13*L, 4*L2,-3*L2],
      [-13*L,-22*L,-3*L2, 4*L2],
    ], dtype=np.float64)
    return Tlk_xd.T @ Mk_xd @ Tlk_xd + Tlk_xr.T @ Mk_xr @ Tlk_xr + Tlk_y.T @ Mk_y @ Tlk_y + Tlk_z.T @ Mk_z @ Tlk_z
