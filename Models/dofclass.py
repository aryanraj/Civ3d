from __future__ import annotations
import warnings
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from typing import ClassVar
from dataclasses import dataclass, field
from itertools import count
import os
from .types import DOFTypes
from . import utils

MAX_DOF: int = int(os.getenv("MAX_DOF", 10000))
MAX_LOADCASE: int = int(os.getenv("MAX_LOADCASE", 1000))

@dataclass
class DOFClass():
  id: int = field(init=False, default_factory=count().__next__)
  represents: DOFTypes
  dir: npt.NDArray[np.float64]
  isRestrained: bool = False
  constraints: tuple[list[DOFClass],list[float]] | None = field(init=False, default=None)

  # Class Variables
  DOFList: ClassVar[list[DOFClass]] = []
  StiffnessMatrix: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,MAX_DOF), dtype=np.float64)
  MassMatrix: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,MAX_DOF), dtype=np.float64)
  ReactionVector: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,MAX_LOADCASE), dtype=np.float64)
  ActionVector: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,MAX_LOADCASE), dtype=np.float64)
  DisplacementVector: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,MAX_LOADCASE), dtype=np.float64)
  ImbalancedActionVector: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,MAX_LOADCASE), dtype=np.float64)
  ImbalancedDisplacementVector: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,MAX_LOADCASE), dtype=np.float64)
  RestraintVector: ClassVar[sp.lil_array] = sp.lil_array((MAX_DOF,1), dtype=np.bool_)

  def __post_init__(self):
    if not type(self.isRestrained) is bool:
      self.isRestrained = bool(self.isRestrained)
    self.dir = utils.ensure1DNumpyArray(self.dir, np.float64, np.array([1.,0.,0.]))
    cls = type(self)
    cls.RestraintVector[self.id,0] = self.isRestrained
    cls.DOFList.append(self)

  def __del__(self):
    raise NotImplementedError(f"Deletion of {type(self).__name__} is not supported")

  def displacement(self, loadCases:list[int]) -> npt.NDArray[np.float64]:
    """Displacement Property"""
    cls = type(self)
    return cls.DisplacementVector[[self.id],loadCases].todense()
  
  def reaction(self, loadCases:list[int]) -> npt.NDArray[np.float64]:
    """Force Property"""
    cls = type(self)
    return cls.ReactionVector[[self.id],loadCases].todense()
  
  def action(self, loadCases:list[int]) -> npt.NDArray[np.float64]:
    """Force Property"""
    cls = type(self)
    return cls.ActionVector[[self.id],loadCases].todense()
  
  @property
  def isConstrained(self):
    """Checks if DOF constrained or not"""
    return not self.constraints is None

  def addDisplacement(self, val:npt.NDArray[np.float64], loadCases:list[int]):
    cls = type(self)
    if not self.isRestrained:
      raise Exception(f"DOF {self.id} needs to be restrained before adding displacement")
    if self.isConstrained:
      raise Exception(f"Constrained DOF {self.id} can't have displacmeent added to it")
    cls.ImbalancedDisplacementVector[[self.id],loadCases] += val
  
  def addAction(self, val:npt.NDArray[np.float64], loadCases:list[int]):
    cls = type(self)
    cls.ActionVector[[self.id],loadCases] += val
    cls.ImbalancedActionVector[[self.id],loadCases] += val

  def addFixedEndReaction(self, val:npt.NDArray[np.float64], locaCases:list[int]):
    cls = type(self)
    cls.ActionVector[[self.id],locaCases] -= val
    cls.ImbalancedActionVector[[self.id],locaCases] -= val

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
    cls = type(self)
    self.isRestrained = True
    cls.RestraintVector[self.id,0] = self.isRestrained

  def removeRestraint(self):
    if not self.isRestrained:
      raise Exception("No Restraint available to remove")
    cls = type(self)
    self.isRestrained = False
    cls.RestraintVector[self.id,0] = self.isRestrained
    cls.ImbalancedActionVector[[self.id],:] -= cls.ReactionVector[[self.id],:]
    cls.ReactionVector[[self.id],:] = 0

  @classmethod
  def generateConstraintMatrix(cls) -> sp.lil_array:
    levels = np.full((len(cls.DOFList),), fill_value=-1, dtype=int)
    def DFS(parent:DOFClass):
      maxChildLevel = -1
      if parent.isConstrained:
        for child in parent.constraints[0]:
          if levels[child.id] == -1:
            DFS(child)
          maxChildLevel = max(maxChildLevel, levels[child.id])
      levels[parent.id] = maxChildLevel + 1
    for id, _level in enumerate(levels):
      if _level == -1:
        DFS(cls.DOFList[id])
    maxLevel = levels.max()
    
    constraintMatrix = sp.lil_array(sp.eye(MAX_DOF, dtype=np.float64))
    for currentLevel in range(1,maxLevel+1):
      m = sp.lil_array(sp.eye(MAX_DOF, dtype=np.float64))
      for id, _level in enumerate(levels):
        if _level == currentLevel:
          DOFs, factors = cls.DOFList[id].constraints
          m[id, id] = 0
          m[id, [_.id for _ in DOFs]] = factors
      constraintMatrix = m @ constraintMatrix
    return constraintMatrix

  @classmethod
  def addStiffness(self, DOFs:list[DOFClass], K:npt.NDArray[np.float64]):
    self.StiffnessMatrix[np.ix_([_.id for _ in DOFs],[_.id for _ in DOFs])] += K

  @classmethod
  def addMass(self, DOFs:list[DOFClass], M:npt.NDArray[np.float64]):
    self.MassMatrix[np.ix_([_.id for _ in DOFs],[_.id for _ in DOFs])] += M

  @staticmethod
  def _analyseSubDomain(stiffnessMatrix:sp.lil_array, massMatrix:sp.lil_array, constraintMatrix:sp.lil_array, restraintVector:sp.lil_array, displacement:sp.lil_array, action:sp.lil_array):
    Kg = constraintMatrix.T @ stiffnessMatrix @ constraintMatrix
    displacement = constraintMatrix @ displacement
    constrainedAction = constraintMatrix.T @ action
    reaction = sp.lil_array(constrainedAction.shape, dtype=np.float64)

    # Mask for all the DOFs which are not restraints
    freeMask = ~restraintVector.toarray().flatten()

    # Filter out DOFs which have zero stiffness and zero unbalanced force
    # TODO: Fix this part using LU_factor and LU_solve
    for i, _Kg, _force in zip(range(len(freeMask)), Kg, constrainedAction):
      if restraintVector[i,0]: continue
      if np.all(_Kg.toarray() == 0) and np.all(_force.toarray() == 0):
        freeMask[i] = False
      if np.all(_Kg.toarray() == 0) and np.any(_force.toarray() != 0):
        raise Exception(f"Instability at DOF {i}")
    
    if np.any(freeMask):
      K11 = Kg[np.ix_(freeMask, freeMask)]
      K12 = Kg[np.ix_(freeMask, ~freeMask)]
      K21 = Kg[np.ix_(~freeMask, freeMask)]
      K22 = Kg[np.ix_(~freeMask, ~freeMask)]

      Qk = constrainedAction[freeMask]
      Dk = displacement[~freeMask]
      if Dk.shape[1] == 1:
        Du = sp.lil_matrix([splinalg.spsolve(K11, Qk - K12 @ Dk)]).T # Not sure why is sp solve returning list object
      else:
        Du = splinalg.spsolve(K11, Qk - K12 @ Dk)
      Qu = K21 @ Du + K22 @ Dk

      displacement[freeMask] = Du
      reaction[~freeMask] = Qu - constrainedAction[~freeMask]
      displacement = constraintMatrix @ displacement
    else:
      K22 = Kg[np.ix_(~freeMask, ~freeMask)]
      Dk = displacement[~freeMask]
      Qu = K22 @ Dk
      reaction[~freeMask] = Qu - constrainedAction[~freeMask]

    return displacement, reaction

  @classmethod
  def _getSubDomainStructureParameters(cls, subDomainMask:npt.NDArray[np.bool_]) -> tuple[sp.lil_array, sp.lil_array, sp.lil_array, sp.lil_array]:
    constraintMatrix = cls.generateConstraintMatrix()
    stiffnessMatrix = cls.StiffnessMatrix
    massMatrix = cls.MassMatrix
    
    stiffnessMatrixSubDomain = stiffnessMatrix[np.ix_(subDomainMask, subDomainMask)]
    massMatrixSubDomain = massMatrix[np.ix_(subDomainMask, subDomainMask)]
    constraintMatrixSubDomain = constraintMatrix[np.ix_(subDomainMask, subDomainMask)]
    restraintVectorSubDomain = cls.RestraintVector[np.ix_(subDomainMask, [0])]

    # Checking if the subDomain DOFs have coupling outside the subDomain
    # Checking Coupling in ConstraintMatrix
    restraintVectorSubDomain += (\
    + (constraintMatrix[np.ix_(subDomainMask, ~subDomainMask)] != 0).sum(axis=1) \
    + (constraintMatrix[np.ix_(~subDomainMask, subDomainMask)] != 0).sum(axis=0) \
    # Checking Coupling in StiffnessMatrix
    + (stiffnessMatrix[np.ix_(subDomainMask, ~subDomainMask)] != 0).sum(axis=1) \
    + (stiffnessMatrix[np.ix_(~subDomainMask, subDomainMask)] != 0).sum(axis=0)
    ).astype(bool).reshape(restraintVectorSubDomain.shape)

    return stiffnessMatrixSubDomain, massMatrixSubDomain, constraintMatrixSubDomain, restraintVectorSubDomain

  @classmethod
  def _getSubDomainDisplacementAction(cls, subDomainMask:npt.NDArray[np.bool_], loadCaseMask:npt.NDArray[np.bool_]) -> tuple[sp.lil_array, sp.lil_array]:
    imbalancedDisplacementVectorSubDomain = cls.ImbalancedDisplacementVector[np.ix_(subDomainMask, loadCaseMask)]
    imbalancedActionVectorSubDomain = cls.ImbalancedActionVector[np.ix_(subDomainMask, loadCaseMask)]
    return imbalancedDisplacementVectorSubDomain, imbalancedActionVectorSubDomain

  @classmethod
  def analyse(cls, subDomainDOFids:list[int]=range(MAX_DOF), loadCaseSubsets:list[int]=range(MAX_LOADCASE)) -> npt.NDArray[np.float64]:
    # Creating Mask for SubDomain
    subDomainMask = np.zeros((MAX_DOF,), dtype=np.bool_)
    subDomainMask[subDomainDOFids] = True
    loadCaseMask = np.zeros((MAX_LOADCASE,), dtype=np.bool_)
    loadCaseMask[loadCaseSubsets] = True

    # Analysing the subdomain
    displacementSubDomain, reactionSubDomain = cls._analyseSubDomain(
      *cls._getSubDomainStructureParameters(subDomainMask),
      *cls._getSubDomainDisplacementAction(subDomainMask, loadCaseMask),
      )

    # Recasting the dispalcement and reaction from subdomain to global
    displacement = sp.lil_array(cls.DisplacementVector.shape, dtype=np.float64)
    displacement[np.ix_(subDomainMask, loadCaseMask)] = displacementSubDomain
    reaction = sp.lil_array(cls.ReactionVector.shape, dtype=np.float64)
    reaction[np.ix_(subDomainMask, loadCaseMask)] = reactionSubDomain

    # Storing the values for displacements and reaction in global domain
    cls.DisplacementVector += displacement
    cls.ReactionVector += reaction
    cls.ImbalancedDisplacementVector[np.ix_(subDomainMask, loadCaseMask)] = 0
    cls.ImbalancedActionVector[np.ix_(subDomainMask, loadCaseMask)] = 0
    return displacement.todense()

  @classmethod
  def resetAllActionsAndReactions(cls):
    cls.ReactionVector[:,:] = 0
    cls.ActionVector[:,:] = 0
    cls.DisplacementVector[:,:] = 0
    cls.ImbalancedActionVector[:,:] = 0
    cls.ImbalancedDisplacementVector[:,:] = 0

  @staticmethod
  def _eigSubDomain(nModes:int, DOFList:list[DOFClass], stiffnessMatrix:sp.lil_array, massMatrix:sp.lil_array, constraintMatrix:sp.lil_array, restraintVector:sp.lil_array) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    Kg = constraintMatrix.T @ stiffnessMatrix @ constraintMatrix
    Mg = constraintMatrix.T @ massMatrix @ constraintMatrix
    EigenVectors = sp.lil_matrix((constraintMatrix.shape[0], nModes))

    # Mask for all the DOFs which are not restraints
    mask = ~restraintVector.toarray().flatten()

    # Filter out DOFs which have zero stiffness
    # TODO: Fix this part using LU_factor and LU_solve
    for i, _Kg in zip(range(len(mask)), Kg):
      if restraintVector[i,0]: continue
      if np.all(_Kg.toarray() == 0):
        mask[i] = False
    
    # Do not proceed if the DOF mask is empty
    if np.any(mask):
      K11 = Kg[np.ix_(mask, mask)]
      M11 = Mg[np.ix_(mask, mask)]
      if nModes == K11.shape[0]:
        EigenValues,V = splinalg.eigsh(K11.toarray(), nModes, M11.toarray(), sigma=0)
      else:
        EigenValues,V = splinalg.eigsh(K11, nModes, M11, sigma=0)
      # TODO: Check implementation for rotational DOFs
      DispDirMatrix = np.array([list(DOF.dir) + [0]*3 if DOF.represents is DOFTypes.DISPLACEMENT else [0]*3 + list(DOF.dir) for DOF, DOFmask in zip(DOFList, mask) if DOFmask])
      ParticipationFactor = V.T @ M11 @ DispDirMatrix
      EffectiveMass = ParticipationFactor**2
      TotalMass = np.diag(DispDirMatrix.T @ M11 @ DispDirMatrix)
      MassParticipationFactor = EffectiveMass/TotalMass
      EigenVectors[mask,:] = V
      EigenVectors = constraintMatrix @ EigenVectors

    return EigenValues, EigenVectors, EffectiveMass, MassParticipationFactor

  @classmethod
  def eig(cls, nModes:int, subDomainDOFids:list[int]=range(MAX_DOF)) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Creating Mask for SubDomain
    subDomainMask = np.zeros((MAX_DOF,), dtype=np.bool_)
    subDomainMask[subDomainDOFids] = True

    # Analysing the subdomain
    EigenValues, EigenVectorsSubDomain, EffectiveMass, MassParticipationFactor = cls._eigSubDomain(
      nModes,
      [DOF for DOF, condition in zip(cls.DOFList, subDomainMask) if condition],
      *cls._getSubDomainStructureParameters(subDomainMask),
      )

    # Recasting the dispalcement and reaction from subdomain to global
    EigenVectors = np.zeros((MAX_DOF, nModes), dtype=np.float64)
    EigenVectors[subDomainMask, :] = EigenVectorsSubDomain.todense()

    return EigenValues, EigenVectors, EffectiveMass, MassParticipationFactor

  @classmethod
  def analyseTimeHistory(cls, dt:float, eigenValues:npt.NDArray[np.float64], eigenVectors:npt.NDArray[np.float64], dampingRatios:npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    eigenValues = utils.ensure1DNumpyArray(eigenValues, np.float64)
    dampingRatios = utils.ensure1DNumpyArray(dampingRatios, np.float64)

    from .dynamicAnalysis import generateTimeHistoryRecurrence
    ConstraintMatrix = cls.generateConstraintMatrix()
    constrainedAction = ConstraintMatrix.T @ cls.ImbalancedActionVector
    modalLoadVector = eigenVectors.T @ constrainedAction
    omegas = eigenValues**0.5
    modalDisplacementVector, modalVelocityVector, modalAccelerationVector = generateTimeHistoryRecurrence(modalLoadVector, dt, omegas, dampingRatios)
    return eigenVectors @ modalDisplacementVector, eigenVectors @ modalVelocityVector, eigenVectors @ modalAccelerationVector

  @classmethod
  def analyseTimeHistoryNewmark(cls, dt:float, eigenValues:npt.NDArray[np.float64], eigenVectors:npt.NDArray[np.float64], dampingRatios:npt.NDArray[np.float64], Bita:float=1./4, Gamma:float=1./2) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    eigenValues = utils.ensure1DNumpyArray(eigenValues, np.float64)
    dampingRatios = utils.ensure1DNumpyArray(dampingRatios, np.float64)

    from .dynamicAnalysis import generateTimeHistoryNewmark
    ConstraintMatrix = cls.generateConstraintMatrix()
    constrainedAction = ConstraintMatrix.T @ cls.ImbalancedActionVector
    modalLoadVector = eigenVectors.T @ constrainedAction
    omegas = eigenValues**0.5
    modalDisplacementVector, modalVelocityVector, modalAccelerationVector = generateTimeHistoryNewmark(modalLoadVector, dt, omegas, dampingRatios, Bita, Gamma)
    return eigenVectors @ modalDisplacementVector, eigenVectors @ modalVelocityVector, eigenVectors @ modalAccelerationVector

  @classmethod
  def getDisplacementVector(cls, loadCases:list[int]) -> npt.NDArray[np.float64]:
    warnings.warn("Calling todense() on DisplacementVector is very costly")
    return cls.DisplacementVector[:, loadCases].todense()

  @classmethod
  def getReactionVector(cls, loadCases:list[int]) -> npt.NDArray[np.float64]:
    warnings.warn("Calling todense() on ReactionVector is very costly")
    return cls.ReactionVector[:, loadCases].todense()

  @classmethod
  def getActionVector(cls, loadCases:list[int]) -> npt.NDArray[np.float64]:
    warnings.warn("Calling todense() on ActionVector is very costly")
    return cls.ActionVector[:, loadCases].todense()
