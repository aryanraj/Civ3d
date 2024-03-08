from __future__ import annotations
import warnings
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from dataclasses import dataclass, field
import os
from .types import DOFTypes
from . import DOFClass
from . import utils

MAX_DOF: int = int(os.getenv("MAX_DOF", 10000))
MAX_LOADCASE: int = int(os.getenv("MAX_LOADCASE", 1000))

def matrix_field(shape, dtype):
  return field(init=False, default_factory=lambda: sp.lil_array(shape, dtype=dtype))

@dataclass
class Domain():
  maxDOF: int = MAX_DOF
  maxLoadcase: int = MAX_LOADCASE
  DOFList: list[DOFClass] = []
  # TODO: Changed MAX_DOF and MAX_LOADCASE will not be accesible :disappointed:
  StiffnessMatrix: sp.lil_array = matrix_field((MAX_DOF,MAX_DOF), dtype=np.float64)
  MassMatrix: sp.lil_array = matrix_field((MAX_DOF,MAX_DOF), dtype=np.float64)
  ReactionVector: sp.lil_array = matrix_field((MAX_DOF,MAX_LOADCASE), dtype=np.float64)
  ActionVector: sp.lil_array = matrix_field((MAX_DOF,MAX_LOADCASE), dtype=np.float64)
  DisplacementVector: sp.lil_array = matrix_field((MAX_DOF,MAX_LOADCASE), dtype=np.float64)
  ImbalancedActionVector: sp.lil_array = matrix_field((MAX_DOF,MAX_LOADCASE), dtype=np.float64)
  ImbalancedDisplacementVector: sp.lil_array = matrix_field((MAX_DOF,MAX_LOADCASE), dtype=np.float64)
  # Generate restraint from DOFs like ConstraintMatrix
  # RestraintVector: sp.lil_array = matrix_field((MAX_DOF,1), dtype=np.bool_)

  def createDOF(self, *args, **kargs):
    newDOF =  DOFClass(*args, **kargs)
    self.DOFList.append(newDOF)

  def displacement(self, DOF: DOFClass, loadCases:list[int]) -> npt.NDArray[np.float64]:
    """Displacement Property"""
    return self.DisplacementVector[[DOF.id],loadCases].todense()
  
  def reaction(self, DOF: DOFClass, loadCases:list[int]) -> npt.NDArray[np.float64]:
    """Force Property"""
    return self.ReactionVector[[DOF.id],loadCases].todense()
  
  def action(self, DOF: DOFClass, loadCases:list[int]) -> npt.NDArray[np.float64]:
    """ Force Property """
    return self.ActionVector[[DOF.id],loadCases].todense()

  def addDisplacement(self, DOF: DOFClass, val:npt.NDArray[np.float64], loadCases:list[int]):
    if not DOF.isRestrained:
      raise Exception(f"DOF {DOF.id} needs to be restrained before adding displacement")
    if DOF.isConstrained:
      raise Exception(f"Constrained DOF {DOF.id} can't have displacmeent added to it")
    self.ImbalancedDisplacementVector[[DOF.id], loadCases] += val

  def addAction(self, DOF:DOFClass, val:npt.NDArray[np.float64], loadCases:list[int]):
    self.ActionVector[[DOF.id],loadCases] += val
    self.ImbalancedActionVector[[DOF.id],loadCases] += val

  def addFixedEndReaction(self, DOF:DOFClass, val:npt.NDArray[np.float64], locaCases:list[int]):
    self.ActionVector[[DOF.id],locaCases] -= val
    self.ImbalancedActionVector[[DOF.id],locaCases] -= val

  # TODO: Reconsider how the restraint and constraint are configured. Should
  # they be part of DOFClass or Domain?
  def addRestraint(self, DOF:DOFClass):
    if DOF.isConstrained:
      raise Exception(f"DOF {DOF.id} is constrained and cannot be further restrained")
    if DOF.isRestrained:
      raise Exception(f"Already Restrained DOF {DOF.id}")
    DOF.isRestrained = True

  def removeRestraint(self, DOF:DOFClass):
    if not DOF.isRestrained:
      raise Exception("No Restraint available to remove")
    DOF.isRestrained = False
    self.ImbalancedActionVector[[DOF.id],:] -= self.ReactionVector[[DOF.id],:]
    self.ReactionVector[[DOF.id],:] = 0

  def generateConstraintMatrix(self) -> sp.lil_array:
    levels = np.full((len(self.DOFList),), fill_value=-1, dtype=int)

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
        DFS(self.DOFList[id])
    maxLevel = levels.max()
    
    constraintMatrix = sp.lil_array(sp.eye(self.maxDOF, dtype=float))
    for currentLevel in range(1,maxLevel+1):
      m = sp.lil_array(sp.eye(self.maxDOF, dtype=float))
      for id, _level in enumerate(levels):
        if _level == currentLevel:
          DOFs, factors = self.DOFList[id].constraints
          m[id, id] = 0
          m[id, [_.id for _ in DOFs]] = factors
      constraintMatrix = m @ constraintMatrix
    return constraintMatrix

  def generateRestraintVector(self) -> sp.lil_array:
    ...

  def addStiffness(self, DOFs:list[DOFClass], K:npt.NDArray[np.float64]):
    self.StiffnessMatrix[np.ix_([_.id for _ in DOFs],[_.id for _ in DOFs])] += K

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

  def _getSubDomainStructureParameters(self, subDomainMask:npt.NDArray[np.bool_]) -> tuple[sp.lil_array, sp.lil_array, sp.lil_array, sp.lil_array]:
    constraintMatrix = self.generateConstraintMatrix()
    stiffnessMatrix = self.StiffnessMatrix
    massMatrix = self.MassMatrix
    
    stiffnessMatrixSubDomain = stiffnessMatrix[np.ix_(subDomainMask, subDomainMask)]
    massMatrixSubDomain = massMatrix[np.ix_(subDomainMask, subDomainMask)]
    constraintMatrixSubDomain = constraintMatrix[np.ix_(subDomainMask, subDomainMask)]
    restraintVectorSubDomain = self.RestraintVector[np.ix_(subDomainMask, [0])]

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

  def _getSubDomainDisplacementAction(self, subDomainMask:npt.NDArray[np.bool_], loadCaseMask:npt.NDArray[np.bool_]) -> tuple[sp.lil_array, sp.lil_array]:
    imbalancedDisplacementVectorSubDomain = self.ImbalancedDisplacementVector[np.ix_(subDomainMask, loadCaseMask)]
    imbalancedActionVectorSubDomain = self.ImbalancedActionVector[np.ix_(subDomainMask, loadCaseMask)]
    return imbalancedDisplacementVectorSubDomain, imbalancedActionVectorSubDomain

  def analyse(self, subDomainDOFids:list[int]|None = None, loadCaseSubsets:list[int]|None = None) -> npt.NDArray[np.float64]:
    subDomainDOFids = subDomainDOFids or [*range(self.maxDOF)]
    loadCaseSubsets = loadCaseSubsets or [*range(self.maxLoadcase)]

    # Creating Mask for SubDomain
    subDomainMask = np.zeros((self.maxDOF,), dtype=np.bool_)
    subDomainMask[subDomainDOFids] = True
    loadCaseMask = np.zeros((self.maxLoadcase,), dtype=np.bool_)
    loadCaseMask[loadCaseSubsets] = True

    # Analysing the subdomain
    displacementSubDomain, reactionSubDomain = self._analyseSubDomain(
      *self._getSubDomainStructureParameters(subDomainMask),
      *self._getSubDomainDisplacementAction(subDomainMask, loadCaseMask),
      )

    # Recasting the dispalcement and reaction from subdomain to global
    displacement = sp.lil_array(self.DisplacementVector.shape, dtype=np.float64)
    displacement[np.ix_(subDomainMask, loadCaseMask)] = displacementSubDomain
    reaction = sp.lil_array(self.ReactionVector.shape, dtype=np.float64)
    reaction[np.ix_(subDomainMask, loadCaseMask)] = reactionSubDomain

    # Storing the values for displacements and reaction in global domain
    self.DisplacementVector += displacement
    self.ReactionVector += reaction
    self.ImbalancedDisplacementVector[np.ix_(subDomainMask, loadCaseMask)] = 0
    self.ImbalancedActionVector[np.ix_(subDomainMask, loadCaseMask)] = 0
    return displacement.todense()

  def resetAllActionsAndReactions(self):
    self.ReactionVector[:,:] = 0
    self.ActionVector[:,:] = 0
    self.DisplacementVector[:,:] = 0
    self.ImbalancedActionVector[:,:] = 0
    self.ImbalancedDisplacementVector[:,:] = 0

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

  def eig(self, nModes:int, subDomainDOFids:list[int]|None = None) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    subDomainDOFids = subDomainDOFids or [*range(self.maxDOF)]
    # Creating Mask for SubDomain
    subDomainMask = np.zeros((self.maxDOF,), dtype=np.bool_)
    subDomainMask[subDomainDOFids] = True

    # Analysing the subdomain
    EigenValues, EigenVectorsSubDomain, EffectiveMass, MassParticipationFactor = self._eigSubDomain(
      nModes,
      [DOF for DOF, condition in zip(self.DOFList, subDomainMask) if condition],
      *self._getSubDomainStructureParameters(subDomainMask),
      )

    # Recasting the dispalcement and reaction from subdomain to global
    EigenVectors = np.zeros((self.maxDOF, nModes), dtype=np.float64)
    EigenVectors[subDomainMask, :] = EigenVectorsSubDomain.todense()

    return EigenValues, EigenVectors, EffectiveMass, MassParticipationFactor

  def analyseTimeHistory(self, dt:float, eigenValues:npt.NDArray[np.float64], eigenVectors:npt.NDArray[np.float64], dampingRatios:npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    eigenValues = utils.ensure1DNumpyArray(eigenValues, np.float64)
    dampingRatios = utils.ensure1DNumpyArray(dampingRatios, np.float64)

    from .dynamicAnalysis import generateTimeHistoryRecurrence
    ConstraintMatrix = self.generateConstraintMatrix()
    constrainedAction = ConstraintMatrix.T @ self.ImbalancedActionVector
    modalLoadVector = eigenVectors.T @ constrainedAction
    omegas = eigenValues**0.5
    modalDisplacementVector, modalVelocityVector, modalAccelerationVector = generateTimeHistoryRecurrence(modalLoadVector, dt, omegas, dampingRatios)
    return eigenVectors @ modalDisplacementVector, eigenVectors @ modalVelocityVector, eigenVectors @ modalAccelerationVector

  def analyseTimeHistoryNewmark(self, dt:float, eigenValues:npt.NDArray[np.float64], eigenVectors:npt.NDArray[np.float64], dampingRatios:npt.NDArray[np.float64], Bita:float=1./4, Gamma:float=1./2) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    eigenValues = utils.ensure1DNumpyArray(eigenValues, np.float64)
    dampingRatios = utils.ensure1DNumpyArray(dampingRatios, np.float64)

    from .dynamicAnalysis import generateTimeHistoryNewmark
    ConstraintMatrix = self.generateConstraintMatrix()
    constrainedAction = ConstraintMatrix.T @ self.ImbalancedActionVector
    modalLoadVector = eigenVectors.T @ constrainedAction
    omegas = eigenValues**0.5
    modalDisplacementVector, modalVelocityVector, modalAccelerationVector = generateTimeHistoryNewmark(modalLoadVector, dt, omegas, dampingRatios, Bita, Gamma)
    return eigenVectors @ modalDisplacementVector, eigenVectors @ modalVelocityVector, eigenVectors @ modalAccelerationVector

  def getDisplacementVector(self, loadCases:list[int]) -> npt.NDArray[np.float64]:
    warnings.warn("Calling todense() on DisplacementVector is very costly")
    return self.DisplacementVector[:, loadCases].todense()

  def getReactionVector(self, loadCases:list[int]) -> npt.NDArray[np.float64]:
    warnings.warn("Calling todense() on ReactionVector is very costly")
    return self.ReactionVector[:, loadCases].todense()

  def getActionVector(self, loadCases:list[int]) -> npt.NDArray[np.float64]:
    warnings.warn("Calling todense() on ActionVector is very costly")
    return self.ActionVector[:, loadCases].todense()

