import numpy as np
import numpy.typing as npt
from . import beam1 as beam, node
from dataclasses import dataclass, field

@dataclass
class structure:
  nodes: list[node] = field(repr=False)
  beams: list[beam] = field(repr=False)
  nodes_mask: list[npt.NDArray[np.int32]] = field(init=False, default_factory=list, repr=False)
  beams_mask: list[npt.NDArray[np.int32]] = field(init=False, default_factory=list, repr=False)
  DOF: list[str] = field(init=False)
  restraint: npt.NDArray[np.bool8] = field(init=False)
  Kg: npt.NDArray[np.float64] = field(init=False)
  force: npt.NDArray[np.float64] = field(init=False)
  disp: npt.NDArray[np.float64] = field(init=False)

  def __post_init__(self):
    DOF1 = [] # unrestrained
    DOF2 = [] # restrained
    for node in self.nodes:
      DOF1.extend(filter(None, [d if ~r else None for d,r in zip(node.DOF, node.restraint)]))
      DOF2.extend(filter(None, [d if r else None for d,r in zip(node.DOF, node.restraint)]))
    self.DOF = DOF1 + DOF2
    self.restraint = np.array([False]*len(DOF1) + [True]*len(DOF2), dtype=np.bool8)
    nDOF = len(self.DOF)

    for node in self.nodes:
      self.nodes_mask.append(np.array([self.DOF.index(_) for _ in node.DOF], dtype=np.int32))
    
    for beam in self.beams:
      self.beams_mask.append(np.array([self.DOF.index(_) for _ in beam.DOF], dtype=np.int32))

    self.Kg = np.zeros((nDOF, nDOF))
    self.disp = np.zeros((nDOF, 1))
    self.force = np.zeros((nDOF, 1))
    for node, mask in zip(self.nodes, self.nodes_mask):
      self.Kg[np.ix_(mask, mask)] += node.Kg
      self.disp[mask] = node.disp
      self.force[mask] += node.force
    
    for beam, mask in zip(self.beams, self.beams_mask):
      self.Kg[np.ix_(mask, mask)] += beam.Kg
    
  def solve(self):
    mask1 = ~self.restraint
    for i, _Kg, _force in zip(range(len(mask1)), self.Kg, self.force):
      if mask1[i]:
        mask1[i] = not np.all(_Kg == 0) or not np.all(_force == 0)
    
    K11 = self.Kg[np.ix_(mask1, mask1)]
    K12 = self.Kg[np.ix_(mask1, ~mask1)]
    K21 = self.Kg[np.ix_(~mask1, mask1)]
    K22 = self.Kg[np.ix_(~mask1, ~mask1)]

    Qk = self.force[mask1].copy()
    Dk = self.disp[~mask1].copy()
    Du = np.linalg.solve(K11, Qk - K12 @ Dk)
    Qu = K21 @ Du + K22 @ Dk

    self.disp[mask1] = Du
    self.force[mask1] -= K11 @ Du + K12 @ Dk
    self.force[~mask1] -= Qu

    self.solveResults()

  
  def solveResults(self):
    for node, mask in zip(self.nodes, self.nodes_mask):
      node.solveResults(self.force[mask], self.disp[mask])

    for beam, mask in zip(self.beams, self.beams_mask):
      beam.solveResults(self.force[mask], self.disp[mask])