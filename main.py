import os
os.environ["MAX_DOF"] = str(2000)

import numpy as np
import numpy.typing as npt
from Models import DOFClass, Node, BeamSection, Beam
from Views.simpleStructure import SimpleView
from Models import utils

class Truss4Bay():
  # bottomChordSections: InitVar[list[BeamSection]] = None
  # topChordSections: InitVar[list[BeamSection]] = None
  # endRakerSection: InitVar[BeamSection] = None
  # diagonalSections: InitVar[list[BeamSection]] = None
  # vecticalSection: InitVar[BeamSection] = None

  def __init__(self, originPoint:npt.NDArray[np.float64], endTrussTopPoint:npt.NDArray[np.float64], sections:list[BeamSection]):
    self.nBottomBays:float = 4 # A bay is the length of beam between joints with diagonals.
    self.nTopBays: float = self.nBottomBays-1
    self.nBottomNodes:float = 2*self.nBottomBays+1
    self.nTopNodes:float = 2*self.nTopBays+1
    self.originPoint:npt.NDArray[np.float64] = utils.ensure1DNumpyArray(originPoint, np.float64, np.array([0,0,0]))
    endTrussTopPoint:npt.NDArray[np.float64] = utils.ensure1DNumpyArray(endTrussTopPoint, np.float64, np.array([5.905,0,7.315]))
    endTrussTopPointProjection:npt.NDArray[np.float64] = np.array([endTrussTopPoint[0], endTrussTopPoint[1], 0])
    self.panelLength:float = np.linalg.norm(endTrussTopPointProjection - self.originPoint)
    self.panelHeight:float = endTrussTopPoint[2]
    self.axis:npt.NDArray[np.float64] = utils.getAxisFromTwoNodesAndBeta(self.originPoint, endTrussTopPointProjection, 0)
    self.nodes:list[Node] = []
    self.beams:list[Beam] = []

    # Bottom Chord nodes
    self.addNodes(self.originPoint, self.axis[0], self.nBottomNodes, self.panelLength)
  
    # Top Chord nodes
    self.addNodes(endTrussTopPoint, self.axis[0], self.nTopNodes, self.panelLength)
  
    # Bottom Chord beams
    self.bottomChordBeam: list[Beam] = []
    _nodes = self.nodes[:self.nBottomNodes]
    _secs = sections[:self.nBottomBays]
    for n0, n1, n2, sec in zip(_nodes[:-2:2], _nodes[1:-1:2], _nodes[2::2], _secs):
      self.bottomChordBeam.append(Beam([n0, n1, n2], sec))
    self.beams.extend(self.bottomChordBeam)
  
    # Top Chord beams
    self.topChordBeams:list[Beam] = []
    _nodes = self.nodes[self.nBottomNodes:self.nBottomNodes+self.nTopNodes]
    _secs = sections[self.nBottomBays:self.nBottomBays+self.nTopBays]
    for n0, n1, n2, sec in zip(_nodes[:-2:2], _nodes[1:-1:2], _nodes[2::2], _secs):
      self.topChordBeams.append(Beam([n0, n1, n2], sec))
    self.beams.extend(self.topChordBeams)
  
    # Diagonal beams
    _secs = sections[self.nBottomBays+self.nTopBays:-1]
    self.diagnoalBeams = [Beam([self.nodes[int((i+1)/2)*2], self.nodes[int(i/2)*2 + self.nBottomNodes]], sec) for i,sec in zip(range(self.nBottomNodes-1), _secs)]
    self.beams.extend(self.diagnoalBeams)
  
    # Vertical beams
    self.verticalBeams = [Beam([self.nodes[n1], self.nodes[n2]], sections[-1]) for n1, n2 in zip(1+np.arange(self.nTopNodes), self.nBottomNodes+np.arange(self.nTopNodes))]
    self.beams.extend(self.verticalBeams)

  def addNodes(self, origin, dir, number, length):
    self.nodes.extend([Node(origin+dir*length*i) for i in range(number)])

  def node_by_name(self, name:str):
    if not name[0].lower() in ['l', 'u']:
      raise Exception(f"{name} is not valid node")
    if name[0].lower() == 'l':
      return self.nodes[int(name[1:])]
    else:
      return [self.nBottomNodes + int(name[1:])]

def main():
  nodes: list[Node] = []
  beams: list[Beam] = []
  sections: dict[str, BeamSection] = {
    "BottomChordL0L2": BeamSection(0.026, 0, 0.0004, 0.0014, 200_000_000_000, 0.3),
    "BottomChordL2L4": BeamSection(0.016, 0, 0.0003, 0.0008, 200_000_000_000, 0.3),
    "TopChordU1U3": BeamSection(0.0165, 0, 0.0004, 0.0007, 200_000_000_000, 0.3),
    "TopChordU3U4": BeamSection(0.0203, 0, 0.0005, 0.0008, 200_000_000_000, 0.3),
    "EndRakerL0U1": BeamSection(0.0231, 0, 0.0005, 0.0011, 200_000_000_000, 0.3),
    "PortalGirderU1": BeamSection(0.0083, 0, 0.0003, 0, 200_000_000_000, 0.3),
    "Verticals": BeamSection(0.0088, 0, 0, 0.0003, 200_000_000_000, 0.3),
    "DiagonalsU1L2": BeamSection(0.0128, 0, 0.0003, 0.0007, 200_000_000_000, 0.3),
    "DiagonalsU3L2": BeamSection(0.0093, 0, 0.0001, 0.0005, 200_000_000_000, 0.3),
    "CrossGirders": BeamSection(0.0254, 0, 0.0043, 0.0002, 200_000_000_000, 0.3),
    "SwayGirders": BeamSection(0.0038, 0, 0.0001, 0, 200_000_000_000, 0.3),
    "BottomLateralBracing": BeamSection(0.0038, 0, 0, 0, 200_000_000_000, 0.3),
    "TopLateralBracing": BeamSection(0.0036, 0, 0, 0, 200_000_000_000, 0.3),
    "StringerMain": BeamSection(0.0183, 0, 0.0019, 0.0001, 200_000_000_000, 0.3),
    "StringerCross": BeamSection(0.0064, 0, 0.0002, 0, 200_000_000_000, 0.3),
    "StringerLateralBracing": BeamSection(0.0014, 0, 0, 0, 200_000_000_000, 0.3),
    "KneeBracing": BeamSection(0.0028, 0, 0, 0, 200_000_000_000, 0.3),
  }

  truss1 = Truss4Bay([0,0,0], [5.905,0,7.315], [
    sections["BottomChordL0L2"], sections["BottomChordL2L4"], sections["BottomChordL2L4"], sections["BottomChordL0L2"],
    sections["TopChordU1U3"], sections["TopChordU3U4"], sections["TopChordU1U3"],
    sections["EndRakerL0U1"],
    sections["DiagonalsU1L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU1L2"],
    sections["EndRakerL0U1"],
    sections["Verticals"],
  ])
  truss1.node_by_name("L0").addRestraint([1,1,1,0,0,0])
  truss1.node_by_name("L8").addRestraint([0,1,1,0,0,0])
  nodes.extend(truss1.nodes)
  beams.extend(truss1.beams)
  truss2 = Truss4Bay([0,5.28,0], [5.905,5.28,7.315], [
    sections["BottomChordL0L2"], sections["BottomChordL2L4"], sections["BottomChordL2L4"], sections["BottomChordL0L2"],
    sections["TopChordU1U3"], sections["TopChordU3U4"], sections["TopChordU1U3"],
    sections["EndRakerL0U1"],
    sections["DiagonalsU1L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU1L2"],
    sections["EndRakerL0U1"],
    sections["Verticals"],
  ])
  truss2.node_by_name("L0").addRestraint([1,1,1,0,0,0])
  truss2.node_by_name("L8").addRestraint([0,1,1,0,0,0])
  nodes.extend(truss2.nodes)
  beams.extend(truss2.beams)

  # Add Load on Bottom Chord
  truss1.bottomChordBeam[0].addUDL(1,-10)
  truss1.bottomChordBeam[1].addUDL(1,-10)
  truss1.bottomChordBeam[2].addUDL(1,-10)
  truss1.bottomChordBeam[3].addUDL(1,-10)
  DOFClass.analyse()

  # Display
  SimpleView().display(nodes, beams)

  np.set_printoptions(suppress=True) # To suppress exponential notation
  print(np.hstack([DOFClass.ActionVector.toarray(), DOFClass.ReactionVector.toarray(), DOFClass.DisplacementVector.toarray()]))

if __name__ == "__main__":
  main()