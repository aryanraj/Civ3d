import os
os.environ["MAX_DOF"] = str(3000)

import numpy as np
import numpy.typing as npt
from Models import DOFClass, Node, BeamSection, Beam
from Views.simpleStructure import SimpleView
from Models import utils

class Truss4Bay():
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
    # TODO: Relate this with the section depth
    self.diagonalBeams: list[Beam] = []
    _secs = sections[self.nBottomBays+self.nTopBays:-1]
    _n1s, _n2s = zip(*[(int((i+1)/2)*2, int(i/2)*2 + self.nBottomNodes) for i in range(self.nBottomNodes-1)])
    for n1, n2, _sec in zip(_n1s, _n2s, _secs):
      nodeA, nodeB = self.nodes[n1], self.nodes[n2]
      axis = utils.getAxisFromTwoNodesAndBeta(nodeA.coord, nodeB.coord, 0)
      bottomVertOffset = 0.8
      n0 = Node(nodeA.coord + axis[0]*bottomVertOffset/axis[0,2])
      topVertOffset = -0.28
      n2 = Node(nodeB.coord + axis[0]*topVertOffset/axis[0,2])
      topVertOffset += -1.25
      n1 = Node(nodeB.coord + axis[0]*topVertOffset/axis[0,2])
      self.nodes.extend([n0,n1,n2])
      self.diagonalBeams.append(Beam([n0,n1,n2], _sec, A=nodeA, B=nodeB))
    self.beams.extend(self.diagonalBeams)
  
    # Vertical beams
    # TODO: Relate this with the section depth
    self.verticalBeams: list[Beam] = []
    _n1s = 1+np.arange(self.nTopNodes, dtype=int)
    _n2s = self.nBottomNodes+np.arange(self.nTopNodes, dtype=int)
    for n1, n2 in zip(_n1s, _n2s):
      nodeA, nodeB = self.nodes[n1], self.nodes[n2]
      axis = utils.getAxisFromTwoNodesAndBeta(nodeA.coord, nodeB.coord, 0)
      bottomVertOffset = 0.8
      n0 = Node(nodeA.coord + axis[0]*bottomVertOffset/axis[0,2])
      topVertOffset = -0.28
      n2 = Node(nodeB.coord + axis[0]*topVertOffset/axis[0,2])
      topVertOffset += -1.25
      n1 = Node(nodeB.coord + axis[0]*topVertOffset/axis[0,2])
      self.nodes.extend([n0,n1,n2])
      self.verticalBeams.append(Beam([n0,n1,n2], sections[-1], A=nodeA, B=nodeB))
    self.beams.extend(self.verticalBeams)

  def addNodes(self, origin, dir, number, length):
    self.nodes.extend([Node(origin+dir*length*i) for i in range(number)])

  def node_by_name(self, name:str):
    if not name[0].lower() in ['l', 'u']:
      raise Exception(f"{name} is not valid node")
    if name[0].lower() == 'l':
      return self.nodes[int(name[1:])]
    else:
      return self.nodes[self.nBottomNodes + int(name[1:])-1]

class StringerAssembly:
  def __init__(self, crossGirderA:Beam, crossGirderB:Beam, crossBeamSpacing:float, verticalOffset:float, sections:list[BeamSection]):
    self.nodes:list[Node] = []
    self.beams:list[Beam] = []

    self.main:list[Beam] = []
    for i in [1,2]:
      nodeA = crossGirderA.nodes[i]
      nodeB = crossGirderB.nodes[i]
      axis = utils.getAxisFromTwoNodesAndBeta(nodeA.coord, nodeB.coord, 0)
      n0 = Node(nodeA.coord + verticalOffset*axis[2])
      n3 = Node(nodeB.coord + verticalOffset*axis[2])
      n1 = Node((n0.coord + n3.coord)/2 - crossBeamSpacing/2*axis[0])
      n2 = Node((n0.coord + n3.coord)/2 + crossBeamSpacing/2*axis[0])
      self.main.append(Beam([n0, n1, n2, n3], sections[0], A=nodeA, B=nodeB, constraintsA=[1,1,1,1,0,1], constraintsB=[1,1,1,1,0,1]))
      self.nodes.extend(self.main[-1].nodes)
      self.beams.append(self.main[-1])

    self.cross:list[Beam] = []
    self.cross.append(Beam([self.main[0].nodes[1], self.main[1].nodes[1]], sections[1], constraintsA=[1,1,1,1,0,0], constraintsB=[1,1,1,1,0,0]))
    self.cross.append(Beam([self.main[0].nodes[2], self.main[1].nodes[2]], sections[1], constraintsA=[1,1,1,1,0,0], constraintsB=[1,1,1,1,0,0]))
    self.beams.extend(self.cross)

    self.lateral:list[Beam] = []
    self.lateral.append(Beam([Node(self.main[0].nodes[0].coord), self.main[1].nodes[1]], sections[2], A=self.main[0].A, constraintsA=[1,1,1,1,0,0], constraintsB=[1,1,1,1,0,0]))
    self.lateral.append(Beam([self.main[1].nodes[1], self.main[0].nodes[2]], sections[2], constraintsA=[1,1,1,1,0,0], constraintsB=[1,1,1,1,0,0]))
    self.lateral.append(Beam([self.main[0].nodes[2], Node(self.main[1].nodes[-1].coord)], sections[2], B=self.main[1].B, constraintsA=[1,1,1,1,0,0], constraintsB=[1,1,1,1,0,0]))
    self.beams.extend(self.lateral)

def createTrussCrossMembers(nodeA:Node, nodeB:Node, section:BeamSection, addlNodeDist:float, nodeOffset:float=0.25, beta:float=0., verticalOffset:float=0., endsPinned=False) -> Beam:
  # TODO: Relate this with the section width
  axis = utils.getAxisFromTwoNodesAndBeta(nodeA.coord, nodeB.coord, beta)
  n0 = Node(nodeA.coord + axis[0]*nodeOffset + axis[2]*verticalOffset)
  n1 = Node(nodeA.coord + (addlNodeDist+nodeOffset)*axis[0] + axis[2]*verticalOffset)
  n2 = Node(nodeB.coord - (addlNodeDist+nodeOffset)*axis[0] + axis[2]*verticalOffset)
  n3 = Node(nodeB.coord - axis[0]*nodeOffset + axis[2]*verticalOffset)
  b = Beam([n0, n1, n2, n3], section, beta, A=nodeA, B=nodeB)
  if endsPinned:
    b.addEndConstrains([1,1,1,1,0,1], [1,1,1,1,0,1])
  return b

def createKneeBracings(truss1Beam:Beam, truss2Beam:Beam, crossMember:Beam, section:BeamSection, Aoffset:float, Boffset:float) -> list[Beam]:
  # TODO: Relate this with the section width
  axis = crossMember.childBeams[0].axis
  # First KneeBracing 
  nodeA = truss1Beam.nodes[-2]
  nodeB = crossMember.nodes[1]
  n0coord = nodeA.coord + Aoffset * axis[0]
  n1coord = nodeB.coord - Boffset * axis[2]
  baxis = utils.getAxisFromTwoNodesAndBeta(n0coord, n1coord, 0)
  n0 = Node(n0coord, axis=baxis)
  n1 = Node(n1coord, axis=baxis)
  b1 = Beam([n0, n1], section, A=nodeA, B=nodeB)
  b1.addEndConstrains([1,1,1,1,0,1], [1,1,1,1,0,1])
  # First KneeBracing 
  nodeA = truss2Beam.nodes[-2]
  nodeB = crossMember.nodes[-2]
  n0coord = nodeA.coord - Aoffset * axis[0]
  n1coord = nodeB.coord - Boffset * axis[2]
  baxis = utils.getAxisFromTwoNodesAndBeta(n0coord, n1coord, 0)
  n0 = Node(n0coord, axis=baxis)
  n1 = Node(n1coord, axis=baxis)
  b2 = Beam([n0, n1], section, A=nodeA, B=nodeB)
  b2.addEndConstrains([1,1,1,1,0,1], [1,1,1,1,0,1])
  return [b1, b2]

def createLaterals(nodeA1:Node, nodeA2:Node, nodeB1:Node, nodeB2:Node, section:BeamSection, yoffset:float) -> list[Beam]:
  # Assuming that the nodes given in arguments form a rectangle
  ncenter = Node((nodeA1.coord+nodeB2.coord)/2)
  axisA1A2 = utils.getAxisFromTwoNodesAndBeta(nodeA1.coord, nodeA2.coord, 0)
  axisA1B2 = utils.getAxisFromTwoNodesAndBeta(nodeA1.coord, nodeB2.coord, 0)
  axisA2B1 = utils.getAxisFromTwoNodesAndBeta(nodeA2.coord, nodeB1.coord, 0)
  xoffset = yoffset/np.dot(axisA1B2[0], axisA1A2[1])
  # First beam
  n0 = Node(nodeA1.coord + axisA1B2[0]*xoffset)
  n1 = Node(nodeB2.coord - axisA1B2[0]*xoffset)
  b1 = Beam([n0,ncenter,n1], section, A=nodeA1, B=nodeB2)
  b1.addEndConstrains([1,1,1,1,0,0], [1,1,1,1,0,0])
  # Second beam
  n0 = Node(nodeA2.coord + axisA2B1[0]*xoffset)
  n1 = Node(nodeB1.coord - axisA2B1[0]*xoffset)
  b2 = Beam([n0,ncenter,n1], section, A=nodeA2, B=nodeB1)
  b2.addEndConstrains([1,1,1,1,0,0], [1,1,1,1,0,0])
  return [b1,b2]

if __name__ == "__main__":
  nodes: list[Node] = []
  beams: list[Beam] = []
  # Increase the density and mass to simulate the additional loads on the members
  sections: dict[str, BeamSection] = {
    "BottomChordL0L2": BeamSection(0.0259569, 0.0000079543, 0.000423932, 0.00136674, 200_000_000_000, 0.3, 9946, 97530.476),
    "BottomChordL2L4": BeamSection(0.0159569, 0.00000323187, 0.000310969, 0.000839274, 200_000_000_000, 0.3, 9946, 97530.476),
    "TopChordU1U3": BeamSection(0.0164833, 0.00000499706, 0.000424139, 0.000729281, 200_000_000_000, 0.3, 9946, 97530.476),
    "TopChordU3U4": BeamSection(0.0202769, 0.00000499706, 0.000508988, 0.000799148, 200_000_000_000, 0.3, 9946, 97530.476),
    "EndRakerL0U1": BeamSection(0.0231169, 0.00000504102, 0.000495082, 0.0010982, 200_000_000_000, 0.3, 9946, 97530.476),
    "PortalGirderU1": BeamSection(0.0083, 0.000000413677, 0.000259889, 0.0000133692, 200_000_000_000, 0.3, 9946, 97530.476),
    "Verticals": BeamSection(0.0088, 0.000000444356, 0.0000133733, 0.000332293, 200_000_000_000, 0.3, 9946, 97530.476),
    "DiagonalsU1L2": BeamSection(0.0127569, 0.0000026403, 0.000304142, 0.000660718, 200_000_000_000, 0.3, 9946, 97530.476),
    "DiagonalsU3L2": BeamSection(0.0092583, 0.0000026403, 0.000128362, 0.00048099, 200_000_000_000, 0.3, 9946, 97530.476),
    "CrossGirders": BeamSection(0.02535, 0.00000431878, 0.0043298, 0.000213411, 200_000_000_000, 0.3, 9946, 97530.476),
    "SwayGirders": BeamSection(0.00384487, 0.000000148625, 0.000144009, 0.00000434948, 200_000_000_000, 0.3, 9946, 97530.476),
    "BottomLateralBracing": BeamSection(0.00383182, 0.000000140658, 0.00000792087, 0.00000792087, 200_000_000_000, 0.3, 9946, 97530.476),
    "TopLateralBracing": BeamSection(0.00360566, 0.000000213409, 0.0000130409, 0.00000350786, 200_000_000_000, 0.3, 9946, 97530.476),
    "StringerMain": BeamSection(0.01831, 0.00000210773, 0.00190898, 0.000104873, 200_000_000_000, 0.3, 9946, 97530.476),
    "StringerCross": BeamSection(0.00637843, 0.000000406964, 0.000152071, 0.00000504857, 200_000_000_000, 0.3, 9946, 97530.476),
    "StringerLateralBracing": BeamSection(0.001402, 0.00000004666666, 0.000000714, 0.000000714, 200_000_000_000, 0.3, 9946, 97530.476),
    "KneeBracing": BeamSection(0.0028, 0.0000000933333333, 0.0000014495610119, 0.0000035533333333, 200_000_000_000, 0.3, 9946, 97530.476),
  }

  # Adding load of rail on the stringer main
  RailLoad = 216
  g = 9.806
  sections["StringerMain"].rho += RailLoad/sections["StringerMain"].Area
  sections["StringerMain"].rhog += RailLoad*g/sections["StringerMain"].Area

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
  truss2.node_by_name("L0").addRestraint([1,0,1,0,0,0])
  truss2.node_by_name("L8").addRestraint([0,0,1,0,0,0])
  nodes.extend(truss2.nodes)
  beams.extend(truss2.beams)

  cross_girders:list[Beam] = []
  for i in range(4*2+1):
    nodeA, nodeB = truss1.node_by_name(f"L{i}"), truss2.node_by_name(f"L{i}")
    cross_girders.append(createTrussCrossMembers(nodeA, nodeB, sections["CrossGirders"], (5.28-1.9)/2-0.25, verticalOffset=0.2875))
    nodes.extend(cross_girders[-1].nodes)
    beams.append(cross_girders[-1])

  stringers:list[StringerAssembly] = []
  for i in range(4*2):
    stringers.append(StringerAssembly(cross_girders[i], cross_girders[i+1], 2.05, 0, [
      sections["StringerMain"],
      sections["StringerCross"],
      sections["StringerLateralBracing"],
    ]))
    nodes.extend(stringers[-1].nodes)
    beams.extend(stringers[-1].beams)

  portal_girders:list[Beam] = []
  for i,_beta in zip([1,7], np.arctan([5.905/7.315,-5.905/7.315])*180/np.pi):
    nodeA, nodeB = truss1.node_by_name(f"U{i}"), truss2.node_by_name(f"U{i}")
    portal_girders.append(createTrussCrossMembers(nodeA, nodeB, sections["PortalGirderU1"], 1.25, 0.25, _beta))
    nodes.extend(portal_girders[-1].nodes)
    beams.append(portal_girders[-1])

  portal_bracings:list[Beam] = []
  portal_bracings.extend(createKneeBracings(truss1.diagonalBeams[0], truss2.diagonalBeams[0], portal_girders[0], sections["KneeBracing"], 0.25, 0.225))
  portal_bracings.extend(createKneeBracings(truss1.diagonalBeams[-1], truss2.diagonalBeams[-1], portal_girders[-1], sections["KneeBracing"], 0.25, 0.225))
  beams.extend(portal_bracings)
  for _ in portal_bracings:
    nodes.extend(_.nodes)
  
  sway_girders:list[Beam] = []
  for i in range(2,7):
    nodeA, nodeB = truss1.node_by_name(f"U{i}"), truss2.node_by_name(f"U{i}")
    sway_girders.append(createTrussCrossMembers(nodeA, nodeB, sections["SwayGirders"], 1.25, 0.25, endsPinned=True))
    nodes.extend(sway_girders[-1].nodes)
    beams.append(sway_girders[-1])
  
  knee_bracings:list[Beam] = []
  for i in range(5):
    knee_bracings.extend(createKneeBracings(truss1.verticalBeams[i+1], truss2.verticalBeams[i+1], sway_girders[i], sections["KneeBracing"], 0.25, 0.278))
  beams.extend(knee_bracings)
  for _ in knee_bracings:
    nodes.extend(_.nodes)

  bottom_laterals:list[Beam] = []
  for i in range(9-1):
    nodeA1 = truss1.node_by_name(f"L{i}")
    nodeB1 = truss2.node_by_name(f"L{i}")
    nodeA2 = truss1.node_by_name(f"L{i+1}")
    nodeB2 = truss2.node_by_name(f"L{i+1}")
    bottom_laterals.extend(createLaterals(nodeA1, nodeA2, nodeB1, nodeB2, sections["BottomLateralBracing"], 0.25))
  beams.extend(bottom_laterals)
  for _ in bottom_laterals:
    nodes.extend(_.nodes)

  top_laterals:list[Beam] = []
  for i in range(1,7):
    nodeA1 = truss1.node_by_name(f"U{i}")
    nodeB1 = truss2.node_by_name(f"U{i}")
    nodeA2 = truss1.node_by_name(f"U{i+1}")
    nodeB2 = truss2.node_by_name(f"U{i+1}")
    top_laterals.extend(createLaterals(nodeA1, nodeA2, nodeB1, nodeB2, sections["TopLateralBracing"], 0.25))
  beams.extend(top_laterals)
  for _ in top_laterals:
    nodes.extend(_.nodes)

  for _ in beams:
    _.addSelfWeight(2)

  DOFClass.analyse()

  np.set_printoptions(suppress=True) # To suppress exponential notation
  # print(np.hstack([DOFClass.ActionVector.toarray(), DOFClass.ReactionVector.toarray(), DOFClass.DisplacementVector.toarray()]))
  print("Static Analysis Results")
  print("Reactions at 4 Nodes (Clockwise from Node L0 of Truss 1)")
  print(truss1.node_by_name("L0").getReaction())
  print(truss1.node_by_name("L8").getReaction())
  print(truss2.node_by_name("L8").getReaction())
  print(truss2.node_by_name("L0").getReaction())

  D,V,EffectiveMass,MassParticipationFactor = DOFClass.eig(50)
  T = 2*np.pi/D**0.5
  print("Eigenvalue Analysis Results")
  print("No.\tTime:\tDX\tDY\tDZ\tRX\tRY\tRz")
  for i, (_T,_MP) in enumerate(zip(T, MassParticipationFactor*100)):
    print(f"{i+1}\t{_T:.3f}:\t"+''.join([f"{_:.2f}\t" for _ in _MP]))

  # Display
  view = SimpleView(nodes, beams, V)
  view.start()
