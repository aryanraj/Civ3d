import os
os.environ["MAX_DOF"] = str(3000)

import numpy as np
import numpy.typing as npt
from Models import DOFClass, Node, BeamSection, Beam
from Models import utils
from Views.simpleStructure import SimpleView
from Truss4Bay import Truss4Bay
from StringerAssembly import StringerAssembly
from sections import sections

def createTrussCrossMembers(nodeA:Node, nodeB:Node, section:BeamSection, addlNodeDist:float, nodeOffset:float=0.25, beta:float=0., verticalOffset:float=0.) -> Beam:
  # TODO: Relate this with the section width
  axis = utils.getAxisFromTwoNodesAndBeta(nodeA.coord, nodeB.coord, beta)
  n0 = Node(nodeA.coord + axis[0]*nodeOffset + axis[2]*verticalOffset)
  n1 = Node(nodeA.coord + (addlNodeDist+nodeOffset)*axis[0] + axis[2]*verticalOffset)
  n2 = Node(nodeB.coord - (addlNodeDist+nodeOffset)*axis[0] + axis[2]*verticalOffset)
  n3 = Node(nodeB.coord - axis[0]*nodeOffset + axis[2]*verticalOffset)
  return Beam([n0, n1, n2, n3], section, beta, A=nodeA, B=nodeB)

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
  b1.setEndConstrains([1,1,1,1,0,1], [1,1,1,1,0,1])
  # First KneeBracing 
  nodeA = truss2Beam.nodes[-2]
  nodeB = crossMember.nodes[-2]
  n0coord = nodeA.coord - Aoffset * axis[0]
  n1coord = nodeB.coord - Boffset * axis[2]
  baxis = utils.getAxisFromTwoNodesAndBeta(n0coord, n1coord, 0)
  n0 = Node(n0coord, axis=baxis)
  n1 = Node(n1coord, axis=baxis)
  b2 = Beam([n0, n1], section, A=nodeA, B=nodeB)
  b2.setEndConstrains([1,1,1,1,0,1], [1,1,1,1,0,1])
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
  b1.setEndConstrains([1,1,1,1,0,1], [1,1,1,1,0,1])
  # Second beam
  n0 = Node(nodeA2.coord + axisA2B1[0]*xoffset)
  n1 = Node(nodeB1.coord - axisA2B1[0]*xoffset)
  b2 = Beam([n0,ncenter,n1], section, A=nodeA2, B=nodeB1)
  b2.setEndConstrains([1,1,1,1,0,1], [1,1,1,1,0,1])
  return [b1,b2]

class Structure3D():
  def __init__(self):
    self.nodes: list[Node] = []
    self.beams: list[Beam] = []

    self.truss1 = Truss4Bay([0,0,0], [5.905,0,7.315], [
      sections["BottomChordL0L2"], sections["BottomChordL2L4"], sections["BottomChordL2L4"], sections["BottomChordL0L2"],
      sections["TopChordU1U3"], sections["TopChordU3U4"], sections["TopChordU1U3"],
      sections["EndRakerL0U1"],
      sections["DiagonalsU1L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU1L2"],
      sections["EndRakerL0U1"],
      sections["Verticals"],
    ])
    self.truss1.node_by_name("L0").addRestraint([1,1,1,0,0,0])
    self.truss1.node_by_name("L8").addRestraint([0,1,1,0,0,0])
    self.nodes.extend(self.truss1.nodes)
    self.beams.extend(self.truss1.beams)
    self.truss2 = Truss4Bay([0,5.28,0], [5.905,5.28,7.315], [
      sections["BottomChordL0L2"], sections["BottomChordL2L4"], sections["BottomChordL2L4"], sections["BottomChordL0L2"],
      sections["TopChordU1U3"], sections["TopChordU3U4"], sections["TopChordU1U3"],
      sections["EndRakerL0U1"],
      sections["DiagonalsU1L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU1L2"],
      sections["EndRakerL0U1"],
      sections["Verticals"],
    ])
    self.truss2.node_by_name("L0").addRestraint([1,0,1,0,0,0])
    self.truss2.node_by_name("L8").addRestraint([0,0,1,0,0,0])
    self.nodes.extend(self.truss2.nodes)
    self.beams.extend(self.truss2.beams)

    self.cross_girders:list[Beam] = []
    for i in range(4*2+1):
      nodeA, nodeB = self.truss1.node_by_name(f"L{i}"), self.truss2.node_by_name(f"L{i}")
      self.cross_girders.append(createTrussCrossMembers(nodeA, nodeB, sections["CrossGirders"], (5.28-1.9)/2-0.25, verticalOffset=0.2875))
      self.nodes.extend(self.cross_girders[-1].nodes)
      self.beams.append(self.cross_girders[-1])

    self.stringers:list[StringerAssembly] = []
    for i in range(4*2):
      self.stringers.append(StringerAssembly(self.cross_girders[i], self.cross_girders[i+1], 2.05, 0))
      self.nodes.extend(self.stringers[-1].nodes)
      self.beams.extend(self.stringers[-1].beams)

    self.portal_girders:list[Beam] = []
    for i,_beta in zip([1,7], np.arctan([5.905/7.315,-5.905/7.315])*180/np.pi):
      nodeA, nodeB = self.truss1.node_by_name(f"U{i}"), self.truss2.node_by_name(f"U{i}")
      self.portal_girders.append(createTrussCrossMembers(nodeA, nodeB, sections["PortalGirderU1"], 1.25, 0.25, _beta))
      self.nodes.extend(self.portal_girders[-1].nodes)
      self.beams.append(self.portal_girders[-1])

    self.portal_bracings:list[Beam] = []
    self.portal_bracings.extend(createKneeBracings(self.truss1.diagonalBeams[0], self.truss2.diagonalBeams[0], self.portal_girders[0], sections["KneeBracing"], 0.25, 0.225))
    self.portal_bracings.extend(createKneeBracings(self.truss1.diagonalBeams[-1], self.truss2.diagonalBeams[-1], self.portal_girders[-1], sections["KneeBracing"], 0.25, 0.225))
    self.beams.extend(self.portal_bracings)
    for _ in self.portal_bracings:
      self.nodes.extend(_.nodes)
  
    self.sway_girders:list[Beam] = []
    for i in range(2,7):
      nodeA, nodeB = self.truss1.node_by_name(f"U{i}"), self.truss2.node_by_name(f"U{i}")
      self.sway_girders.append(createTrussCrossMembers(nodeA, nodeB, sections["SwayGirders"], 1.25, 0.25))
      self.nodes.extend(self.sway_girders[-1].nodes)
      self.beams.append(self.sway_girders[-1])
  
    self.knee_bracings:list[Beam] = []
    for i in range(5):
      self.knee_bracings.extend(createKneeBracings(self.truss1.verticalBeams[i+1], self.truss2.verticalBeams[i+1], self.sway_girders[i], sections["KneeBracing"], 0.25, 0.278))
    self.beams.extend(self.knee_bracings)
    for _ in self.knee_bracings:
      self.nodes.extend(_.nodes)

    self.bottom_laterals:list[Beam] = []
    for i in range(9-1):
      nodeA1 = self.truss1.node_by_name(f"L{i}")
      nodeB1 = self.truss2.node_by_name(f"L{i}")
      nodeA2 = self.truss1.node_by_name(f"L{i+1}")
      nodeB2 = self.truss2.node_by_name(f"L{i+1}")
      self.bottom_laterals.extend(createLaterals(nodeA1, nodeA2, nodeB1, nodeB2, sections["BottomLateralBracing"], 0.25))
    self.beams.extend(self.bottom_laterals)
    for _ in self.bottom_laterals:
      self.nodes.extend(_.nodes)

    self.top_laterals:list[Beam] = []
    for i in range(1,7):
      nodeA1 = self.truss1.node_by_name(f"U{i}")
      nodeB1 = self.truss2.node_by_name(f"U{i}")
      nodeA2 = self.truss1.node_by_name(f"U{i+1}")
      nodeB2 = self.truss2.node_by_name(f"U{i+1}")
      self.top_laterals.extend(createLaterals(nodeA1, nodeA2, nodeB1, nodeB2, sections["TopLateralBracing"], 0.25))
    self.beams.extend(self.top_laterals)
    for _ in self.top_laterals:
      self.nodes.extend(_.nodes)

  def addSelfWeight(self, dir:int=2, factor:float=-1):
    for _ in self.beams:
      _.addSelfWeight(dir, factor)
    for _ in self.nodes:
      _.addSelfWeight(dir, factor)

  def addFixityFactorForLongitudinalActions(self, fixityFactor:float):
    self.addFixityFactorForBeams(self.truss1.topChordBeams + self.truss1.bottomChordBeams + self.truss1.diagonalBeams + self.truss1.verticalBeams, fixityFactor)
    self.addFixityFactorForBeams(self.truss2.topChordBeams + self.truss2.bottomChordBeams + self.truss2.diagonalBeams + self.truss2.verticalBeams, fixityFactor)
    for beam in self.stringers:
      self.addFixityFactorForBeams(beam.main, fixityFactor)

  def addPartialFixityForTransverseActions(self, fixityFactor:float):
    self.addFixityFactorForBeams(self.cross_girders + self.sway_girders, self.portal_girders, fixityFactor)

  @staticmethod
  def addFixityFactorForBeams(beams:list[Beam], fixityFactor:float):
    constraintsA=[1,1,1,1,0,1]
    constraintsB=[1,1,1,1,0,1]
    for _beam in beams:
      _beam.setEndConstrains(constraintsA, constraintsB)
      endStiffnessAMy = fixityFactor * _beam.childBeams[0].section.E * _beam.childBeams[0].section.Iyy / _beam.childBeams[0].L 
      endStiffnessBMy = fixityFactor * _beam.childBeams[-1].section.E * _beam.childBeams[-1].section.Iyy / _beam.childBeams[-1].L 
      _beam.addEndStiffness([0,0,0,0,endStiffnessAMy,0], [0,0,0,0,endStiffnessBMy,0])

if __name__ == "__main__":
  structure = Structure3D()
  structure.addSelfWeight()
  DOFClass.analyse()

  np.set_printoptions(suppress=True) # To suppress exponential notation
  print("Static Analysis Results")
  print("Reactions at 4 Nodes (Clockwise from Node L0 of Truss 1)")
  print(structure.truss1.node_by_name("L0").getReaction())
  print(structure.truss1.node_by_name("L8").getReaction())
  print(structure.truss2.node_by_name("L8").getReaction())
  print(structure.truss2.node_by_name("L0").getReaction())

  fixityFactor = 0
  structure.addFixityFactorForLongitudinalActions(fixityFactor)
  D,V,EffectiveMass,MassParticipationFactor = DOFClass.eig(50)
  T = 2*np.pi/D**0.5
  print(f"Eigenvalue Analysis Results with {fixityFactor=}")
  print("No.\tTime:\tDX\tDY\tDZ\tRX\tRY\tRz")
  for i, (_T,_MP) in enumerate(zip(T, MassParticipationFactor*100)):
    print(f"{i+1}\t{_T:.3f}:\t"+''.join([f"{_:.2f}\t" for _ in _MP]))
  structure.addFixityFactorForLongitudinalActions(-fixityFactor)

  fixityFactor = 1
  structure.addFixityFactorForLongitudinalActions(fixityFactor)
  D,V,EffectiveMass,MassParticipationFactor = DOFClass.eig(50)
  T = 2*np.pi/D**0.5
  print(f"Eigenvalue Analysis Results with {fixityFactor=}")
  print("No.\tTime:\tDX\tDY\tDZ\tRX\tRY\tRz")
  for i, (_T,_MP) in enumerate(zip(T, MassParticipationFactor*100)):
    print(f"{i+1}\t{_T:.3f}:\t"+''.join([f"{_:.2f}\t" for _ in _MP]))
  structure.addFixityFactorForLongitudinalActions(-fixityFactor)

  fixityFactor = 1000
  structure.addFixityFactorForLongitudinalActions(fixityFactor)
  D,V,EffectiveMass,MassParticipationFactor = DOFClass.eig(50)
  T = 2*np.pi/D**0.5
  print(f"Eigenvalue Analysis Results with {fixityFactor=}")
  print("No.\tTime:\tDX\tDY\tDZ\tRX\tRY\tRz")
  for i, (_T,_MP) in enumerate(zip(T, MassParticipationFactor*100)):
    print(f"{i+1}\t{_T:.3f}:\t"+''.join([f"{_:.2f}\t" for _ in _MP]))
  structure.addFixityFactorForLongitudinalActions(-fixityFactor)

  # Display
  view = SimpleView(structure.nodes, structure.beams, V)
  view.start()