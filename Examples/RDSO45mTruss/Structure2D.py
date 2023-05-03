import os
os.environ["MAX_DOF"] = str(1000)

import numpy as np
import numpy.typing as npt
from Models import DOFClass, Node, BeamSection, Beam
from Views.simpleStructure import SimpleView
from Models import utils
from Examples.RDSO45mTruss.Truss4Bay import Truss4Bay
from Examples.RDSO45mTruss.sections import sections

class Structure2D():
  def __init__(self):
    self.nodes: list[Node] = []
    self.beams: list[Beam] = []

    g = 9.806

    BottomChordAdditionalWeights:list[float] = [
      9 * (5.28-0.5) * sections["CrossGirders"].Area * sections["CrossGirders"].rhog, 
      16 * 5.905 * sections["StringerMain"].Area * sections["StringerMain"].rhog,
      16 * 1.9 * sections["StringerCross"].Area * sections["StringerCross"].rhog,
      24 * ((1.9**2+(5.905/3)**2)**0.5) * sections["StringerLateralBracing"].Area * sections["StringerLateralBracing"].rhog,
      16 * ((5.905**2+5.28**2)**0.5*(5.28-0.5)/5.28) * sections["BottomLateralBracing"].Area * sections["BottomLateralBracing"].rhog,
    ]

    TopChordEndAdditionalWeights:list[float] = [
      2 * (5.28-0.5) * sections["PortalGirderU1"].Area * sections["PortalGirderU1"].rhog,
      4 * (1.25*2**0.5) * sections["KneeBracing"].Area * sections["KneeBracing"].rhog,
      2 * ((5.905**2+5.28**2)**0.5*(5.28-0.5)/5.28) * sections["TopLateralBracing"].Area * sections["TopLateralBracing"].rhog,
    ]
  
    TopChordMiddleAdditionalWeights:list[float] = [
      5 * (5.28-0.5) * sections["SwayGirders"].Area * sections["SwayGirders"].rhog,
      4 * (1.25*2**0.5) * sections["KneeBracing"].Area * sections["KneeBracing"].rhog,
      10 * ((5.905**2+5.28**2)**0.5*(5.28-0.5)/5.28) * sections["TopLateralBracing"].Area * sections["TopLateralBracing"].rhog,
    ]
  
    self.truss1 = Truss4Bay([0,0,0], [5.905,0,7.315], [
      sections["BottomChordL0L2"], sections["BottomChordL2L4"], sections["BottomChordL2L4"], sections["BottomChordL0L2"],
      sections["TopChordU1U3"], sections["TopChordU3U4"], sections["TopChordU1U3"],
      sections["EndRakerL0U1"],
      sections["DiagonalsU1L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU3L2"], sections["DiagonalsU1L2"],
      sections["EndRakerL0U1"],
      sections["Verticals"],
    ])

    for _node in self.truss1.nodes:
      try:
        _node.addRestraint([0,1,0,1,0,1])
      except Exception as e:
        print(str(e))

    self.truss1.node_by_name("L0").addRestraint([1,1,1,0,0,0])
    self.truss1.node_by_name("L8").addRestraint([0,1,1,0,0,0])
    self.nodes.extend(self.truss1.nodes)
    self.beams.extend(self.truss1.beams)

    # Adding Nodal Mass and load to bottom chord
    perNodeWeight = sum(BottomChordAdditionalWeights)/16/2
    print(f"Bottom Chord {perNodeWeight=}")
    for _ in list(range(8))+list(range(1,9)):
      self.truss1.node_by_name(f"L{_}").addLumpedMass(perNodeWeight/g)

    # Adding Nodal Mass and load to top chord ends
    perNodeWeight = sum(TopChordEndAdditionalWeights)/2/2
    print(f"Top Chord End {perNodeWeight=}")
    for _ in [1,7]:
      self.truss1.node_by_name(f"U{_}").addLumpedMass(perNodeWeight/g)

    # Adding Nodal Mass and load to top chord middle
    perNodeWeight = sum(TopChordMiddleAdditionalWeights)/5/2
    print(f"Top Chord Middle {perNodeWeight=}")
    for _ in range(2,7):
      self.truss1.node_by_name(f"U{_}").addLumpedMass(perNodeWeight/g)
  
  def addSelfWeight(self, dir:int, factor:float, loadCases:list[int]):
    for _ in self.beams:
      _.addSelfWeight(dir, factor, loadCases)
    for _ in self.nodes:
      _.addSelfWeight(dir, factor, loadCases)

  def setFixityFactorForLongitudinalActions(self, fixityFactor:float, shouldConstrain:bool=False):
    beams = self.truss1.bottomChordBeams + self.truss1.topChordBeams + self.truss1.diagonalBeams + self.truss1.verticalBeams
    for beam in beams:
      if shouldConstrain:
        beam.setEndConstrains([1,1,1,1,1,1], [1,1,1,1,1,1])
      else:
        beam.setEndConstrains([0,1,1,1,1,1], [0,1,1,1,1,1])
      beam.setEndStiffnessFactor([fixityFactor,0,0,0,0,0], [fixityFactor,0,0,0,0,0])

if __name__ == "__main__":
  structure = Structure2D()
  structure.addSelfWeight(2, -1, [0])
  DOFClass.analyse()

  np.set_printoptions(suppress=True) # To suppress exponential notation
  print("Static Analysis Results")
  print("Reactions at 2 Nodes")
  print(structure.truss1.node_by_name("L0").getReaction([0]).flatten())
  print(structure.truss1.node_by_name("L8").getReaction([0]).flatten())
  
  D,V,EffectiveMass,MassParticipationFactor = DOFClass.eig(50)
  T = 2*np.pi/D**0.5
  print(f"Eigenvalue Analysis Results with Full Fixity")
  print("No.\tTime\tFreq.:\tDX\tDY\tDZ\tRX\tRY\tRz")
  for i, (_T,_MP) in enumerate(zip(T, MassParticipationFactor*100)):
    print(f"{i+1}\t{_T:.3f}\t{1/_T:.2f}:\t"+''.join([f"{_:.2f}\t" for _ in _MP]))

  fixityFactor = 1000
  structure.setFixityFactorForLongitudinalActions(fixityFactor)
  D,V,EffectiveMass,MassParticipationFactor = DOFClass.eig(50)
  T = 2*np.pi/D**0.5
  print(f"Eigenvalue Analysis Results with {fixityFactor=}")
  print("No.\tTime\tFreq.:\tDX\tDY\tDZ\tRX\tRY\tRz")
  for i, (_T,_MP) in enumerate(zip(T, MassParticipationFactor*100)):
    print(f"{i+1}\t{_T:.3f}\t{1/_T:.2f}:\t"+''.join([f"{_:.2f}\t" for _ in _MP]))
  structure.setFixityFactorForLongitudinalActions(0, True)

  fixityFactor = 1
  structure.setFixityFactorForLongitudinalActions(fixityFactor)
  D,V,EffectiveMass,MassParticipationFactor = DOFClass.eig(50)
  T = 2*np.pi/D**0.5
  print(f"Eigenvalue Analysis Results with {fixityFactor=}")
  print("No.\tTime\tFreq.:\tDX\tDY\tDZ\tRX\tRY\tRz")
  for i, (_T,_MP) in enumerate(zip(T, MassParticipationFactor*100)):
    print(f"{i+1}\t{_T:.3f}\t{1/_T:.2f}:\t"+''.join([f"{_:.2f}\t" for _ in _MP]))
  structure.setFixityFactorForLongitudinalActions(0, True)

  fixityFactor = 0.01
  structure.setFixityFactorForLongitudinalActions(fixityFactor)
  D,V,EffectiveMass,MassParticipationFactor = DOFClass.eig(50)
  T = 2*np.pi/D**0.5
  print(f"Eigenvalue Analysis Results with {fixityFactor=}")
  print("No.\tTime\tFreq.:\tDX\tDY\tDZ\tRX\tRY\tRz")
  for i, (_T,_MP) in enumerate(zip(T, MassParticipationFactor*100)):
    print(f"{i+1}\t{_T:.3f}\t{1/_T:.2f}:\t"+''.join([f"{_:.2f}\t" for _ in _MP]))
  structure.setFixityFactorForLongitudinalActions(0, True)

  # Display
  view = SimpleView(structure.nodes, structure.beams, V)
  view.start()
