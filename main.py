import os
os.environ["MAX_DOF"] = str(300)

import numpy as np
from Models import DOFClass, Node, Beam
from Views.simpleStructure import SimpleView

if __name__ == "__main__":
  nodes:list[Node] = []
  beams:list[Beam] = []
  
  # Bottom Chord nodes
  nBottomNodes = 9
  nodes.extend([Node(coord) for coord in zip(np.arange(nBottomNodes)*5.905, np.zeros(nBottomNodes), np.zeros(nBottomNodes))])
  nodes[0].addRestraint([1,1,1,0,0,0])
  nodes[-1].addRestraint([0,1,1,0,0,0])
  
  # Top Chord nodes
  nTopNodes = nBottomNodes - 2
  nodes.extend([Node(coord) for coord in zip(5.905+np.arange(nTopNodes)*5.905, np.zeros(nTopNodes), 7.315+np.zeros(nTopNodes))])
  
  # Bottom Chord beams
  bottomChordBeam = Beam(nodes[:nBottomNodes])
  beams.append(bottomChordBeam)
  
  # Top Chord beams
  topChordBeam = Beam(nodes[nBottomNodes:nBottomNodes+nTopNodes])
  beams.append(topChordBeam)
  
  # Diagonal beams
  diagnoalBeams = [Beam([nodes[n1], nodes[n2]]) for n1, n2 in zip([0,2,2,4,4,6,6,8], [9,9,11,11,13,13,15,15])]
  beams.extend(diagnoalBeams)
  
  # Vertical beams
  verticalBeams = [Beam([nodes[n1], nodes[n2]]) for n1, n2 in zip(1+np.arange(nTopNodes), nBottomNodes+np.arange(nTopNodes))]
  beams.extend(verticalBeams)
  
  # Add Load on Bottom Chord
  bottomChordBeam.addUDL(1,-10)
  DOFClass.analyse()

  # Display
  SimpleView().display(nodes, beams)

  np.set_printoptions(suppress=True) # To suppress exponential notation
  print(np.hstack([DOFClass.ActionVector, DOFClass.ReactionVector, DOFClass.DisplacementVector]))