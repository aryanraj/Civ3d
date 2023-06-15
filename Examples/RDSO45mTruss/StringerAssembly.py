from Models import DOFClass, Node, Beam
from Models import utils
from Examples.RDSO45mTruss.sections import sections

class StringerAssembly:
  def __init__(self, crossGirderA:Beam, crossGirderB:Beam, crossBeamSpacing:float, verticalOffset:float):
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
      self.main.append(Beam([n0, n1, n2, n3], sections["StringerMain"], A=nodeA, B=nodeB))
      self.nodes.extend(self.main[-1].nodes)
      self.beams.append(self.main[-1])

    self.cross:list[Beam] = []
    self.cross.append(Beam([self.main[0].nodes[1], self.main[1].nodes[1]], sections["StringerCross"], constraintsA=[1,1,1,1,0,0], constraintsB=[1,1,1,1,0,0]))
    self.nodes.extend(self.cross[-1].nodes)
    self.cross.append(Beam([self.main[0].nodes[2], self.main[1].nodes[2]], sections["StringerCross"], constraintsA=[1,1,1,1,0,0], constraintsB=[1,1,1,1,0,0]))
    self.nodes.extend(self.cross[-1].nodes)
    self.beams.extend(self.cross)

    self.lateral:list[Beam] = []
    self.lateral.append(Beam([self.main[0].nodes[0], self.main[1].nodes[1]], sections["StringerLateralBracing"], constraintsA=[1,1,1,1,0,0], constraintsB=[1,1,1,1,0,0]))
    self.nodes.extend(self.lateral[-1].nodes)
    self.lateral.append(Beam([self.main[1].nodes[1], self.main[0].nodes[2]], sections["StringerLateralBracing"], constraintsA=[1,1,1,1,0,0], constraintsB=[1,1,1,1,0,0]))
    self.nodes.extend(self.lateral[-1].nodes)
    self.lateral.append(Beam([self.main[0].nodes[2], self.main[1].nodes[3]], sections["StringerLateralBracing"], constraintsA=[1,1,1,1,0,0], constraintsB=[1,1,1,1,0,0]))
    self.nodes.extend(self.lateral[-1].nodes)
    self.beams.extend(self.lateral)

if __name__ == "__main__":
  beams:list[Beam] = []
  nodes:list[Node] = [] 
  crossGirderA = Beam([Node([0,0.25,0]), Node([0,(5.28-1.9)/2,0]), Node([0,(5.28+1.9)/2,0]), Node([0,5.28-0.25,0])], sections["CrossGirders"])
  crossGirderB = Beam([Node([5.905,0.25,0]), Node([5.905,(5.28-1.9)/2,0]), Node([5.905,(5.28+1.9)/2,0]), Node([5.905,5.28-0.25,0])], sections["CrossGirders"])
  beams.extend([crossGirderA, crossGirderB])
  nodes.extend(crossGirderA.nodes + crossGirderB.nodes)

  stringer = StringerAssembly(crossGirderA, crossGirderB, 2.05, 0)
  beams.extend(stringer.beams)
  nodes.extend(stringer.nodes)
  crossGirderA.A.addRestraint([1,1,1,0,0,0])
  crossGirderA.B.addRestraint([1,1,1,0,0,0])
  crossGirderB.A.addRestraint([1,1,1,0,0,0])
  crossGirderB.B.addRestraint([1,1,1,0,0,0])

  for _ in beams:
    _.addSelfWeight(2, -1, [0])
  
  DOFClass.analyse()

  import numpy as np
  np.set_printoptions(suppress=True) # To suppress exponential notation
  print("Selfweight analysis")
  print("Reactions at 4 Nodes (Clockwise from CrossGirderA.A)")
  print(crossGirderA.A.getReaction([0]).flatten())
  print(crossGirderA.B.getReaction([0]).flatten())
  print(crossGirderB.B.getReaction([0]).flatten())
  print(crossGirderB.A.getReaction([0]).flatten())

  from Views.simpleStructure import SimpleView
  # Display
  view = SimpleView(nodes, beams)
  view.start()