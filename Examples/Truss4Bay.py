import numpy as np
import numpy.typing as npt
from Models import Node, BeamSection, Beam
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
    self.bottomChordBeams: list[Beam] = []
    _nodes = self.nodes[:self.nBottomNodes]
    _secs = sections[:self.nBottomBays]
    for n0, n1, n2, sec in zip(_nodes[:-2:2], _nodes[1:-1:2], _nodes[2::2], _secs):
      self.bottomChordBeams.append(Beam([n0, n1, n2], sec))
    self.beams.extend(self.bottomChordBeams)
  
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
      n0 = Node(nodeA.coord + axis[0]*bottomVertOffset/axis[0,2], axis=axis)
      topVertOffset = -0.28
      n2 = Node(nodeB.coord + axis[0]*topVertOffset/axis[0,2], axis=axis)
      topVertOffset += -1.25
      n1 = Node(nodeB.coord + axis[0]*topVertOffset/axis[0,2], axis=axis)
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
      n0 = Node(nodeA.coord + axis[0]*bottomVertOffset/axis[0,2], axis=axis)
      topVertOffset = -0.28
      n2 = Node(nodeB.coord + axis[0]*topVertOffset/axis[0,2], axis=axis)
      topVertOffset += -1.25
      n1 = Node(nodeB.coord + axis[0]*topVertOffset/axis[0,2], axis=axis)
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
