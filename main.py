import numpy as np
from Models import beam1 as beam, Node, structure
from typing import Dict
from Views.simpleStructure import SimpleStructureView

if __name__ == "__main__":
  nodes_map = {}
  beams_map = {}
  node_index = 0
  beam_index = 0
  
  # Bottom Chord nodes
  n = 9
  nodes_map.update({f"{x},{y},{z}":(i, x, y, z) for i, x, y, z in zip(np.arange(n)+node_index, np.arange(n)*5.905, np.zeros(n), np.zeros(n))})
  node_index += n
  
  
  # Top Chord nodes
  n = 7
  nodes_map.update({f"{x},{y},{z}":(i, x, y, z) for i, x, y, z in zip(np.arange(n)+node_index, 5.905+np.arange(n)*5.905, np.zeros(n), 7.315+np.zeros(n))})
  node_index += n
  
  # Bottom Chord beams
  n = 8
  beams_map.update({f"{n1},{n2}":(i, n1, n2) for i, n1, n2 in zip(np.arange(n)+beam_index, 0+np.arange(n), 1+np.arange(n))})
  beam_index += n
  
  # Top Chord beams
  n = 6
  beams_map.update({f"{n1},{n2}":(i, n1, n2) for i, n1, n2 in zip(np.arange(n)+beam_index, 9+np.arange(n), 10+np.arange(n))})
  beam_index += n
  
  # Diagonal beams
  n = 8
  beams_map.update({f"{n1},{n2}":(i, n1, n2) for i, n1, n2 in zip(np.arange(n)+beam_index, [0,2,2,4,4,6,6,8], [9,9,11,11,13,13,15,15])})
  beam_index += n
  
  # Vertical beams
  n = 7
  beams_map.update({f"{n1},{n2}":(i, n1, n2) for i, n1, n2 in zip(np.arange(n)+beam_index, 1+np.arange(n), 9+np.arange(n))})
  beam_index += n
  
  nodes: Dict[int, Node] = {}
  for i, x, y, z in nodes_map.values():
    nodes[i] = Node(i, [x, y, z], [0,0,0,0,0,0])

  nodes[0].addRestraint([1,1,1,1,0,0])
  nodes[8].addRestraint([0,1,1,1,0,0])

  beams: Dict[int, beam] = {}
  for i, n1, n2 in beams_map.values():
    beams[i] = beam(i, nodes[n1], nodes[n2], nodes[n1], nodes[n2])
  
  s = structure(nodes.values(), beams.values())
  s.solve()
  # print(s.Kg)

  view = SimpleStructureView(s)
  view.display()