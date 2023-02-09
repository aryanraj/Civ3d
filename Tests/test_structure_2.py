from Models import beam1, Node, structure
from Views.simpleStructure import SimpleStructureView

n1 = Node(1, [0,0,0], [1, 1, 1, 1, 0, 0])
n2 = Node(2, [10,0,0], [0, 1, 1, 1, 0, 0])
n3 = Node(3, [10,10,0], [0, 0, 0, 1, 0, 0])
n4 = Node(4, [10,1,0])
n5 = Node(5, [10,9,0])

b1 = beam1(1, n1, n2, n1, n2)
b2 = beam1(2, n2, n3, n4, n5)
b3 = beam1(3, n1, n3, n1, n3)
b1.addPointLoad(1, -10, 5)

s1 = structure([n1, n2, n3], [b1, b2, b3])
s1.solve()

print(b1.force)
print(b2.force)
print(b3.force)

view = SimpleStructureView(s1)
view.display()