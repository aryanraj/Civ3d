from Models import beam1, Node, structure

n1 = Node(1, [0,0,0], [1, 1, 1, 1, 0, 0])
n2 = Node(2, [10,0,0], [0, 1, 1, 1, 0, 0])
n3 = Node(3, [5,0,0], [0, 0, 0, 1, 0, 0])
n4 = Node(4, [1,0,0])
n5 = Node(5, [9,0,0])

b1 = beam1(1, n1, n3, n4, n3)
b2 = beam1(2, n3, n2, n3, n5)
b1.addUDL(1, -10)
b2.addUDL(1, -10)
b1.addPointLoad(1, -10, 4)

s1 = structure([n1, n2, n3], [b1, b2])
s1.solve()

print(b1.force)
print(b2.force)
