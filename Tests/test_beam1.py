from Models import beam1, Node, structure

n1 = Node(1, [0,0,0], [1, 1, 1, 1, 0, 0])
n2 = Node(2, [10,0,0], [0, 1, 1, 1, 0, 0])
n3 = Node(3, [1,0,0])
n4 = Node(4, [9,0,0])

b1 = beam1(1, n1, n2, n3, n4)
b1.addUDL(1, -10)

s1 = structure([n1, n2], [b1])
s1.solve()

print(b1.force)