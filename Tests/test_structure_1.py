from Models import beam1, node, structure

n1 = node(1, [0,0,0], [1, 1, 1, 1, 0, 0])
n2 = node(2, [10,0,0], [0, 1, 1, 1, 0, 0])
n3 = node(3, [5,0,0], [0, 0, 0, 1, 0, 0])
n4 = node(4, [1,0,0])
n5 = node(5, [9,0,0])

b1 = beam1(1, n1, n3, n4, n3)
b2 = beam1(2, n3, n2, n3, n5)
b1.addUDL(1, -10)
b2.addUDL(1, -10)
b1.addPointLoad(1, -10, 4)

s1 = structure([n1, n2, n3], [b1, b2])
s1.solve()

print(b1.force)
print(b2.force)
