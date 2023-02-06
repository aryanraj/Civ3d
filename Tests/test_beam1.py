from Models import beam1, node, structure

n1 = node(1, [0,0,0], [1, 1, 1, 1, 0, 0])
n2 = node(2, [10,0,0], [0, 1, 1, 1, 0, 0])
n3 = node(3, [1,0,0])
n4 = node(4, [9,0,0])

b1 = beam1(1, n1, n2, n3, n4)
b1.addUDL(1, -10)

s1 = structure([n1, n2], [b1])
s1.solve()

print(b1.force)