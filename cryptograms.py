#!/usr/bin/python

# XX + YY + ZZ === XYZ

for x in range(1, 3):
    for y in range(1, 10):
        for z in range(6, 10):
            xx = 10*x + x
            yy = 10*y + y
            zz = 10*z + z
            xyz = xx + yy + zz
            print("{} + {} + {} == {}").format(xx, yy, zz, xyz)

# 11 + 99 + 88 == 198