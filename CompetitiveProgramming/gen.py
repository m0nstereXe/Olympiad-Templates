from random import randint, shuffle
from math import gcd

n,k=3,2

print(n,k)
for i in range(k):
    #make string s with random 0 and 1
    s = ""
    for j in range(n):
        s += str(randint(0,1))
    print(s)