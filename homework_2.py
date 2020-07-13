"""
make dictionary data type with given keys and values
the set of key and value have to be always same
return the made dictionary
"""
def makeDict(K, V):
    dic = {}
    i=0
    for i in range(len(K)):
       dic[K[i]] = V[i]
    return dic

K = ('Korean', 'Mathematics','English')
V = (90.3,85.5,92.7)
dic = makeDict(K,V)
for i in range(0,len(K)):
    print(K[i], " : ", V[i], " == ",K[i], " : ", dic[K[i]]) #compare original value and given value from dictionary