import math #import math to use floor

"""
check whether the number is prime or not
divide the number in range of 2 to root of the number
if any remainder is 0 , the number isn't prime number
"""
def isPrime(n) :
    prime = True
    for i in range(2,math.floor(math.sqrt(n))+1):
        if n%i==0 :
            print(i) #print the factor of given number
            prime = False
    if prime==False :
        return 1
    return 0

for i in range(2,32767):
    if isPrime(i)==0:
        print(i," is prime")
    else:
        print(i," is not prime")
