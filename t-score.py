import numpy as np
from scipy import stats

N=10

a = np.array([45,38,52,48,25,39,51,46,55,46])
b = np.array([34,22,15,27,37,41,24,19,26,36])

var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

s = np.sqrt((var_a+var_b)/2)

t = (a.mean()-b.mean())/(s*np.sqrt(2/N))

df = 2*N-2

p = 1 - stats.t.cdf(t,df=df)

print("t = " + str(t))
print("p = " + str(2*p))

t2, p2 = stats.ttest_rel(a,b)
print("t = " + str(t2))
print("p = " + str(p2))