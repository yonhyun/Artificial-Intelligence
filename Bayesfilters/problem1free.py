import numpy as np



p0=np.array([0,1,0])
A=np.array([[0.6,0.3,0.3],[0.1,0.4,0.4],[0.3,0.3,0.3]])


print(np.dot(A,p0))



def run(p0,i):
    temp=np.dot(p0,A)

    if np.array_equal(temp,p0):
        print(p0,i)
        return p0
    else:
        i=i+1
        return print(temp)
        run(temp,i)
run(p0,0)
