import numpy as np
import tensorflow as tf
print(np.log(2.0 * np.pi * np.e), 1)
a=list([1,1])
v=1
print(type(1)("123"))
print([1,1,1]+[[3,3,3],2,2])
c=np.array([1,2,3])
if any(c)>1:
    print(c)
comm=[1,2,3]
comm=[111]
print(comm)
print(np.shape([[1,2],[2,2]]))
print('GPU',tf.test.is_gpu_available())
