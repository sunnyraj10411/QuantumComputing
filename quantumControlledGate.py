import numpy as np
import tensorflow as tf



sess = tf.Session()
#we are dealing with complex numbers so two for one !

unit = tf.complex(1.0,0.0)
zero = tf.complex(0.0,0.0)
tInput = tf.Variable([unit, zero],tf.complex64)

#
a = tf.cast(np.sqrt(0.5),tf.float32)
sqrtc = tf.complex(a,0.0)
sqrtn = tf.complex(-a,0.0)
hGate = tf.Variable([[sqrtc,sqrtc],[sqrtc,sqrtn]],tf.complex64)



copyNp = np.empty((2,2,2),dtype=complex)
xorNp = np.empty((2,2,2),dtype=complex)
for i in range(2):
    for j in range(2):
        for k in range(2):
            #copyNp[i][j][k] = 1 
            copyNp[i][j][k] = float((1-i)*(1-j)*(1-k)+i*j*k)+0.0j
            xorNp[i][j][k] = float(1-(i+j+k)+2*(i*j+j*k+k*i)-4*i*j*k) + 0.0j



#print(copyNp)
#print(xorNp)

# tensor
#cNot = tf.placeholder(tf.float32)


#cNot1 = np.array([[[[ unit, zero],[zero,zero]],[[ zero, unit],[zero,zero]]],[[[ zero, zero],[zero,unit]],[[ zero, zero],[unit,zero]]]])
cNot1 = tf.Variable([[[[ unit, zero],[zero,zero]],[[ zero, unit],[zero,zero]]],[[[ zero, zero],[zero,unit]],[[ zero, zero],[unit,zero]]]],tf.complex64)
#r3 = tf.ones([3, 4, 5])

cNot = tf.tensordot(copyNp,xorNp,[[1],[1]])


print(tf.shape(tInput),tf.shape(hGate),tf.shape(cNot))

init = tf.global_variables_initializer()
sess.run(init)

print("Hardcoded cnot")
valcNot1 = sess.run(cNot1)

for q in range(2):
    for i in range(2):
        for r in range(2):
                print(valcNot1[q][r][i][0], valcNot1[q][r][i][1])


print("contraction cnot")
print(sess.run(cNot))





