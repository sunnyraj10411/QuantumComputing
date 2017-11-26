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

# tensor
#cNot = tf.placeholder(tf.float32)


#cNot = tf.Variable([[[[ unit, zero],[zero,zero]],[[ zero, unit],[zero,zero]]],[[[ zero, zero],[zero,unit]],[[ zero, zero],[unit,zero]]]],tf.complex64)


#2piiPhi = cos2piPhi + i*sin2piPhi

Phi = np.pi*0.6
angle = 2.0*np.pi*Phi
uVal = tf.complex(tf.cast(np.cos(angle),tf.float32),tf.cast(np.sin(angle),tf.float32))

U1 = tf.Variable([[[[ unit, zero],[zero,zero]],[[ zero, unit],[zero,zero]]],[[[ zero, zero],[uVal,zero]],[[ zero, zero],[zero,unit]]]],tf.complex64)

zeroVal = tf.Variable([[ unit, zero],[zero,zero]],tf.complex64)
zeroVal = tf.Variable([[ zero, zero],[zero,unit]],tf.complex64)


#hadamard transpose gate
hGateT = tf.conj(hGate)
U1T = tf.conj(U1)



#r3 = tf.ones([3, 4, 5])

print(tf.shape(tInput),tf.shape(hGate),tf.shape(U1))

dot1 =  tf.tensordot(tInput,hGate,[[0],[1]])
dot2 = tf.tensordot(U1,dot1,[[2],[0]])
dot3 = tf.tensordot(dot2,tInput,[[2],[0]])


dot4 = tf.tensordot(hGate,dot3,[[1],[0]])
dot5 = tf.tensordot(zeroVal,dot4,[[1],[0]])


dot6 = tf.tensordot(hGateT,dot5,[[1],[0]])
dot7 = tf.tensordot(U1T,dot6,[[2,3],[0,1]])

dot8 = tf.tensordot(hGateT,dot7,[[1],[0]])


dot9 = tf.tensordot(dot8,tInput,[[0],[0]])
dot10 = tf.tensordot(dot9,tInput,[[0],[0]])




alt1 = tf.tensordot(U1,hGate, [[2],[0]])

alt2 = tf.tensordot(alt1,tInput,[[3],[0]])

alt3 = tf.tensordot(alt2,tInput,[[2],[0]])



init = tf.global_variables_initializer()
sess.run(init)


#print(sess.run([tInput,hGate,cNot,dot1,dot2]))

print("dot1")
print(sess.run([dot1]))
print("dot2")
print(sess.run([dot2]))
print("dot3")
print(sess.run([dot3]))
print("dot4")
print(sess.run([dot4]))
print("dot5")
print(sess.run([dot5]))
print("dot6")
print(sess.run([dot6]))
print("dot7")
print(sess.run([dot7]))
print("dot8")
print(sess.run([dot8]))
print("dot9")
print(sess.run([dot9]))
print("dot10")
print(sess.run([dot10]))

print("Final answer should be: ",1-0.5*(1.0+np.cos(angle)))

#print("alt1")
#print(sess.run([alt1]))
#print("alt2")
#print(sess.run([alt2]))
#print("alt3")
#print(sess.run([alt3]))


