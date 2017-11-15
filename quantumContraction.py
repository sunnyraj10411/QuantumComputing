import numpy as np
import tensorflow as tf



sess = tf.Session()
#we are dealing with complex numbers so two for one !
tInput = tf.Variable([1.0, 0.0],tf.float32)

#
a = np.sqrt(0.5)
hGate = tf.constant([[a,a],[a,-a]],tf.float32)

# tensor
#cNot = tf.placeholder(tf.float32)
cNot = tf.Variable([[[[ 1.0, 0.0],[0.0,0.0]],[[ 0.0, 1.0],[0.0,0.0]]],[[[ 0.0, 0.0],[0.0,1.0]],[[ 0.0, 0.0],[1.0,0.0]]]],tf.float32)

r3 = tf.ones([3, 4, 5])


dot1 =  tf.tensordot(tInput,hGate,[[0],[1]])
dot2 = tf.tensordot(cNot,dot1,[[2],[0]])
dot3 = tf.tensordot(dot2,tInput,[[2],[0]])


alt1 = tf.tensordot(cNot,hGate, [[2],[0]])

alt2 = tf.tensordot(alt1,tInput,[[3],[0]])

alt3 = tf.tensordot(alt2,tInput,[[2],[0]])


print(tf.shape(tInput),tf.shape(hGate),tf.shape(cNot),tf.shape(r3))

init = tf.global_variables_initializer()
sess.run(init)


#print(sess.run([tInput,hGate,cNot,dot1,dot2]))

print("dot1")
print(sess.run([dot1]))
print("dot2")
print(sess.run([dot2]))
print("dot3")
print(sess.run([dot3]))


print("alt1")
print(sess.run([alt1]))
print("alt2")
print(sess.run([alt2]))
print("alt3")
print(sess.run([alt3]))


