
# coding: utf-8

# In[57]:


import tensorflow as tf
import numpy as np


# In[58]:


hello=tf.constant("hello world")


# In[59]:


type(hello)


# In[60]:


sess=tf.Session()


# In[61]:


sess.run(hello)


# In[62]:


x=tf.placeholder(tf.int32)


# In[63]:


y=tf.placeholder(tf.int32)


# In[64]:


add=tf.add(x,y)
sub=tf.subtract(x,y)
d={x:20,y:30}


# In[65]:


with tf.Session() as sess:
    print("Addition",sess.run(add,feed_dict={x:20,y:30}))
    print("Subtraction",sess.run(sub,feed_dict=d))


# In[66]:


a=np.array([[5.0,5.0]])
b=np.array([[2.0],[3.0]])


# In[67]:


mat1=tf.constant(a)
mat2=tf.constant(b)


# In[68]:


matrix_multi=tf.matmul(mat1,mat2)


# In[69]:


with tf.Session() as sess:
    result=sess.run(matrix_multi)
    print(result)


# In[70]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# In[71]:


type(mnist)


# In[72]:


type(mnist.train.images)


# In[73]:


mnist.train.images[2].shape


# In[74]:


sample = mnist.train.images[2].reshape(28,28)


# In[75]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(sample)


# In[166]:


learning_rate = 0.001
training_epochs = 100
batch_size = 10


# In[167]:


n_hidden_1=256
n_hidden_2=256
n_input=784
n_classes=10
n_samples=mnist.train.num_examples


# In[168]:


def multilayer_perceptron(x, weights, biases):
    '''
    x : Place Holder for Data Input
    weights: Dictionary of weights
    biases: Dicitionary of biases
    '''
    
    # First Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    # Second Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    # Last Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# In[169]:


weights={
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}


# In[170]:


biases={
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}


# In[171]:


x=tf.placeholder('float',[None,n_input])


# In[172]:


y=tf.placeholder('float',[None,n_classes])


# In[173]:


pred=multilayer_perceptron(x,weights,biases)


# In[174]:


cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[175]:


t=mnist.train.next_batch(10)


# In[176]:


sess=tf.InteractiveSession()


# In[177]:


init=tf.initialize_all_variables()


# In[178]:


sess.run(init)


# In[179]:


for epoch in range(training_epochs):
    avg_cost=0.0
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
        avg_cost+=c/total_batch
    print("Epoch: {} cost={:.4f}".format(epoch+1,avg_cost))
    
print("Model has completed {} Epochs of Training".format(training_epochs))


# In[180]:


# Test model
correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))


# In[181]:


print(correct_predictions[0])


# In[182]:


correct_predictions = tf.cast(correct_predictions, "float")


# In[183]:


print(correct_predictions[0])


# In[184]:


accuracy = tf.reduce_mean(correct_predictions)


# In[185]:


mnist.test.labels


# In[186]:


mnist.test.images


# In[187]:


print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

