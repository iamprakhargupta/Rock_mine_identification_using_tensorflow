
# coding: utf-8

# In[3]:


import tensorflow as tf


# In[4]:


node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)


# In[5]:


sess = tf.Session()
print(sess.run([node1, node2]))


# In[6]:


node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))


# In[7]:


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)


# In[8]:


print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))


# In[1]:


import pandas as pd
df=pd.read_csv("sonar.csv")


# In[2]:


df


# In[3]:


import tensorflow as tf

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# In[4]:


def read_dataset():
    X=df[df.columns[0:60]].values
    y=df[df.columns[60]]
    
    
    Y=broadcasting_based(y)
    print(X.shape)
    return(X,Y)
def broadcasting_based(A):  # A is Input array
    a = np.unique(A, return_inverse=1)[1]
    return (a.ravel()[:,None] == np.arange(a.max()+1)).astype(int)


# In[5]:


X,Y= read_dataset()


# In[6]:


X,Y=shuffle(X,Y,random_state=2)


# In[7]:


train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.10,random_state=42)


# In[15]:


#tensorflow starts

learning_rate=0.3
training_epochs=1000
cost_history=np.empty(shape=[1],dtype=float)
n_dim=X.shape[1]
n_class=2
model_paths="C:\\tensorflow"

n_hidden_1=60

n_hidden_2=60

n_hidden_3=60

n_hidden_4=60

x=tf.placeholder(tf.float32,[None,n_dim])
W=tf.Variable(tf.zeros([n_dim,n_class]))
b=tf.Variable(tf.zeros([n_class]))
y_=tf.placeholder(tf.float32,[None,n_class])


# In[16]:


def multilayer_perceptron(x,weights,biases):
    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1=tf.nn.sigmoid(layer_1)
    
    layer_2=tf.add(tf.matmul(layer_1,weights['h1']),biases['b1'])
    layer_2=tf.nn.sigmoid(layer_1)
    
    layer_3=tf.add(tf.matmul(layer_2,weights['h1']),biases['b1'])
    layer_3=tf.nn.sigmoid(layer_1)
    
    layer_4=tf.add(tf.matmul(layer_3,weights['h1']),biases['b1'])
    layer_1=tf.nn.relu(layer_1)
    
    out_layer=tf.matmul(layer_4,weights['out'])+biases['out']
    return out_layer

weights={
    'h1':tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
    'h2':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
    'h3':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
    'h4':tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
    'out':tf.Variable(tf.truncated_normal([n_hidden_4,n_class]))
}

biases={
    'b1':tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2':tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b3':tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4':tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out':tf.Variable(tf.truncated_normal([n_class]))
}


# In[17]:


init=tf.global_variables_initializer()
saver=tf.train.Saver()
y=multilayer_perceptron(x,weights,biases)
cost_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
training_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
sess=tf.Session()
sess.run(init)


# In[9]:


#Using desicion tree to show why sklearn fails on this data set

from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(train_x,train_y)


# In[21]:


mse_history=[]
accuracy_history=[]

for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={x:train_x,y_:train_y})
    cost=sess.run(cost_function,feed_dict={x:train_x,y_:train_y})
    cost_history=np.append(cost_history,cost)
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    pred_y=sess.run(y,feed_dict={x:test_x})
    mse=tf.reduce_mean(tf.square(pred_y-test_y))
    mse_=sess.run(mse)
    mse_history.append(mse_)
    accuracy=(sess.run(accuracy,feed_dict={x:train_x,y_:train_y}))
    accuracy_history.append(accuracy)
    
    print('epoch:',epoch,"-","cost:",cost,"accuracy:",accuracy)
    
save_path=saver.save(sess,model_paths)
print("saved to %s "% save_path)


# In[24]:



correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print("Test Accuracy:",(sess.run(accuracy,feed_dict={x:test_x,y_:test_y})))

pred_y=sess.run(y,feed_dict={x:test_x})
mse=tf.reduce_mean(tf.square(pred_y-test_y))
print("MSE %.4f" % sess.run(mse))


# In[ ]:




