#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(0)

def f(x):
    return 3*x + 2


# My custom model
class MyKerasModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x, training=False):
        return self.w * x + self.b



class keras_custom_model(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.w=tf.Variable(tf.random.uniform([],-5,5))
        self.b=tf.Variable(tf.random.uniform([],-5,5))
    
    def __str__(self):
        return "w: " + str(self.w) + " b: " + str(self.b)

    def __call__(self, x, training=False):
        return self.w * x + self.b




class custom_model(tf.Module):
    def __init__(self):
        super().__init__()
        self.w=tf.Variable(tf.random.uniform([],-5,5))
        self.b=tf.Variable(tf.random.uniform([],-5,5))
        self.history=[]
        self.loss_history=[]
    def __str__(self):
        return "w: " + str(self.w) + " b: " + str(self.b)




    def MeanSquareError(self, y_data, y_truth):
        MSQ=0
        for i in range(tf.size(y_data).numpy()):
            MSQ+=(y_data[i]-y_truth[i])**2
        MSQ=MSQ/tf.size(y_data).numpy()
        return MSQ

    def evaluate(self, x):
        return self.w*x + self.b
    
    def train(self, x_tr,y_tr, rate):
        with tf.GradientTape() as tape:
            dw, db = tape.gradient(self.MeanSquareError(self.evaluate(x_tr), y_tr), [self.w, self.b])

        self.w.assign(self.w-dw*rate)
        self.b.assign(self.b-db*rate)

    def training_loop(self, n_loop, x_tr, y_tr):
        for i in range(n_loop):
            self.train(x_tr,y_tr, 2/float(i+1))
            l_value=self.MeanSquareError(self.evaluate(x_tr), y_tr)
            print(l_value)
            self.history.append([self.w, self.b,])
            self.loss_history.append(l_value)



def main():
    #Data generation
    start = tf.constant(-2, dtype = tf.float32)
    end = tf.constant(2, dtype = tf.float32)
    x=tf.linspace(start, end, 200)
    y_data=f(x)+tf.random.normal([200])
    y_truth=f(x)

    plt.figure(figsize=(10, 10))
    plt.suptitle("Evolution of the model")

    plt.subplot(3,1,1)   
    plt.title("Data generation")
    plt.scatter(x, y_data, label="Data")
    plt.plot(x, y_truth, label="Truth", color='pink')
    plt.legend()

    # End data generation, create untrained model
    plt.subplot(3,1,2)
    plt.title("Untrained model")

    mod=custom_model()
    plt.scatter(x, y_data,label="Data")
    plt.plot(x, y_truth,label="Truth",color='pink')
    plt.scatter(x,mod.evaluate(x), label="Model", s=10)
    
    print(mod.MeanSquareError(mod.evaluate(x), y_truth))
    plt.legend()


    # Begin training
    n_steps=10
    mod.training_loop(n_steps, x, y_data)
    # Final plots
    plt.subplot(3,1,3)
    plt.title("Trained model")
    plt.scatter(x, y_data,label="Data")
    plt.plot(x, y_truth,label="Truth",color='pink')
    plt.scatter(x,mod.evaluate(x), label="Model",s=10)
    plt.legend()


    # Do the same exercise with keras
    keras_mod=keras_custom_model()

    keras_mod.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.2),
                loss='mean_squared_error'
                )


    # Data generation
    plt.figure(figsize=(10, 10))
    plt.suptitle("Evolution of the keras model")

    plt.subplot(3,1,1)   
    plt.title("Data generation")
    plt.scatter(x, y_data, label="Data")
    plt.plot(x, y_truth, label="Truth", color='pink')
    plt.legend()

    # End data generation, create untrained model
    plt.subplot(3,1,2) 
    plt.scatter(x, y_data,label="Data")
    plt.plot(x, y_truth,label="Truth",color='pink')
    plt.scatter(x,keras_mod(x), label="Model", s=10)
    plt.legend()

    keras_mod.fit(x, y_data, epochs=10, batch_size=len(x))
    # Final plots
    plt.subplot(3,1,3) 
    plt.title("Keras model")
    plt.scatter(x, y_data,label="Data")
    plt.plot(x, y_truth,label="Truth",color='pink')
    plt.scatter(x,keras_mod(x), label="Model",s=10)
    plt.legend()
    plt.show()
    

    
if __name__=='__main__':
    main()



