import numpy as np
import matplotlib.pyplot as plt
np.random.seed(17)



def sign(x):
    return np.where(x >= 0, 1, -1)

def calculate_energy(weights, pattern):
         return -weights @ pattern @ pattern.T 


class Hopfield():

     def __init__(self, neurons):
         self.neurons= neurons
         self.weights = np.zeros((neurons,neurons))


     def fit(self, patterns):
         self.weights =  (patterns.T @ patterns)  * (1/self.neurons)
         np.fill_diagonal(self.weights, 0)

     def random_weights(self):
         self.weights = np.random.normal(size=(self.neurons,self.neurons))   
         np.fill_diagonal(self.weights, 0) 

     def symmetric_weight(self):   
       self.weights =  0.5 * (self.weights + self.weights.T)


     def predict(self,pattern, max_iteratiosn, approach):
          all_energi=[]

          current_pattern = pattern
          if(approach=="synchronous"):
            current_pattern = pattern
            for _ in range(max_iteratiosn):
              temp = self.weights @ current_pattern

              
              if(np.array_equal(sign(temp),current_pattern)):
                  break
              
              all_energi.append(calculate_energy(self.weights,current_pattern))

              current_pattern = sign(temp)

            
            return current_pattern, np.array(all_energi)
        
               


          elif(approach=="asynchronous"):
              current_pattern = pattern
              pre_pattern = np.zeros(self.neurons)
              all_energi=[]

              for i in range(max_iteratiosn):
                 indices = np.random.permutation(self.neurons)
                                 
                 if(np.array_equal(current_pattern,pre_pattern)):
                       break
                 
                 pre_pattern =  current_pattern.copy()
                 for index in indices:
                    
                    all_energi.append(calculate_energy(self.weights,current_pattern)) 
                    temp_weight =  np.dot(self.weights[index], current_pattern)
                    current_pattern[index]= sign(temp_weight)




             
              return current_pattern, np.array(all_energi)
              








