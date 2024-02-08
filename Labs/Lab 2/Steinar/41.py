from SOM import SOM
import numpy as np

dataset = None
animal_names = None

with open('../Dataset/animals.dat') as f:
    d = f.readline()
    d = np.array(d.split(','), dtype=float)
    d = d.reshape((32, 84))
    dataset = d.T

with open('../Dataset/animalnames.txt') as f:
    d = f.readlines()
    d = np.array(d, dtype=str).reshape((1, -1))
    animal_names = d
    
    
model = SOM(100, 84)
model.fit_data(dataset)

results = []

for i in range(100):
    results.append([])

for i in range(dataset.shape[1]):
    index = model.predict(dataset[:, i])
    results[index].append(animal_names[0, i])


string = ''

for r in results:
    for n in r:
        string = string + ', ' + str.strip(n, '\t\n\'') 

print (string)

#print(results)

