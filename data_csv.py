import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

unit_step = lambda x:0 if x<0 else 1

def mat_mul(w , x):
	p = 0
	for i in range(len(x)):
		p = p + w[0][i]*x[i]
	#print p
	return p
		

dataset = pd.read_csv('data_csv.csv')
w = np.random.rand(1,3)

eta = input("Enter the learning rate")
n = input("Enter the number of iterations")

x = dataset.iloc[:, :-1].values
expected = dataset.iloc[:, 3].values
#print x
#print expected

for j in range(n):
	for i in range(len(dataset)):
		r = x[i]
		result = mat_mul(w,r)
		#print result
		error = expected[i] - unit_step(result)
		#print("{}: {} -> {}".format(expected[i],unit_step(result), error))
		w+= eta*error*x[i]

"""for i in range(len(dataset)):
	result = mat_mul(w,x[i])
	print("{}: {} -> {}".format(x[i], result, unit_step(result)))
#print(x)"""
y_pred = []

x_test = pd.read_csv('test.csv')
q =x_test.iloc[:, :].values
for i in range(len(q)):
		r = q[i]
		result = mat_mul(w,r)
		y_pred += [unit_step(result)]	

x1 = q[:,0]
x2 = q[:,1]
b  = q[:,2]
df = pd.DataFrame(data = {"a" : x1,"b" : x2,"bias" : b,  "prediction" : y_pred})
df.to_csv("test_result.csv", sep = ',')


		


