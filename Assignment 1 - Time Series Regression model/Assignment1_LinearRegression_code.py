from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_excel(r"Assignment-TimeSeriesData.xlsx", header = None)
array = data.to_numpy()
array = array.T[0] #Transpose

'''
h(theta) = theta_0 + theta_1*x_1  + theta_1*x_2  + theta_1*x_3 + ....
'''
Tap_delay_values = [2, 5, 8, 10]
y_predicted_taps, mse = [], []
for Tap in Tap_delay_values:
    y_matrix = array[Tap+1:]
    x_matrix = []
    for i in range(len(y_matrix)):
        x_temp = []
        for j in range(i, i+Tap+1):
            x_temp.append(array[j])
        x_matrix.append(x_temp)
    x_matrix = np.array(x_matrix)
    reg = LinearRegression().fit(x_matrix, y_matrix)
    y_predicted = []
    y_predicted.extend([None]*(Tap+1))
    for i in range(len(y_matrix)):
        y_predicted.append(reg.predict([x_matrix[i]])[0])
    y_predicted_taps.append(y_predicted)
    mse.append(mean_squared_error(array[Tap+1:], y_predicted_taps[0][Tap+1:], squared=False))
    
x_val = [i for i in range(len(array))]
plt.figure()
plt.scatter(x_val, array)
plt.plot(x_val, y_predicted_taps[0], label = 'Tap delay = 2, MSE = {}'.format(mse[0]))
plt.plot(x_val, y_predicted_taps[1], label = 'Tap delay = 5, MSE = {}'.format(mse[1]))
plt.plot(x_val, y_predicted_taps[2], label = 'Tap delay = 8, MSE = {}'.format(mse[2]))
plt.plot(x_val, y_predicted_taps[3], label = 'Tap delay = 10, MSE = {}'.format(mse[3]))
plt.xlabel('Time Series')
plt.ylabel('Values')
plt.legend()
plt.show()


'''
h(theta) = theta_0 + theta_1*x_1  + theta_1*x_2^2  + theta_1*x_3^3 + ...
'''
Tap_delay_values = [2, 5, 8, 10]
y_predicted_taps, mse = [], []
for Tap in Tap_delay_values:
    y_matrix = array[Tap+1:]
    x_matrix = []
    for i in range(len(y_matrix)):
        x_temp = []
        for j in range(i, i+Tap+1):
            x_temp.append(array[j])
        x_temp[1] = x_temp[1]**2
        x_temp[2] = x_temp[2]**3
        x_matrix.append(x_temp)
    x_matrix = np.array(x_matrix)
    reg = LinearRegression().fit(x_matrix, y_matrix)
    y_predicted = []
    y_predicted.extend([None]*(Tap+1))
    for i in range(len(y_matrix)):
        y_predicted.append(reg.predict([x_matrix[i]])[0])
    y_predicted_taps.append(y_predicted)
    mse.append(mean_squared_error(array[Tap+1:], y_predicted_taps[0][Tap+1:], squared=False))

x_val = [i for i in range(len(array))]
plt.figure()
plt.scatter(x_val, array)
plt.plot(x_val, y_predicted_taps[0], label = 'Tap delay = 2, MSE = {}'.format(mse[0]))
plt.plot(x_val, y_predicted_taps[1], label = 'Tap delay = 5, MSE = {}'.format(mse[1]))
plt.plot(x_val, y_predicted_taps[2], label = 'Tap delay = 8, MSE = {}'.format(mse[2]))
plt.plot(x_val, y_predicted_taps[3], label = 'Tap delay = 10, MSE = {}'.format(mse[3]))
plt.xlabel('Time Series')
plt.ylabel('Values')
plt.legend()
plt.show()

    

'''
h(theta) = theta_0 + theta_1*x_1  + theta_1*log(x_2)  + theta_1*sin(x_3) + ...
'''

Tap_delay_values = [2, 5, 8, 10]
y_predicted_taps, mse = [], []
for Tap in Tap_delay_values:
    y_matrix = array[Tap+1:]
    x_matrix = []
    for i in range(len(y_matrix)):
        x_temp = []
        for j in range(i, i+Tap+1):
            x_temp.append(array[j])
        x_temp[1] = np.log(x_temp[1])
        x_temp[2] = np.sin(x_temp[2])
        x_matrix.append(x_temp)
    x_matrix = np.array(x_matrix)
    reg = LinearRegression().fit(x_matrix, y_matrix)
    y_predicted = []
    y_predicted.extend([None]*(Tap+1))
    for i in range(len(y_matrix)):
        y_predicted.append(reg.predict([x_matrix[i]])[0])
    y_predicted_taps.append(y_predicted)
    mse.append(mean_squared_error(array[Tap+1:], y_predicted_taps[0][Tap+1:], squared=False))
    
x_val = [i for i in range(len(array))]
plt.figure()
plt.scatter(x_val, array)
plt.plot(x_val, y_predicted_taps[0], label = 'Tap delay = 2, MSE = {}'.format(mse[0]))
plt.plot(x_val, y_predicted_taps[1], label = 'Tap delay = 5, MSE = {}'.format(mse[1]))
plt.plot(x_val, y_predicted_taps[2], label = 'Tap delay = 8, MSE = {}'.format(mse[2]))
plt.plot(x_val, y_predicted_taps[3], label = 'Tap delay = 10, MSE = {}'.format(mse[3]))
plt.xlabel('Time Series')
plt.ylabel('Values')
plt.legend()
plt.show()


'''
Linear Regression from scratch :-
To observe the Convergence of Cost function over number of iterations.

'''

def linear_regression(x, y, slope_curr=0, intercept_curr=0, epoch=1000, learn_rate=0.0001):
    N = float(len(y))
    cost_arr = []
    for i in range(epoch):
        y_curr = slope_curr * x + intercept_curr
        cost = sum([data**2 for data in (y-y_curr)]) / N
        cost_arr.append(cost)
        del_slope = -(2/N) * sum(x * (y - y_curr))
        del_intercept = -(2/N) * sum(y - y_curr)
        slope_curr -= (learn_rate * del_slope)
        intercept_curr -= (learn_rate * del_intercept)
    return slope_curr, intercept_curr, cost_arr


x_axis = [i for i in range(len(array))]
Time_limit_delay = [2, 5, 8, 10]
Time_limit_delay_Degree_1 = [[] for i in range(4)] 
index = 0
for T in Time_limit_delay:
    n = T + 1
    Time_limit_delay_Degree_1[index].extend([None] * n)
    for i in range(len(array)-n):
        x, y = [], []
        for j in range(n):
            x.append(i+j)
        for j in x:
            y.append(array[j])
        m,c,b = linear_regression(np.array(x),np.array(y))
        y_pred = m*(i+n) + c
        Time_limit_delay_Degree_1[index].append(y_pred)
    index += 1

slope_curr, intercept_curr, cost_arr = linear_regression(np.array(x_axis), array, slope_curr=0, intercept_curr=0, epoch=100000, learn_rate=0.0001)
plt.plot([i for i in range(len(cost_arr)-1)], cost_arr[1:])
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.show()
