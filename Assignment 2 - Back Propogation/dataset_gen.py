import random
import math
import matplotlib.pyplot as plt
import numpy as np
import csv

'''
Class for (x,y) point
'''
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str((self.x, self.y))

'''
Class for Circle 
'''
class Circle:
    def __init__(self, origin, radius):
        self.origin = origin
        self.radius = radius

origin = Point(0, 0)
radius = 1
circle = Circle(origin, radius)

circle_array, rectangle_array = [], []
plt.figure()

'''
Fill random points inside the circle;
Set range as 50 for testing data and 100 for training data
Plot the data points in a figure
'''
for i in range(0, 50):
    p = random.random() * 2 * math.pi
    r = circle.radius * math.sqrt(random.random())
    x = math.cos(p) * r
    y = math.sin(p) * r
    circle_array.append([x,y,1])
    plt.scatter(x, y, color = 'blue')
    
    
'''
Fill random points outside the circle but inside the square boundary;
Set range as 50 for testing data and 100 for training data
Plot the data points in a figure
'''
while len(rectangle_array) != 50:
    x_point = random.uniform(-2,2)
    y_point = random.uniform(-2,2)
    if np.sqrt(x_point**2 + y_point**2) > 1:
        rectangle_array.append([x_point, y_point, 0])
        plt.scatter(x_point, y_point, color = 'red')    

'''
Add the data points to their respective csv files
Training = 100_data_points.csv
Testing = 50_data_points.csv
'''
with open('50_data_points.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    for row in circle_array:
        writer.writerow(row)
    for row in rectangle_array:
        writer.writerow(row)

'''
Plot the Circle and square class boundaries of the dataset
'''
plt.xlabel('X')
plt.ylabel('Y')
circle_1 = plt.Circle((0,0), 1, color='b', fill=False)
rectangle_1 = plt.Rectangle((-2,-2),4,4,linewidth=1,edgecolor='r',facecolor='none')
ax = plt.gca()
ax.add_artist(circle_1)
ax.add_patch(rectangle_1)
plt.show()

