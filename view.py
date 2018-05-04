import matplotlib.pyplot as plt
import numpy as np



directory = '/home/ljf/LJF/LJFNet/train_0.5/data/'
datasetPredicted = 'results(xyz)_0.txt'
datasetGT = 'vo_gt.txt'

x_gt = []
y_gt = []
x_predicted = []
y_predicted = []
x_acc = []
y_acc = []

with open(directory + datasetGT) as f3:
        next(f3)  # skip the 3 header lines
        next(f3)
        next(f3)
        for line in f3:
            p0, p1, p2, p3, p4, p5, p6= line.split()
            #p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            #p3 = float(p3)
            #p4 = float(p4)
            #p5 = float(p5)
            #p6 = float(p6)
            x_gt.append(p1)
            y_gt.append(p2)
f3.close()

with open(directory + datasetPredicted) as f2:
        for line in f2:
            p1, p2, p3 = line.split()
            #p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            #p3 = float(p3)
            #p4 = float(p4)
            #p5 = float(p5)
            #p6 = float(p6)
            x_predicted.append(p1)
            y_predicted.append(p2)
f2.close()

fx = x_gt[1]
fy = y_gt[1]
for i in range(1,len(x_predicted),2):
	fx = fx + x_predicted[i]
	fy = fy + y_predicted[i]
	x_acc.append(fx)
	y_acc.append(fy)
'''
for i in range(len(x_gt)):
	x_gt[i] = x_gt[i] - x_gt[1]
	y_gt[i] = y_gt[i] - y_gt[1]
'''
plt.plot(x_gt, y_gt, linewidth=1, linestyle="--", color="orange")
plt.plot(x_acc, y_acc)
plt.title("matplotlib")
plt.xlabel("height")
plt.ylabel("width")
#plt.legend(["Y","Z"], loc="upper right")
plt.grid(True)
plt.show()