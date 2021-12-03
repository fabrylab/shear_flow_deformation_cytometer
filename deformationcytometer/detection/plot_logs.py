import matplotlib.pyplot as plt
import numpy as np
import glob

def loadtxt(file):
    data = []
    with open(file, "r") as fp:
        for line in fp:
            time, name, onof, index = line.split()
            data.append([float(time), int(onof), int(index)])
    return np.array(data)

def plot(x, y, offsety, **kwargs):
    x2 = np.vstack([x, x]).T.ravel()
    y2 = np.vstack([np.hstack([0, y[:-1]]), y]).T.ravel()
    plt.fill_between(x2, y2+offsety, y2*0+offsety, **kwargs)

files = sorted(glob.glob("log_*.txt"))
print(files)

min_time = np.inf
data = loadtxt(files[0])
min_time = data[0, 0]

for index, file in enumerate(files):
    data = loadtxt(file)
    plot(data[:, 0] - min_time, data[:, 1]*0.8, index)
    for d in data[-100:]:
        if d[1] and d[0] > 700:
            plt.text(d[0] - min_time, index, int(d[2]), rotation=90)

plt.show()
