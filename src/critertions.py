import matplotlib.pyplot as plt
import numpy as np


# This critertion is not great, it only depends on the number of nodes in the graph...
def ICL_critertion(Q, n):
    return -1 / 2 * Q * (Q + 1) / 2 * np.log(n * (n - 1) / 2) - (Q - 1) / 2 * np.log(n)


def draw_critertion(n, min_Q=1, max_Q=20):
    y = []
    x = list(range(min_Q, max_Q + 1))
    for Q in x:
        y.append(ICL_critertion(Q, n))

    plt.plot(x, y)
    index = np.argmax(y)
    plt.axvline(x=x[index], color="red")
    plt.title("ICL criterion")
    plt.xlabel("Number of classes")
    plt.ylabel("Critertion")
    plt.xscale("log")
    plt.show()
