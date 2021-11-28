import numpy as np
import matplotlib.pyplot as plt

pr1 = [0, 0, 0.001, 0.271, 3.192, 4.526, 9.014, 10.36, 15.62, 17.13, 22.53, 24.15, 29.53, 31.22, 36.31, 37.77]
pr1 = np.array(pr1)
pr2 = [0, 0.0206, 0.4703, 1.537, 2.967, 4.840, 7.010, 9.400, 12.03, 14.74, 17.57, 20.41, 23.28, 26.37, 29.20, 32.04]
pr2 = np.array(pr2)
x = np.arange(15,31,1)
p1, = plt.plot(x,pr1)
p2, = plt.plot(x,pr2)
plt.xlabel('Step')
plt.ylabel('Probability, [%]')
l1 = plt.legend([p1, p2], ["Stand_still = FALSE", "Stand_still = TRUE"], loc='upper left')
plt.gca().add_artist(l1)

plt.show()