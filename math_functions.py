import numpy as np

def rosenbrock(x, y):
    z = (1 - x)**2 + 100 * (y - x**2)**2
    return z

def michalewicz(x, y):
    m = 10
    z = -1 * np.sin(x) * np.power(np.sin(x**2/np.pi), 2*m)
    z -= np.sin(y) * np.power(np.sin(2*y**2/np.pi), 2*m)
    return z

if __name__ == "__main__":
    print(michalewicz(4, 1))
    print(rosenbrock(1.8, 1.1))