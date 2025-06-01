import numpy as np
import matplotlib.pyplot as plt

def get_functions_details(F):
    # Returns lb, ub, dim, fobj (function handle)
    def F1(x):
        return np.sum(x ** 2)
    def F2(x):
        return np.sum(np.abs(x)) + np.prod(np.abs(x))
    def F3(x):
        return sum([np.sum(x[:i+1]) ** 2 for i in range(len(x))])
    def F4(x):
        return np.max(np.abs(x))
    def F5(x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)
    def F6(x):
        return np.sum(np.abs(x + 0.5) ** 2)
    def F7(x):
        return np.sum(np.arange(1, len(x)+1) * (x ** 4)) + np.random.rand()
    def F8(x):
        return np.sum(-x * np.sin(np.sqrt(np.abs(x))))
    def F9(x):
        return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * len(x)
    def F10(x):
        dim = len(x)
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim)) - \
               np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1)
    def F11(x):
        dim = len(x)
        return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1)))) + 1
    def Ufun(x, a, k, m):
        return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)
    def F12(x):
        dim = len(x)
        res = (np.pi / dim) * (10 * (np.sin(np.pi * (1 + (x[0] + 1) / 4))) ** 2 +
               np.sum(((x[:-1] + 1) / 4) ** 2 * (1 + 10 * (np.sin(np.pi * (1 + (x[1:] + 1) / 4))) ** 2)) +
               ((x[-1] + 1) / 4) ** 2)
        return res + np.sum(Ufun(x, 10, 100, 4))
    def F13(x):
        dim = len(x)
        res = 0.1 * ((np.sin(3 * np.pi * x[0])) ** 2 +
                     np.sum((x[:-1] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[1:])) ** 2)) +
                     ((x[-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[-1])) ** 2))
        return res + np.sum(Ufun(x, 5, 100, 4))
    def F14(x):
        aS = np.array([[-32, -16, 0, 16, 32]*5,
                       [-32]*5 + [-16]*5 + [0]*5 + [16]*5 + [32]*5])
        bS = [np.sum((x - aS[:, j]) ** 6) for j in range(25)]
        return (1 / 500 + np.sum(1.0 / (np.arange(1, 26) + bS))) ** -1
    def F15(x):
        aK = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
        bK = np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
        bK = 1.0 / bK
        return np.sum((aK - ((x[0] * (bK ** 2 + x[1] * bK)) / (bK ** 2 + x[2] * bK + x[3]))) ** 2)
    def F16(x):
        return 4 * (x[0] ** 2) - 2.1 * (x[0] ** 4) + (x[0] ** 6) / 3 + x[0] * x[1] - 4 * (x[1] ** 2) + 4 * (x[1] ** 4)
    def F17(x):
        return (x[1] - x[0] ** 2 * 5.1 / (4 * np.pi ** 2) + 5 / np.pi * x[0] - 6) ** 2 + \
               10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10
    def F18(x):
        return (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) * \
               (30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
    def F19(x):
        aH = np.array([[3, 10, 30], [.1, 10, 35], [3, 10, 30], [.1, 10, 35]])
        cH = np.array([1, 1.2, 3, 3.2])
        pH = np.array([[.3689, .117, .2673], [.4699, .4387, .747], [.1091, .8732, .5547], [.03815, .5743, .8828]])
        return -np.sum([cH[i] * np.exp(-np.sum(aH[i, :] * (x - pH[i, :]) ** 2)) for i in range(4)])
    def F20(x):
        aH = np.array([[10, 3, 17, 3.5, 1.7, 8], [.05, 10, 17, .1, 8, 14],
                       [3, 3.5, 1.7, 10, 17, 8], [17, 8, .05, 10, .1, 14]])
        cH = np.array([1, 1.2, 3, 3.2])
        pH = np.array([[.1312, .1696, .5569, .0124, .8283, .5886],
                       [.2329, .4135, .8307, .3736, .1004, .9991],
                       [.2348, .1415, .3522, .2883, .3047, .6650],
                       [.4047, .8828, .8732, .5743, .1091, .0381]])
        return -np.sum([cH[i] * np.exp(-np.sum(aH[i, :] * (x - pH[i, :]) ** 2)) for i in range(4)])
    def F21(x):
        aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                        [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        cSH = np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
        return -np.sum([1 / ((np.dot(x - aSH[i, :], x - aSH[i, :]) + cSH[i])) for i in range(5)])
    def F22(x):
        aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                        [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        cSH = np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
        return -np.sum([1 / ((np.dot(x - aSH[i, :], x - aSH[i, :]) + cSH[i])) for i in range(7)])
    def F23(x):
        aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                        [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        cSH = np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
        return -np.sum([1 / ((np.dot(x - aSH[i, :], x - aSH[i, :]) + cSH[i])) for i in range(10)])

    # Mapping of function name to properties
    mapping = {
        'F1':   (F1, -100, 100, 30),
        'F2':   (F2, -10, 10, 10),
        'F3':   (F3, -100, 100, 10),
        'F4':   (F4, -100, 100, 10),
        'F5':   (F5, -30, 30, 10),
        'F6':   (F6, -100, 100, 10),
        'F7':   (F7, -1.28, 1.28, 10),
        'F8':   (F8, -500, 500, 10),
        'F9':   (F9, -5.12, 5.12, 10),
        'F10':  (F10, -32, 32, 10),
        'F11':  (F11, -600, 600, 10),
        'F12':  (F12, -50, 50, 10),
        'F13':  (F13, -50, 50, 10),
        'F14':  (F14, -65.536, 65.536, 2),
        'F15':  (F15, -5, 5, 4),
        'F16':  (F16, -5, 5, 2),
        'F17':  (F17, np.array([-5, 0]), np.array([10, 15]), 2),
        'F18':  (F18, -2, 2, 2),
        'F19':  (F19, 0, 1, 3),
        'F20':  (F20, 0, 1, 6),
        'F21':  (F21, 0, 10, 4),
        'F22':  (F22, 0, 10, 4),
        'F23':  (F23, 0, 10, 4),
    }
    fobj, lb, ub, dim = mapping[F]
    return lb, ub, dim, fobj
    
def func_plot(func_name):
    lb, ub, dim, fobj = get_functions_details(func_name)
    if func_name == 'F1': x = np.arange(-100, 101, 2); y = x
    elif func_name == 'F2': x = np.arange(-100, 101, 2); y = x
    elif func_name == 'F3': x = np.arange(-100, 101, 2); y = x
    elif func_name == 'F4': x = np.arange(-100, 101, 2); y = x
    elif func_name == 'F5': x = np.arange(-200, 201, 2); y = x
    elif func_name == 'F6': x = np.arange(-100, 101, 2); y = x
    elif func_name == 'F7': x = np.arange(-1, 1.01, 0.03); y = x
    elif func_name == 'F8': x = np.arange(-500, 501, 10); y = x
    elif func_name == 'F9': x = np.arange(-5, 5.1, 0.1); y = x
    elif func_name == 'F10': x = np.arange(-20, 20.5, 0.5); y = x
    elif func_name == 'F11': x = np.arange(-500, 501, 10); y = x
    elif func_name == 'F12': x = np.arange(-10, 10.1, 0.1); y = x
    elif func_name == 'F13': x = np.arange(-5, 5.08, 0.08); y = x
    elif func_name == 'F14': x = np.arange(-100, 101, 2); y = x
    elif func_name == 'F15': x = np.arange(-5, 5.1, 0.1); y = x
    elif func_name == 'F16': x = np.arange(-1, 1.01, 0.01); y = x
    elif func_name == 'F17': x = np.arange(-5, 5.1, 0.1); y = x
    elif func_name == 'F18': x = np.arange(-5, 5.06, 0.06); y = x
    elif func_name == 'F19': x = np.arange(-5, 5.1, 0.1); y = x
    elif func_name == 'F20': x = np.arange(-5, 5.1, 0.1); y = x
    elif func_name == 'F21': x = np.arange(-5, 5.1, 0.1); y = x
    elif func_name == 'F22': x = np.arange(-5, 5.1, 0.1); y = x
    elif func_name == 'F23': x = np.arange(-5, 5.1, 0.1); y = x
    else: raise ValueError("Unknown function name")
    L = len(x)
    f = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            if func_name not in ['F15', 'F19', 'F20', 'F21', 'F22', 'F23']:
                f[i, j] = fobj(np.array([x[i], y[j]]))
            elif func_name == 'F15':
                f[i, j] = fobj(np.array([x[i], y[j], 0, 0]))
            elif func_name == 'F19':
                f[i, j] = fobj(np.array([x[i], y[j], 0]))
            elif func_name == 'F20':
                f[i, j] = fobj(np.array([x[i], y[j], 0, 0, 0, 0]))
            elif func_name in ['F21', 'F22', 'F23']:
                f[i, j] = fobj(np.array([x[i], y[j], 0, 0]))
    X, Y = np.meshgrid(x, y)
    fig = plt.gcf()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, f, linewidth=0, antialiased=False, cmap='viridis')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    ax.set_zlabel('fit')