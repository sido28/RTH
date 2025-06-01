import numpy as np
import matplotlib.pyplot as plt
from rth import RTH
from benchmarks import get_functions_details, func_plot

if __name__ == "__main__":
    Npop = 30
    Function_name = 'F5'  # You can change from 'F1' to 'F23'
    Tmax = 1000
    lb, ub, dim, fobj = get_functions_details(Function_name)
    Best_fit, Best_pos, Convergence_curve = RTH(Npop, Tmax, lb, ub, dim, fobj)

    # Draw search space
    plt.figure(figsize=(10, 4))
    # plt.subplot(121, projection='3d')
    func_plot(Function_name)
    plt.title('Search space')
    
    
    # Draw objective space
    plt.subplot(122)
    plt.semilogy(Convergence_curve, color='r')
    plt.title('Convergence curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best score obtained so far')
    plt.tight_layout()
    plt.grid(True)
    plt.legend(['RTH'])
    plt.show()

    print("The best solution is:", Best_pos)
    print("The best fitness value is:", Best_fit)