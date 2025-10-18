import numpy as np
import random
import matplotlib.pyplot as plt


def generate_grid(Nx,Ny):

    LV_grid= np.random.choice([-1, 1], size=(Ny, Nx))
    
    #print(LV_grid)
    return (LV_grid)


def grid_sweep(grid, Nx,Ny,J,B, no_sweeps,kT):
    avg_mag_arr = []

    for k in range (0,no_sweeps+1):
        avg_magnetisation = np.sum(grid)/(Nx*Ny)
        avg_mag_arr.append(avg_magnetisation)

        for i in range(0,Ny):
        #Pseudocode yes Ny-1 but remember python is exclusive on upper bound
        
            for j in range (0,Nx):
                # Flip the current spin
                original = grid[i,j]
                flipped = -grid[i,j]

                
                #Tally the 4 nearest neighbours, considering the periodic boundary conditions - Modulo thing works by:
                #If at the start, we get a negative modulo, causing to wrap to the end (how many we need to get from -1 to -(Nx-1)) i.e takes to Nx-1
                #If at the end, we get Nx-1 + 1 = Nx, which modulo with Nx is 0, taking us to start.
                #Similar logic to the top and bottom
                #If in the middle, Nx and Ny > i or j, so modulo returns i or j (i.e divides 0 times, and remainder is i or j)

                
                f = grid[(i-1)%Ny,j] + grid[(i+1)%Ny,j] + grid[i,(j-1)%Nx] + grid[i,(j+1)%Nx]
                energy_change = 2*J * grid[i,j] * f + 2*B*grid[i,j]
                beta = -energy_change/kT

                #See if we keep the flip
                probability = np.exp(beta)
                if probability > 1 or np.random.rand() < probability:
                    grid[i,j] = flipped
                
    #avg_magnetisation = np.sum(grid)/(Nx*Ny)
    #return avg_magnetisation
    return avg_mag_arr

            




Nx = int(input("Enter number of grid points in x-direction (Nx): "))
Ny = int(input("Enter number of grid points in y-direction (Ny): "))
grid = generate_grid(Nx, Ny)
print(grid)
print("Avg magnetisation is: ", np.sum(grid)/ (Nx*Ny))

J = float(input("Enter interaction strength (J): "))
B = float(input("Enter external magnetic field (B): "))
no_sweeps = int(input("Enter number of sweeps: "))
kT = 1
sweep_arr = np.arange(0,no_sweeps+1)
avg_mag_arr = grid_sweep(grid, Nx, Ny, J, B, no_sweeps, kT)
print("Final magnetisation after ", no_sweeps, "sweeps is: ", avg_mag_arr[-1])


plt.plot(sweep_arr, avg_mag_arr)
plt.xlabel('Number of Sweeps')
plt.ylabel('Average Magnetisation')
plt.title('Average Magnetisation vs Number of Sweeps')
plt.show()


