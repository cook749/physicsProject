import numpy as np


def generate_grid(Nx,Ny):

    LV_grid= np.random.choice([-1, 1], size=(Ny, Nx))
    
    print(LV_grid)
    return (LV_grid)


def grid_sweep(grid, Nx,Ny):
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
            print (f)

            





Nx = int(input("Enter number of grid points in x-direction (Nx): "))
Ny = int(input("Enter number of grid points in y-direction (Ny): "))
grid = generate_grid(Nx, Ny)

print(grid_sweep(grid, Nx, Ny))

#2J = float(input("Enter interaction strength (J): "))
#B = float(input("Enter external magnetic field (B): "))
kT = 1



