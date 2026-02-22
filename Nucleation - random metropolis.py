from os import times
import numpy as np
import random
import matplotlib.pyplot as plt
from numba import njit

def generate_random_grid(Nx,Ny):

    LV_grid= np.random.choice([-1, 1], size=(Ny, Nx))
    
    #print(LV_grid)
    return (LV_grid)


#Following function is to create the grid we will force into an equlibrium state and continue to use for all analysis. Note to change variables such as Nx, Ny, J, B, no_sweeps etc..
#for future use. 
#Change the file name of the saved grid as well for new interations. 
@njit
def grid_sweep_equilibrium(Nx,Ny,J,B,kT,no_sweeps, grid): #,snapshot_interval = 1):
    avg_mag_arr = []

    
    sweep_arr = np.arange(0,no_sweeps+1)

    for k in range (0,no_sweeps+1):
        avg_magnetisation = np.sum(grid)/(Nx*Ny)
        avg_mag_arr.append(avg_magnetisation)

        #running average code: May not need here yet.
        
        '''if k > 0:
            running_avg.append(sum(avg_mag_arr)/k)
'''
        
        # --- 2. Save a snapshot of the current grid for animation ---
        # (this does NOT change the grid: it's just a copy)
        '''
        if snapshot_interval is not None and k % snapshot_interval == 0:
    
            snapshots.append(grid.copy())
       '''

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

    #Plot the results:
    #filename = f"equil_state_{Nx}x{Ny}_J={J}_kT={kT}_B={B}.npy"
    # Formatting to 2 decimal places ensures they are clean strings
  


   
   
    return sweep_arr, avg_mag_arr, grid #, running_avg, snapshots

    return sweep_arr, avg_mag_arr, grid #, running_avg, snapshots


#Animation thing:

'''
import imageio
import os

def animate_snapshots(snapshots, 
                      show_animation=True, 
                      save_gif=False, 
                      save_frames=False, 
                      gif_name="ising_evolution.gif",
                      frame_prefix="ising_frame",
                      fps=5):
    """
    Takes a list of grid snapshots and:
    - plays an animation (optional)
    - saves a GIF (optional)
    - saves each frame as a PNG (optional)
    """

    # ---------- 1. Show animation in a window ----------
    if show_animation:
        plt.ion()
        fig, ax = plt.subplots()
        img = ax.imshow(snapshots[0], cmap='gray', vmin=-1, vmax=1)
        plt.title("Ising Model Evolution")

        for snap in snapshots:
            img.set_data(snap)
            plt.draw()
            plt.pause(0.1)  # animation speed

        plt.ioff()
        plt.show()

    # ---------- 2. Save frames as images ----------
    frame_files = []
    if save_frames:
        for i, snap in enumerate(snapshots):
            filename = f"{frame_prefix}_{i:04d}.png"
            plt.figure(figsize=(4, 4))
            plt.imshow(snap, cmap='gray', vmin=-1, vmax=1)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
            frame_files.append(filename)
        print(f"Saved {len(frame_files)} individual frames.")

    # ---------- 3. Save GIF ----------
    if save_gif:
        if not save_frames:
            # Generate frames in memory if no PNGs saved
            frame_files = []
            for i, snap in enumerate(snapshots):
                temp_name = f"_temp_frame_{i:04d}.png"
                plt.figure(figsize=(4, 4))
                plt.imshow(snap, cmap='gray', vmin=-1, vmax=1)
                plt.axis('off')
                plt.savefig(temp_name, dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()
                frame_files.append(temp_name)

        # Build GIF
        images = [imageio.imread(fname) for fname in frame_files]
        imageio.mimsave(gif_name, images, fps=fps)
        print(f"Saved GIF as: {gif_name}")

        # Clean up temporary files (only if they were generated internally)
        for fname in frame_files:
            if fname.startswith("_temp_frame_"):
                os.remove(fname)

'''





def steady_state(running_avg):


    threshold = 0.05* abs(running_avg[-1]-running_avg[0]) 
    for i in range(0, len(running_avg)):
        difference = abs(running_avg[i]-running_avg[-1])
        if difference < threshold:
            return i

    return None

@njit
def grid_sweep_nucleation(saved_grid, Nx, Ny, J, B, kT): #,snapshot_interval = 1):
    grid = saved_grid
    #print(grid)
    #perform the 'quench'
  
    carlo_sweeps = 0
    stop_cond = False
    avg_mag_arr = []

    while stop_cond == False and carlo_sweeps < 100000:
        avg_magnetisation = np.sum(grid)/(Nx*Ny)
        avg_mag_arr.append(avg_magnetisation)
        carlo_sweeps +=1
        if avg_magnetisation < -0.90:
            stop_cond = True
     


       
    

        N = Nx * Ny
    # One "sweep" = N random attempts
        for k in range(N):
        # Pick a random site
            i = np.random.randint(0, Ny)
            j = np.random.randint(0, Nx)
            
            # Calculate dE 
            f = grid[(i-1)%Ny,j] + grid[(i+1)%Ny,j] + grid[i,(j-1)%Nx] + grid[i,(j+1)%Nx]
            dE = 2 * grid[i,j] * ((J * f) + B)
            
            # Metropolis logic
            if dE <= 0 or np.random.rand() < np.exp(-dE/kT):
                grid[i,j] *= -1
       
                
        
                   
    #avg_magnetisation = np.sum(grid)/(Nx*Ny)
    #return avg_magnetisation

    #Plot the results:
    

    sweep_arr = np.arange(0,carlo_sweeps)
    nucleation_time = len(sweep_arr)
    
    return avg_mag_arr, sweep_arr, nucleation_time


'''
Nx = int(input("Enter number of grid points in x-direction (Nx): "))
Ny = int(input("Enter number of grid points in y-direction (Ny): "))
grid = generate_random_grid(Nx, Ny)
#print(grid)
#print("Avg magnetisation is: ", np.sum(grid)/ (Nx*Ny))

J = float(input("Enter interaction strength (J): "))
B = float(input("Enter external magnetic field (B): "))
no_sweeps = int(input("Enter number of sweeps: "))
kT = 1
sweep_arr = np.arange(0,no_sweeps+1)

avg_mag_arr,grid_equilibrium = grid_sweep_equilibrium(grid,Nx,Ny,J,B, no_sweeps,kT )




np.save('equil_state_30x30_J=0,5_kT=1_B=0_05.npy', grid_equilibrium)

#saved_state = np.load('equil_state.npy')

plt.plot(sweep_arr, avg_mag_arr)
plt.xlabel('Number of Sweeps')
plt.ylabel('Average Magnetisation, <M>')
plt.title('Average Magnetisation vs Number of Sweeps')
plt.show()
'''
print("Started6")

#grid_sweep_equilibrium(64,64,0.5,0.015,1,5000)
#avg_mag_arr, equil_grid = grid_sweep_equilibrium()

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def data_sweeps(L,J,B,kT,num_runs, plot):
    saved_grid = np.load(f'equil_state_{L}x{L}_J={J}_kT={kT}_B={abs(B)}.npy')
    
    filename = f"{L}x{L}_nucleation_J={J}_kT={kT}_B={B}.txt"
    
    B = -abs(B)  # Ensure B is negative for nucleation
    nucleation_time_arr = []
    print (f"Starting nucleation {num_runs} runs. Saving to file {filename}")
    for i in range(num_runs):
        print(f"Starting run {i+1} of {num_runs}")
        saved_grid = np.load(f'equil_state_{L}x{L}_J={J}_kT={kT}_B={abs(B)}.npy')
        avg_mag_arr, sweep_arr, nucleation_time = grid_sweep_nucleation(saved_grid, L,L,J,B,kT)

        nucleation_time_arr.append(nucleation_time)
        with open(filename, 'a') as f:
            f.write(f"{nucleation_time}\n")

    
    print(f"Nucleation of {num_runs} runs complete. Nucleation times saved to {filename}")

    #grid_sweep_nucleation(saved_grid, 64,64,0.5,-0.015,1)
    #avg_mag_arr, sweep_arr = grid_sweep_nucleation(saved_grid, 64,64,0.5,-0.015,1)
    if plot == True:
        plt.plot(sweep_arr, avg_mag_arr)
        plt.xlabel('Number of Sweeps')
        plt.ylabel('Average Magnetisation, <M>')
        plt.title('Average Magnetisation vs Number of Sweeps')
        plt.show()
    print("Finished")



def survival_model(t, nu, td):
    return -nu * (t - td)

def data_analysis(L,J,B,kT, plot):
    filename = f"{L}x{L}_nucleation_J={J}_kT={kT}_B={B}.txt"
    nucleation_times = np.loadtxt(filename)
    sorted_times = np.sort(nucleation_times)

    N = len(sorted_times)
    indices = np.arange(1, N + 1)
    Survival_probabilities = (N - indices) / N

    ln_survival = np.log(Survival_probabilities)
    
    mask_inf = ln_survival != -np.inf

    # Apply the mask to both arrays
    ln_survival = ln_survival[mask_inf]
    sorted_times = sorted_times[mask_inf]

    

    
    mask = (ln_survival < -0.5) & (ln_survival > -3.5)
    popt, pcov = curve_fit(survival_model, sorted_times[mask], ln_survival[mask])
    nu_fit, td_fit = popt
    perr = np.sqrt(np.diag(pcov)) # Standard deviation of the parameters
    print(f"avg time {1/nu_fit}")
    print(f"Nucleation Rate (nu): {nu_fit:.2e} +/- {perr[0]:.2e}")
    print(f"Delay Time (td): {td_fit:.2f} +/- {perr[1]:.2f}")
    print(f"Mean Nucleation Time (1/nu): {1/nu_fit:.2f} sweeps")

    # 5. Plot the result



   


    times = np.sort(nucleation_times)

    t0 = 2000
    survivors = times[times > t0]

    t_prime = np.median(survivors)
    tau = (t_prime - t0) / np.log(2)


    if plot == True:
        plt.plot(sorted_times, ln_survival, 'o')
        plt.xlabel('Nucleation Time (sweeps)')
        plt.ylabel('ln(Survival Probability)')
        plt.title('Survival Probability vs Nucleation Time')
        plt.show()

        plt.scatter(sorted_times, ln_survival, label='Data', s=10)
        plt.plot(sorted_times, survival_model(sorted_times, *popt), 'r--', 
                label=f'Fit: nu={nu_fit:.2e}, td={td_fit:.1f}')
        plt.ylabel('ln(Survival Probability)')
        plt.xlabel('Time (sweeps)')
        plt.legend()
        plt.show()
    return tau, nu_fit



def main():
    exit = False
    while exit == False:
        choice = input("Enter 1 for new equilibrium grid generation, 2 for nucleation data collection, 3 for data analysis, 4 for B sweep, or 'q' to quit: ")
        if choice == '1':
            Nx = int(input("Enter number of grid points in x-direction (Nx): "))
            Ny = int(input("Enter number of grid points in y-direction (Ny): "))
            grid = generate_random_grid(Nx, Ny)
            J = float(input("Enter interaction strength (J): "))
            B = abs(float(input("Enter external magnetic field (B): ")))
            no_sweeps = int(input("Enter number of sweeps: "))
            kT = float(input("Enter temperature (kT): "))
            sweep_arr, avg_mag_arr, grid = grid_sweep_equilibrium(Nx,Ny,J,B,kT,no_sweeps, grid)
            filename = f"equil_state_{Nx}x{Ny}_J={J}_kT={kT}_B={B}.npy"
            np.save(filename, grid)
            print(f"Successfully saved to: {filename}")
            print("Grid generated")
            plt.plot(sweep_arr, avg_mag_arr)
            plt.xlabel('Number of Sweeps')
            plt.ylabel('Average Magnetisation, <M>')
            plt.title('Average Magnetisation vs Number of Sweeps')
            plt.show()
        elif choice == '2':
            L = int(input("Enter grid size L (for LxL grid): "))
            J = float(input("Enter interaction strength (J): "))
            B = -abs(float(input("Enter external magnetic field (B): ")))
            kT = float(input("Enter temperature (kT): "))
            num_runs = int(input("Enter number of nucleation runs: "))
            data_sweeps(L,J,B,kT,num_runs, plot=True)
        elif choice == '3':
            L = int(input("Enter grid size L (for LxL grid): "))
            J = float(input("Enter interaction strength (J): "))
            B = -abs(float(input("Enter external magnetic field (B): ")))
            kT = float(input("Enter temperature (kT): "))
            
            Tau1 = data_analysis(L,J,B,kT, plot = True)
            print("Tau:",Tau1)


            
        elif choice.lower() == 'q':
            exit = True
        

        
        elif choice.lower() == '4':
            # Code here want to for a specfic kT value, run through various B field values
            L = int(input("Enter grid size L (for LxL grid): "))
            J = float(input("Enter interaction strength (J): "))
            B_lower = abs(float(input("Enter the lower external magnetic field (B): ")))
            B_upper = abs(float(input("Enter the upper external magnetic field (B): ")))
            kT = float(input("Enter temperature (kT): "))
            no_sweeps = int(input("Enter number of sweeps for equilibration: "))
            rand_grid = generate_random_grid(L, L)
           
            B_iteration = int((B_upper-B_lower)/0.01)
            MCC_iterations = int(input("Enter number of nucelation runs for each B"))
            for i in range (B_iteration+1):
                B = i*0.01 + B_lower
                sweep_arr, avg_mag_arr, grid = grid_sweep_equilibrium(L,L,J,B,kT,no_sweeps, rand_grid)
                filename_eq = f"equil_state_{L}x{L}_J={J}_kT={kT}_B={B}.npy"
                np.save(filename_eq, grid)
                print(f"Successfully saved to: {filename_eq}")
                print("Grid generated")

                data_sweeps(L,J,-B,kT,MCC_iterations, plot = False)
                Tau, nu_fit =  data_analysis(L,J,-B,kT, plot = False)
                filename_nuc = f"nuc_rate_{L}x{L}_J={J}_kT={kT}.txt"
                with open(filename_nuc, 'a') as f:
                    f.write(f"B = {B}, Fitting param Tau: {Tau}, Fitting param nu {nu_fit}\n")
            
                


                


        
        else:
            print("Invalid choice. Please try again.")


print("Updatedd86")

main()



