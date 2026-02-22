# --- INSTALLATION CELL (Run this once if needed) ---
import sys
import subprocess
import pkg_resources

required = {'numba', 'joblib', 'scipy', 'matplotlib', 'numpy'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print(f"Installing missing packages: {missing}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
    print("Packages installed. Please RESTART YOUR KERNEL if you see import errors.")

# --- MAIN CODE ---
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.optimize import curve_fit
import os
import time

print("Version 9:")

# Attempt to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    PARALLEL_AVAILABLE = True
except ImportError:
    print("Joblib not found. Running in serial mode (slower).")
    PARALLEL_AVAILABLE = False

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------


#For this code, we start grid from +1 or -1
def generate_random_grid(Nx, Ny):
    """Generates a random spin grid of size Nx x Ny."""
    LV_grid = np.random.choice([1, 1], size=(Ny, Nx))
    return LV_grid

# -------------------------------------------------------------------------
# 1. EQUILIBRIUM FUNCTION (Optimized)
# -------------------------------------------------------------------------
@njit(fastmath=True, nogil=True, cache=True)
def grid_sweep_equilibrium(Nx, Ny, J, B, beta, no_sweeps, grid):
    """
    Equilibrates the grid using Metropolis Monte Carlo.
    Optimized with Numba fastmath.
    """
    # Create array to store magnetization history
    avg_mag_arr = np.zeros(no_sweeps + 1)
    sweep_arr = np.arange(0, no_sweeps + 1)
    
    N = Nx * Ny
    
    for k in range(no_sweeps + 1):
        # Calculate full magnetization once per sweep
        current_mag = np.sum(grid) / N
        avg_mag_arr[k] = current_mag

        # Perform one Monte Carlo Sweep (N random attempts)


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
                energy_change = 2*J* grid[i,j] * f + 2*B*grid[i,j]
                exponent = -energy_change*beta

                #See if we keep the flip
                probability = np.exp(exponent)
                if probability > 1 or np.random.rand() < probability:
                    grid[i,j] = flipped
       
                    
    return sweep_arr, avg_mag_arr, grid


'''
# -------------------------------------------------------------------------
# 2. NUCLEATION RUNNER (Single Core Optimized)
# -------------------------------------------------------------------------
@njit(fastmath=True, nogil=True, cache=True)
def single_nucleation_run(saved_grid, Nx, Ny, J, B, beta):
    """
    Runs a single nucleation event until magnetization < -0.90.
    Returns the number of sweeps taken.
    """
    grid = saved_grid.copy()
    carlo_sweeps = 0
    N = Nx * Ny
    limit = 3000000  # Safety cap
    
    # Optimization: Track magnetization incrementally instead of summing entire grid
    current_mag_sum = np.sum(grid) 
    
    while carlo_sweeps < limit:
        # Check condition - Paper uses when m = 0.7
        if (current_mag_sum / N) < 0.7:
            return carlo_sweeps

        carlo_sweeps += 1

       
       
       
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
                exponent = -energy_change*beta

                #See if we keep the flip
                probability = np.exp(exponent)
                if probability > 1 or np.random.rand() < probability:
                    grid[i,j] = flipped
    return limit  # Return limit if we hit the cap (indicates failure to nucleate)
       '''
''' for _ in range(N):
            i = np.random.randint(0, Ny)
            j = np.random.randint(0, Nx)
            
            f = grid[(i-1)%Ny, j] + grid[(i+1)%Ny, j] + grid[i, (j-1)%Nx] + grid[i, (j+1)%Nx]
            dE = 2 * grid[i, j] * (J * f + B)
            


            
            #Not sure about this right now, will test
            accept = False
            if dE <= 0:
                accept = True
            elif np.random.rand() < np.exp(-beta * dE):
                accept = True
            
            if accept:
                # If we flip, the change in sum is -2 * old_spin
                current_mag_sum -= 2 * grid[i, j]
                grid[i, j] *= -1
                '''

@njit(fastmath=True, nogil=True, cache=True)
def single_nucleation_run(saved_grid, Nx, Ny, J, B, beta):
    grid = saved_grid.copy()
    N = Nx * Ny
    limit = 3000000  
    
    # 1. PRE-CALCULATE METROPOLIS PROBABILITIES
    # There are only 5 possible neighbor sums in a 2D lattice: -4, -2, 0, 2, 4
    # dE = 2 * s * (J * sum_neighbors + B)
    # We pre-calculate exp(-beta * dE) for all possible neighbor sums
    possible_sums = np.array([-4, -2, 0, 2, 4])
    prob_lookup_pos = np.zeros(5) # For spin +1
    prob_lookup_neg = np.zeros(5) # For spin -1
    
    for idx in range(5):
        s_sum = possible_sums[idx]
        # Probability for flipping a +1 spin
        dE_pos = 2 * 1 * (J * s_sum + B)
        prob_lookup_pos[idx] = np.exp(-beta * dE_pos)
        # Probability for flipping a -1 spin
        dE_neg = 2 * (-1) * (J * s_sum + B)
        prob_lookup_neg[idx] = np.exp(-beta * dE_neg)

    # 2. INCREMENTAL MAGNETIZATION TRACKING
    # This prevents summing the whole grid every loop
    current_mag_sum = 0.0
    for i in range(Ny):
        for j in range(Nx):
            current_mag_sum += grid[i, j]
            
    carlo_sweeps = 0
    while carlo_sweeps < limit:
        # Check condition using the paper's 0.7 cutoff [cite: 113, 118]
        if (current_mag_sum / N) < 0.7:
            return carlo_sweeps

        carlo_sweeps += 1

        # 3. SEQUENTIAL UPDATE SWEEP (Strictly row-by-row) [cite: 80, 246]
        for i in range(Ny):
            for j in range(Nx):
                s_i = grid[i, j]
                
                # Nearest Neighbors Sum [cite: 79, 100]
                neighbors_sum = (grid[(i-1)%Ny, j] + grid[(i+1)%Ny, j] + 
                                 grid[i, (j-1)%Nx] + grid[i, (j+1)%Nx])
                
                # Map neighbor sum to lookup index (e.g., -4 -> 0, 0 -> 2, 4 -> 4)
                idx = int((neighbors_sum + 4) // 2)
                
                if s_i == 1:
                    p = prob_lookup_pos[idx]
                else:
                    p = prob_lookup_neg[idx]

                # Metropolis Acceptance [cite: 81, 96]
                if p >= 1.0 or np.random.rand() < p:
                    grid[i, j] = -s_i
                    # Update the magnetization sum incrementally
                    current_mag_sum -= 2 * s_i 
                    
    return limit

# -------------------------------------------------------------------------
# 3. DATA COLLECTION (Parallelized)
# -------------------------------------------------------------------------
def data_sweeps(L, J, B, beta, num_runs, plot=False):
    """
    Runs multiple nucleation events in PARALLEL using joblib.
    """
    B_quench = -abs(B) 
    filename_in = f'equil_state_{L}x{L}_J={J}_beta={beta}_B={abs(B)}.npy'
    
    try:
        base_grid = np.load(filename_in)
    except FileNotFoundError:
        print(f"Error: Could not find file {filename_in}. Run equilibrium (Choice 1) first.")
        return [], [] # Return empty if failed


    filename_out = f"{L}x{L}_nucleation_J={J}_beta={beta}_B={B_quench}.txt"
    print(f"Starting {num_runs} nucleation runs. Saving to {filename_out}")
    
    # -- Parallel Execution --
    start_time = time.time()
    
    if PARALLEL_AVAILABLE:
        # n_jobs=-1 uses all available cores
        nucleation_times = Parallel(n_jobs=-1)(
            delayed(single_nucleation_run)(base_grid, L, L, J, B_quench, beta) 
            for _ in range(num_runs)
        )
    else:
        # Fallback for serial execution
        nucleation_times = []
        for i in range(num_runs):
            t = single_nucleation_run(base_grid, L, L, J, B_quench, beta)
            nucleation_times.append(t)

    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds.")

    # Save to file
    with open(filename_out, 'a') as f:
        for val in nucleation_times:
            f.write(f"{val}\n")
            
    # Optional Plotting
    if plot:
        plt.hist(nucleation_times, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Nucleation Time (sweeps)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Nucleation Times')
        plt.show()

    return nucleation_times


# -------------------------------------------------------------------------
# 4. ANALYSIS
# -------------------------------------------------------------------------
def survival_model(t, nu, td):
    return -nu * (t - td)

def data_analysis(L, J, B, beta, plot=True):
    filename = f"{L}x{L}_nucleation_J={J}_beta={beta}_B={B}.txt"
    try:
        nucleation_times = np.loadtxt(filename)
    except OSError:
        print(f"File {filename} not found.")
        return 0, 0

    # Ensure array is 1D (loadtxt might return 0D if only 1 point)
    nucleation_times = np.atleast_1d(nucleation_times)

    average_time = np.mean(nucleation_times)
    if len(nucleation_times) < 2:
        print("Not enough data for analysis.")
        return 0, 0

    sorted_times = np.sort(nucleation_times)
    N = len(sorted_times)
    indices = np.arange(1, N + 1)
    Survival_probabilities = (N - indices) / N

    with np.errstate(divide='ignore'):
        ln_survival = np.log(Survival_probabilities)

    mask_inf = np.isfinite(ln_survival)
    ln_survival = ln_survival[mask_inf]
    sorted_times = sorted_times[mask_inf]

    # Filter for linear region
    mask = (ln_survival < -0.1) & (ln_survival > -3.5)
    
    if len(sorted_times[mask]) < 2:
        print("Data insufficient for curve fitting in range [-0.1, -3.5].")
        return 0, 0

    try:
        popt, pcov = curve_fit(survival_model, sorted_times[mask], ln_survival[mask])
        nu_fit, td_fit = popt
        perr = np.sqrt(np.diag(pcov))
    except:
        print("Curve fit failed.")
        return 0, 0

    # Calculate Tau (Median Method)
    t0 = 0
    survivors = sorted_times[sorted_times > t0]
    if len(survivors) > 0:
        t_prime = np.median(survivors)
        tau_median = (t_prime - t0) / np.log(2)
    else:
        tau_median = 0

    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(sorted_times, ln_survival, label='Data', s=10)
        plt.plot(sorted_times[mask], survival_model(sorted_times[mask], *popt), 'r--', 
                 label=f'Fit: nu={nu_fit:.2e}, td={td_fit:.1f}')
        plt.xlabel('Nucleation Time (sweeps)')
        plt.ylabel('ln(Survival Probability)')
        plt.title(f'Survival Analysis: J={J}, B={B}, beta={beta}')
        plt.legend()
        plt.show()

        print(f"Nucleation Rate (nu): {nu_fit:.2e} +/- {perr[0]:.2e}")
        print(f"Mean Time (1/nu): {1/nu_fit:.2f} sweeps")

    return tau_median, nu_fit, average_time





B_val, Tau, nu, avg_time = np.loadtxt("64x64Paperdata.txt", unpack =True)
H = 1/B_val*1/0.550858
logtime = np.log(avg_time)
plt.scatter(H,logtime)
plt.xlabel('H')
plt.ylabel('log(avg_time)')
plt.show()
# -------------------------------------------------------------------------
# MAIN INTERFACE (Now asking for Beta directly)
# -------------------------------------------------------------------------
def main():
    exit_prog = False
    while not exit_prog:
        print("\n--- OPTIMIZED ISING MODEL ---")
        print("1. Generate Equilibrium Grid (Prepare state)")
        print("2. Run Nucleation Data Collection (Parallelized)")
        print("3. Analyze Data")
        print("4. Full B-Field Sweep (Equilibrate + Nucleate)")
        print("q. Quit")
        
        choice = input("Enter choice: ")

        if choice == '1':
            Nx = int(input("Enter Nx: "))
            Ny = int(input("Enter Ny: "))
            J = float(input("Enter J: "))
            B = abs(float(input("Enter B (Equil Field): ")))
            # UPDATED: Ask for Beta directly
            beta = float(input("Enter Beta (1/kT): ")) 
            no_sweeps = int(input("Enter number of sweeps: "))
            
            grid = generate_random_grid(Nx, Ny)
            
            print("Running Equilibrium...")
            start = time.time()
            sweep_arr, avg_mag_arr, grid = grid_sweep_equilibrium(Nx, Ny, J, B, beta, no_sweeps, grid)
            print(f"Done in {time.time()-start:.2f}s")
            
            filename = f"equil_state_{Nx}x{Ny}_J={J}_beta={beta}_B={B}.npy"
            np.save(filename, grid)
            print(f"Saved to: {filename}")
            
            plt.plot(sweep_arr, avg_mag_arr)
            plt.xlabel('Sweeps')
            plt.ylabel('<M>')
            plt.show()

        elif choice == '2':
            L = int(input("Enter grid size L: "))
            J = float(input("Enter J: "))
            B = -abs(float(input("Enter B (Quench Field, e.g. 0.01): ")))
            # UPDATED: Ask for Beta directly
            beta = float(input("Enter Beta (1/kT): ")) 
            num_runs = int(input("Enter number of runs: "))
            
            data_sweeps(L, J, B, beta, num_runs, plot=True)

        elif choice == '3':
            L = int(input("Enter grid size L: "))
            J = float(input("Enter J: "))
            B = -abs(float(input("Enter B (Quench Field): ")))
            # UPDATED: Ask for Beta directly
            beta = float(input("Enter Beta (1/kT): "))
            
            Tau, nu = data_analysis(L, J, B, beta, plot=True)
            print(f"Calculated Tau (Median): {Tau}")

        elif choice == '4':
            L = int(input("Enter grid size L: "))
            J = float(input("Enter J: "))
            B_lower = abs(float(input("Enter lower B (start): ")))
            B_upper = abs(float(input("Enter upper B (end): ")))
            step_size = float(input("Enter step size for B: "))
            
            # UPDATED: Ask for Beta directly
            beta = float(input("Enter Beta (1/kT): "))
            no_sweeps = int(input("Enter equilibration sweeps: "))
            MCC_iterations = int(input("Enter nucleation runs per B: "))
            
            # Create range of B values
            num_steps = int(round((B_upper - B_lower) / step_size)) + 1
            B_values = np.linspace(B_lower, B_upper, num_steps)

            for B_val in B_values:
                B_val = float(f"{B_val:.4f}") # Clean float precision
                print(f"\n--- Processing B = {B_val} ---")
                
                # 1. Equilibrate
                rand_grid = generate_random_grid(L, L)
                _, _, eq_grid = grid_sweep_equilibrium(L, L, J, B_val, beta, no_sweeps, rand_grid)
                
                filename_eq = f"equil_state_{L}x{L}_J={J}_beta={beta}_B={B_val}.npy"
                np.save(filename_eq, eq_grid)
                
                # 2. Nucleate (Parallel)
                data_sweeps(L, J, -B_val, beta, MCC_iterations, plot=False)
                
                # 3. Analyze
                Tau, nu_fit, avg_time = data_analysis(L, J, -B_val, beta, plot=False)
                
                filename_nuc = f"nuc_rate_{L}x{L}_J={J}_beta={beta}.txt"
                with open(filename_nuc, 'a') as f:
                    f.write(f"{B_val} {Tau} {nu_fit} {avg_time}\n")
                    
        elif choice.lower() == 'q':
            exit_prog = True

if __name__ == "__main__":
    main()



