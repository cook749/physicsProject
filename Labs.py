import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib
import scipy
import math
import pandas as pd
from joblib import Parallel, delayed

'''Code to calcualte resistance'''

def Resistance_Calculator(PD, CURRENT):
    resistance=np.array(abs(PD))/np.array(CURRENT)
    return resistance

'''Code to bin temperature and resistance data'''

def bin_temp_resistance(temperatures, resistances):
 

    # Ensure NumPy arrays AND force them to be 1-dimensional using .ravel()
    temperatures = np.array(temperatures).ravel()
    resistances = np.array(resistances).ravel()

    # --- Define and calculate Nobins based on desired bin width ---
    desired_bin_width = 0.1 # K

    min_temp = np.min(temperatures)
    max_temp = np.max(temperatures)
    
    # Calculate the number of bins needed
    Nobins = math.ceil((max_temp - min_temp) / desired_bin_width)
    
    # --- Create bins ---
    # Create bin edges using the calculated Nobins, ensuring the last edge is max_temp
    bin_edges = np.linspace(min_temp, max_temp, Nobins + 1) 
 
    bin_indices = np.digitize(temperatures, bin_edges) - 1  # shift to 0-based index

    # --- Compute mean resistance and temperature per bin ---
    binned_temps = []
    binned_resistances = []

    for i in range(Nobins):
        # 'mask' will now be 1D, correctly indexing the 1D 'resistances'
        mask = bin_indices == i
        if np.any(mask):
            binned_temps.append(np.mean(temperatures[mask]))
            binned_resistances.append(np.mean(resistances[mask]))

    return np.array(binned_temps), np.array(binned_resistances), bin_edges


'''Code for rolling average:'''



def calculate_rolling_average(temperatures, resistances, window_size):
    
   
    # 1. Ensure inputs are NumPy arrays for safety, and convert to 1D
    temperatures = np.array(temperatures).ravel()
    resistances = np.array(resistances).ravel()

    # 2. Combine into a DataFrame for easy rolling computation
    data = pd.DataFrame({'temp': temperatures, 'res': resistances})

    # 3. Calculate the rolling mean, centered on the current point.
    # .dropna() removes the NaN values at the beginning & end that couldn't form a full window.
    smoothed_data = data.rolling(window=window_size, center=True).mean().dropna()

    # 4. Extract and return the smoothed results as NumPy arrays
    smoothed_temps = smoothed_data['temp'].values
    smoothed_resistances = smoothed_data['res'].values

    return smoothed_temps, smoothed_resistances


'''Code to convert resistance to temperature using the calibration fit'''

def Resis_Temp_converter(PTresistance, model_params):
    def single_res(R):
        coefficients = [
            model_params[3]*model_params[0],
            -100*model_params[3]*model_params[0],
            model_params[2]*model_params[0],
            model_params[1]*model_params[0],
            model_params[0] - R
        ]
        roots = np.roots(coefficients)
        real_roots = roots[np.isreal(roots)].real
        actual_temp = real_roots[real_roots <= 0]
        return actual_temp + 273.15  # Kelvin to Celsius

    # Run in parallel using all CPU cores
    results = Parallel(n_jobs=-1, prefer="threads")(delayed(single_res)(R) for R in PTresistance)
    return np.array(results)


'''Model function for platinum thermometer calibration & chi squared'''
def model_function_PT(x, *params):
    return params[0] *(1+  params[1]*x+ params[2]*x**2 + (x-100)*params[3]*x**3)

def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params))/y_err)**2) 



def optimised_params(model_function,temperatures, resistance, inital_params): 
    popt, cov = scipy.optimize.curve_fit(model_function, # function to fit
                                        temperatures, # x data
                                        resistance, # y data
                                        #sigma=PT_model_resis_err[min_index:max_index], # array of error bars for the fit
                                        #absolute_sigma=True, # errors bars DO represent 1 std error
                                        p0=inital_params, # starting point for fit
                                        check_finite=True) # raise ValueError if NaN encountered (don't allow errors to pass)
    return popt, cov






'''Load data to be processed from files'''
TimeArr_Calib,PTPD_Calib, PTCurr_Calib, ITCTEMP_Calib, YBCOPD_Calib, YBCOCurr_Calib = np.loadtxt('0.04ampCalibration.txt', unpack=True)  # Load and unpack data
TimeArr,PTPD, PTCurr, ITCTEMP, YBCOPD, YBCOCurr = np.loadtxt('Test27.txt', unpack=True) 



YBCOresistance_Calib =Resistance_Calculator(YBCOPD_Calib, YBCOCurr_Calib)
PTresistance_Calib = Resistance_Calculator(PTPD_Calib, PTCurr_Calib)
YBCOresistance_DatCol =Resistance_Calculator(YBCOPD, YBCOCurr)
PTresistance_DatCol = Resistance_Calculator(PTPD, PTCurr)

#Trying to remove data:
'''
lower_bound = 59500
upper_bound = 62000
mask = ( TimeArr >= lower_bound) & (TimeArr <= upper_bound)


indices = np.where(mask)
print (indices[0][0])
print(indices[0][-1])

removedTemp = ITCTEMP[indices[0][0]:indices[0][-1]+1]
removedResistance = YBCOresistance[indices[0][0]:indices[0][-1]+1]
removedYBCOPD = YBCOPD[indices[0][0]:indices[0][-1]+1]
adjustedTemp = np.delete(ITCTEMP,indices)
adjustedResistance = np.delete(YBCOresistance,indices)
adjustedYBCOPD = np.delete(YBCOPD,indices)

print(len(ITCTEMP)-len(adjustedTemp))'''

#Code for Chi-squared for the platinum thermometer calibration:
CelsiusTemp_Calib = np.array(ITCTEMP_Calib)-273.15  # Convert to Celsius
CelsiusTemp_DatCol = np.array(ITCTEMP)-273.15  # Convert to Celsius

max_index = np.abs(ITCTEMP_Calib-273).argmin()
min_index = np.abs(ITCTEMP_Calib-77).argmin()
max_index_Dat = np.abs(ITCTEMP_Calib-94).argmin()
min_index_Dat = np.abs(ITCTEMP_Calib-87).argmin()

no_vals = len(PTresistance_Calib[min_index:max_index])
PT_model_resis_err = np.full(no_vals, 0.002)  # Constant error of 0.0002 Ohms

initial_values_PT = np.array([ 100,  3.8e-3,  -5.4e-7, -8.47603047e-12])
degrees_of_freedom_PT = PTresistance_Calib[min_index:max_index].size - initial_values_PT.size


popt,cov = optimised_params(model_function_PT, CelsiusTemp_Calib[min_index:max_index], PTresistance_Calib[min_index:max_index], initial_values_PT)
print('Optimised parameters = ', popt, '\n')


'''

chi_squared_min_PT = chi_squared(popt, model_function_PT, CelsiusTemp[min_index:max_index], PTresistance[min_index:max_index], PT_model_resis_err[min_index:max_index])
print('chi^2_min = {}'.format(chi_squared_min_PT))
print('reduced chi^2 = {}'.format(chi_squared_min_PT/degrees_of_freedom_PT))
print('P(chi^2_min, DoF) = {}'.format(scipy.stats.chi2.sf(chi_squared_min_PT, degrees_of_freedom_PT)))

print('best fit slope = {} units?'.format(popt[0]))
print('best fit intercept = {} units?'.format(popt[1]))
'''
#plt.plot(ITCTEMP,YBCOresistance)
#plt.plot(CelsiusTemp[min_index:max_index]+273.15, PTresistance[min_index:max_index], 'o', label='Data', markersize=4)
#plt.plot( CelsiusTemp[min_index:max_index]+273.15,model_function_PT(CelsiusTemp[min_index:max_index], *popt), label='Best fit', color='orange')
#plt.show()
'''

plt.plot(ITCTEMP, YBCOresistance)
#plt.plot(adjustedTemp, adjustedResistance)

#plt.plot(removedTemp, removedResistance, 'r')  # Plot removed data in red

plt.xlabel('Temperature (K)')
plt.ylabel('Resistance (Ohms)')
plt.title('YCBO Resistance vs Temperature')
#Limit so not zoomed out from noise crap when cooling down - can try removing this 
plt.ylim(0,0.001)


plt.show()
'''







Calibrated_Temp = (Resis_Temp_converter(PTresistance_DatCol, popt))

#plt.plot(Calibrated_Temp, YBCOresistance_DatCol)

#plt.show()




#Test at binnning data:





binned_Temp, binned_Resis, bin_edges = bin_temp_resistance(ITCTEMP_Calib[min_index_Dat:max_index_Dat], YBCOresistance_Calib[min_index_Dat:max_index_Dat])
#plt.plot(binned_Temp, binned_Resis)
#plt.plot(binned_Temp,np.gradient(binned_Resis,binned_Temp))
#plt.show()
#Binned_Temp, Binned_Resis, bin_edges = bin_temp_resistance(Calibrated_Temp, YBCOresistance)
#plt.plot(Binned_Temp, Binned_Resis)
#plt.show()





roll_avg_temp, roll_avg_resis = calculate_rolling_average(ITCTEMP_Calib[min_index_Dat:max_index_Dat], YBCOresistance_Calib[min_index_Dat:max_index_Dat], window_size=400)
#plt.plot(roll_avg_temp, roll_avg_resis)
plt.plot(roll_avg_temp, np.gradient(roll_avg_resis, roll_avg_temp))
plt.show()