import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib
import scipy









PTPDArr = []
PTCurrArr = []
ITCTEMPArr = []
YCBOPDArr = []
YCBOCurrArr = []


TimeArr,PTPD, PTCurr, ITCTEMP, YBCOPD, YBCOCurr = np.loadtxt('Test19.txt', unpack=True)  # Load and unpack data
TimeArr2,PTPD2, PTCurr2, ITCTEMP2, YBCOPD2, YBCOCurr2 = np.loadtxt('Test27.txt', unpack=True) 



YBCOresistance =np.array( abs(YBCOPD) )/ np.array(YBCOCurr)
PTresistance = np.array( abs(PTPD) )/ np.array(PTCurr)

YBCOresistance2 =np.array( abs(YBCOPD2) )/ np.array(YBCOCurr2)
PTresistance2 = np.array( abs(PTPD2) )/ np.array(PTCurr2)

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
CelsiusTemp = np.array(ITCTEMP)-273.15  # Convert to Celsius
CelsiusTemp2 = np.array(ITCTEMP2)-273.15  # Convert to Celsius
def model_function_PT(x, *params):
    return params[0] *(1+  params[1]*x+ params[2]*x**2 + (x-100)*params[3]*x**3)
def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params))/y_err)**2) 

max_index = np.abs(ITCTEMP-273).argmin()
min_index = np.abs(ITCTEMP-77).argmin()
no_vals = len(PTresistance)
PT_model_resis_err = np.full(no_vals, 0.002)  # Constant error of 0.0002 Ohms

initial_values_PT = np.array([ 100,  3.8e-3,  -5.4e-7, -8.47603047e-12])
degrees_of_freedom_PT = PTresistance[min_index:max_index].size - initial_values_PT.size

popt, cov = scipy.optimize.curve_fit(model_function_PT, # function to fit
                                 CelsiusTemp[min_index:max_index], # x data
                                     PTresistance[min_index:max_index], # y data
                                     #sigma=PT_model_resis_err[min_index:max_index], # array of error bars for the fit
                                     #absolute_sigma=True, # errors bars DO represent 1 std error
                                     p0=initial_values_PT, # starting point for fit
                                     check_finite=True) # raise ValueError if NaN encountered (don't allow errors to pass)

print('Optimised parameters = ', popt, '\n')
print('Covariance matrix = \n', cov)

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
'''
def Resis_Temp_converter(PTresistance, model_params):
    coefficients = [model_params[3]*model_params[0], -100*model_params[3]*model_params[0],model_params[2]*model_params[0], model_params[1]*model_params[0], model_params[0]-PTresistance]
    roots = np.roots(coefficients)
    real_roots = roots[np.isreal(roots)].real
    # TEMP IN CELSIUS
    actual_temp = real_roots[(real_roots <= 0)]
    return actual_temp + 273.15  # Convert to Kelvin


print(Resis_Temp_converter(0.0005, popt))

'''
from joblib import Parallel, delayed
import numpy as np

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


Calibrated_Temp = (Resis_Temp_converter(PTresistance2, popt))

plt.plot(Calibrated_Temp, YBCOresistance2)

plt.show()

'''

#Test at binnning data:
import numpy as np
import math
import pandas as pd

def bin_temp_resistance(temperatures, resistances):
    


    # Ensure NumPy arrays
    temperatures = np.array(temperatures)
    resistances = np.array(resistances)

    # --- Sturges' rule ---
    N = len(temperatures)
    Nobins = round(1 + math.log2(N))

    # --- Create bins ---
    bin_edges = np.linspace(np.min(temperatures), np.max(temperatures), Nobins + 1)

    # Assign each temperature to a bin
    bin_indices = np.digitize(temperatures, bin_edges) - 1  # shift to 0-based index

    # --- Compute mean resistance and temperature per bin ---
    binned_temps = []
    binned_resistances = []

    for i in range(Nobins):
        mask = bin_indices == i
        if np.any(mask):
            binned_temps.append(np.mean(temperatures[mask]))
            binned_resistances.append(np.mean(resistances[mask]))

    return np.array(binned_temps), np.array(binned_resistances), bin_edges


Binned_Temp, Binned_Resis, bin_edges = bin_temp_resistance(Calibrated_Temp, YBCOresistance[min_index:max_index])
plt.plot(Binned_Temp, Binned_Resis)
plt.show()
'''