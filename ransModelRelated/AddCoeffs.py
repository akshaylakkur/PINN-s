import pandas as pd
#importing the parametrization code to access function - this import will be different for everybody.
#This import is based on where the parametrization code is located in your machine
from ScratchWorkPlace.parametrization.parameter import *

'rewrite the original training data file with - file will be different for everybody (but not too different)'
inputFile = '/Users/akshaylakkur/GitHub/aso_pinn/Training_Data/Naca2412/Naca2412_iteration500.csv'

'file with 101 coordinates - file will be different for everybody (but not too different)'
coordsFile = '/Users/akshaylakkur/GitHub/aso_pinn/Training_Data/Naca2412/Naca2412Coords.csv'

coeffsU1, coeffsL1, coeffsU2, coeffsL2  = SplineFit(False)
print(f'Spline 1 Upper coeffs: {coeffsU1}')
print(f'Spline 1 Lower coeffs: {coeffsU1}')
print(f'Spline 2 Upper coeffs: {coeffsU1}')
print(f'Spline 2 Lower coeffs: {coeffsU1}')

coeffsDF = pd.DataFrame(list(zip(coeffsU1, coeffsL1, coeffsU2, coeffsL2)), columns=['Upper coeffs 1', 'Lower coeffs 1', 'Upper coeffs 2', 'Lower coeffs 2'])

df = pd.read_csv(inputFile)
df['coeffsU1'] = [coeffsU1] * len(df)
df['coeffsL1'] = [coeffsL1] * len(df)
df['coeffsU2'] = [coeffsU2] * len(df)
df['coeffsL2'] = [coeffsL2] * len(df)
print(df.columns)
df.to_csv(inputFile, index=False)

