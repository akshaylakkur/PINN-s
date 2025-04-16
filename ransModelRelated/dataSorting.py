import os
import numpy as np

airfoil = f'Naca{input('Airfoil number: ')}'

target_directory = f'/Users/akshaylakkur/Training_Data/{airfoil}'
os.makedirs(target_directory, exist_ok=True)  # Ensure the directory exists
# Define the input and output file paths
for i in range(50, 501):

    if i%50 == 0 :
        input_file_U = f'/Users/akshaylakkur/AirfoilsData/{airfoil}/{i}/U'
        input_file_R = f'/Users/akshaylakkur/AirfoilsData/{airfoil}/{i}/R'
        input_file_p = f'/Users/akshaylakkur/AirfoilsData/{airfoil}/{i}/p'
        input_file_nut = f'/Users/akshaylakkur/AirfoilsData/{airfoil}/{i}/nut'
        input_file_C = f'/Users/akshaylakkur/AirfoilsData/{airfoil}/{i}/C'
        output_file = f'{airfoil}_iteration{i}.csv'

        # Open the input files for reading
        with open(input_file_U, 'r') as file_u, open(input_file_R, 'r') as file_r, open(input_file_p, 'r') as file_p, open(input_file_nut, 'r') as file_nut, open(input_file_C, 'r') as file_C:
            lines_U = file_u.readlines()
            lines_R = file_r.readlines()
            lines_p = file_p.readlines()
            lines_nut = file_nut.readlines()
            lines_C = file_C.readlines()


        # Open the output file for writing
        with open(output_file, 'w') as file:
            # file.write(f'x, y, p, Ux, Uy, viscosity(nut), Rx, Ry, Rxy\n')

            # Iterate through each set of lines from U, R, p, nut, and C
            for u_values, r_values, p_value, nut_value, c_values in zip(lines_U, lines_R, lines_p, lines_nut, lines_C):
                # Check if U and R lines contain tuples and p_value is a single value
                if u_values.startswith('(') and u_values.endswith(')\n') and r_values.startswith(
                        '(') and r_values.endswith(')\n') and c_values.startswith('(') and c_values.endswith(')\n'):
                    # Split the U, R, and C components
                    u_components = u_values.strip('()\n').split()
                    r_components = r_values.strip('()\n').split()
                    c_components = c_values.strip('()\n').split()
                    p_component = p_value.strip()
                    nut_component = nut_value.strip()
                    modified_line = f"{c_components[0]}, {c_components[1]}, {p_component}, {u_components[0]}, {u_components[1]}, {nut_component}, {r_components[0]}, {r_components[1]}, {r_components[3]}\n"

                    file.write(modified_line)

        # Read the output file, remove lines with '(', and rewrite it
        with open(output_file, 'r+') as file:
            lines = file.readlines()
            file.seek(0)  # Move the cursor to the beginning of the file
            file.truncate()  # Clear the file content
            file.write(f'x, y, p, Ux, Uy, viscosity(nut), Rx, Ry, Rxy\n')

            # Write back only the lines that do not contain '('
            for line in lines:
                if '(' not in line:
                    file.write(line)

        target_path = os.path.join(target_directory, output_file)
        os.rename(output_file, target_path)
        print(f"File successfully saved to {target_path}")
        os.makedirs(target_directory, exist_ok=True)