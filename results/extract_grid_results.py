from posydon.grids import utils_BNS
import kicks
import pandas
import os
import glob

project = '/projects/b1095/monicagg/nsbh'
binary_grid = '/20Msun'

# This will hold all the grid results
all_grid_results = pandas.DataFrame()

# these are the files we need to read (and will be passed to the utils.convert_output_to_table function
star1_history_file = os.path.join('LOGS1', 'history.data')
binary_history_file = 'binary_history.data'
output_file  = 'out.txt'

# These are all the runs we did for this grid
list_of_grid_runs = glob.glob(project+'/grids'+binary_grid+'/newZ*')

# Because we name the directories smartly, we can extract the initial conditions from the directory names
grid_columns = ['m1', 'm2', 'initial_period_in_days']
parameter_values = [grid.split('{0}_'.format(param))[-1].split('_')[0] for grid in list_of_grid_runs for param in grid_columns] 

# loop over results and parse the output for each MESA run
for grid in list_of_grid_runs:
    print(grid)
    # extract initial conditions for this run
    initial_conditions = [grid.split('{0}_'.format(param))[-1].split('_')[0] for param in grid_columns]
    print(initial_conditions)

    # extract the result of this run
    mesa_result = utils_BNS.convert_output_to_table(star1_history_file=os.path.join(grid, star1_history_file),
                                                binary_history_file=os.path.join(grid, binary_history_file),
                                                output_file=os.path.join(grid, output_file))
    # Save the intiial conditions
    mesa_result[grid_columns] = pandas.DataFrame([initial_conditions])

    # append to the big table
    all_grid_results = all_grid_results.append(mesa_result, sort=False)
data_preSN = project+'/results'+binary_grid+'/grid_results_preSN.csv'
all_grid_results.to_csv(data_preSN, index=False)

kicks.impart_kick(all_grid_results, model='Tauris')
kicks.impart_kick(all_grid_results, model='Pod')