"""
Simulate hypersonic wedge SU2 with conjugate heat transfer for a variety of
boundary conditions.
Inputs are angle-of-attack, Mach number, inner (TPS) wall temp.
"""

import concurrent.futures
from datetime import datetime
import os
import subprocess

import numpy as np

N_CPUS = 4
X_NAMES = ['I_F', 'MACH', 'AOA', 'T_WALL']
N_DIM = len(X_NAMES)
N_PTS = [100, 20] # N samples, lowest to highest fidelity
N_F = len(N_PTS) # number of fidelities
# Also sample all corner points using highest fidelity model
AOA_RANGE = [-15, 15]
MA_RANGE = [2, 8]
T_WALL_RANGE = [300, 700]
SOS = 343 # m/s (speed of sound)

# TODO: eval corner points for highest fidelity?

# Create an empty DataFrame with the specified column names
fidelity_arr = np.hstack([np.full(N_PTS[N_F - i], i) for i in np.arange(N_F, 0, -1)])
data_rows = []
for i in range(sum(N_PTS)):
    data_rows.append({
        'IND' : i,
        'I_F' : fidelity_arr[i],
        'MACH' : np.random.uniform(*MA_RANGE),
        'AOA' : np.random.uniform(*AOA_RANGE),
        'T_WALL' : np.random.uniform(*T_WALL_RANGE)
        })

def eval_job(row):
    # Maybe some info about job in dir name?
    dir_name = './simulations/'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")+str(row['IND'])
    os.makedirs(dir_name)
    fidelity_lvl = int(row['I_F'])
    job_summary = 'Hypersonic wedge CHT job:\n'
    job_summary += 'Fidelity= {}\n'.format(str(fidelity_lvl))
    job_summary += 'Mach= {}\n'.format(str(row['MACH'])[:6])
    job_summary += 'AoA= {}\n'.format(str(row['AOA'])[:6])
    job_summary += 'T_wall= {}\n'.format(str(row['T_WALL'])[:6])
    with open(dir_name + '/job_summary.txt', 'w') as file:
        file.write(job_summary)

    # Set up and run simulation
    with open('cht_wedge_base.cfg', 'r') as file:
        contents = file.read()
        contents = contents.replace('INP_MESH', 'wedge_mesh_{}.su2'.format(fidelity_lvl))
    with open(dir_name + '/run_cht.cfg', 'w') as file:
        file.write(contents)
    with open('flow_wedge_base.cfg', 'r') as file:
        contents = file.read()
        contents = contents.replace('INP_MACH', str(row['MACH'])[:6])
        contents = contents.replace('INP_AOA', str(row['AOA'])[:6])
        vx = SOS * row['MACH'] * np.cos(np.deg2rad(row['AOA']))
        vy = SOS * row['MACH'] * np.sin(np.deg2rad(row['AOA']))
        contents = contents.replace('INP_VX', str(vx)[:8])
        contents = contents.replace('INP_VY', str(vy)[:8])
    with open(dir_name + '/flow_wedge.cfg', 'w') as file:
        file.write(contents)
    with open('solid_wedge_base.cfg', 'r') as file:
        contents = file.read()
        contents = contents.replace('INP_WALL_TEMP', str(row['T_WALL'])[:6])
    with open(dir_name + '/solid_wedge.cfg', 'w') as file:
        file.write(contents)
    print('Starting job ' + dir_name.split('/')[-1])
    out = subprocess.run(['SU2_CFD', 'run_cht.cfg'], cwd=dir_name, capture_output=True, text=True)
    with open(dir_name + '/output.txt', 'w') as file:
        file.write(str(out.stdout))
    print('Finished job ' + dir_name.split('/')[-1])

# with concurrent.futures.ProcessPoolExecutor(max_workers=N_CPUS) as executor:
    # results = list(executor.map(eval_job, data_rows))

for row in data_rows: eval_job(row)

print("Done.")


