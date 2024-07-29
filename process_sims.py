import numpy as np
import pandas as pd
import subprocess
import vtk

# n_hi_train = 10 # at the moment, use all data to create map
# CONV_TOL = -6 # log density residual to accept simulation as converged
CONV_TOL = -4 # log density residual to accept simulation as converged
THICKNESS = 1.0 # thickness of wedge to compute Qdot

def process_vtm_data(vtm_filename):
    """
    Process vtm file to get heat transfer rate and mean temperature.
    """
    # Create a reader for the VTM file
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(vtm_filename)
    reader.Update()
    output = reader.GetOutput()
    
    # ---------- Get boundary heat transfer ----------
    block = output.GetBlock(0)  # block 0 is fluid
    info = output.GetMetaData(0)
    name = info.Get(vtk.vtkCompositeDataSet.NAME())
    for i in range(block.GetNumberOfBlocks()):
        subblock = block.GetBlock(i)
        subinfo = block.GetMetaData(i)
        subname = subinfo.Get(vtk.vtkCompositeDataSet.NAME())
        if subname == 'Boundary':
            boundary_data = subblock.GetBlock(0)  # block 0 is wall

    # Extract point data
    points = boundary_data.GetPoints()
    point_data = boundary_data.GetPointData()
    array_names = [point_data.GetArrayName(j) 
            for j in range(point_data.GetNumberOfArrays())]
    array_dict = dict(zip(array_names,
        list(range(point_data.GetNumberOfArrays()))))

    # Work through cells, extract point data
    Q_dot_sum = 0
    heat_flux_data = point_data.GetArray(array_dict['Heat_Flux'])
    for i in range(boundary_data.GetNumberOfCells()):
        cell = boundary_data.GetCell(i)
        pointIds = cell.GetPointIds()  # Get the point indices that the line connects
        # print(f"Line connects the following point indices: {pointIds.GetId(0)}, {pointIds.GetId(1)}")
        point0 = boundary_data.GetPoint(pointIds.GetId(0))
        point1 = boundary_data.GetPoint(pointIds.GetId(1))
        qdot0 = heat_flux_data.GetValue(pointIds.GetId(0))
        qdot1 = heat_flux_data.GetValue(pointIds.GetId(1))
        ds = np.sqrt((point0[0] - point1[0])**2 + (point0[1] - point1[1])**2
                + (point0[2] - point1[2])**2)
        qdot = 0.5 * (qdot0 + qdot1)
        # I'm a bit worried about these boundary points -- ds seems OK?
        # if pointIds.GetId(0) == 0:
            # Check some stuff?
        Q_dot_sum += qdot * ds * THICKNESS

    # ---------- Get mean internal temp ----------
    block = output.GetBlock(1)  # block 1 is solid
    info = output.GetMetaData(1)
    name = info.Get(vtk.vtkCompositeDataSet.NAME())
    for i in range(block.GetNumberOfBlocks()):
        subblock = block.GetBlock(i)
        subinfo = block.GetMetaData(i)
        subname = subinfo.Get(vtk.vtkCompositeDataSet.NAME())
        if subname == 'Internal':
            solid_data = subblock.GetBlock(0)  # block 0 is solid

    # Extract point data
    points = solid_data.GetPoints()
    point_data = solid_data.GetPointData()
    array_names = [point_data.GetArrayName(j)
            for j in range(point_data.GetNumberOfArrays())]
    array_dict = dict(zip(array_names,
        list(range(point_data.GetNumberOfArrays()))))

    # Work through cells, extract point data
    TA_sum = 0  # sum(T_i * A_i) [temp, area]
    A_sum = 0  # sum(A_i)
    temp_data = point_data.GetArray(array_dict['Temperature'])
    for i in range(solid_data.GetNumberOfCells()):
        cell = solid_data.GetCell(i)
        A_sum += cell.ComputeArea()
        pointIds = cell.GetPointIds()  # Get the point indices that the line connects
        point0 = solid_data.GetPoint(pointIds.GetId(0))
        point1 = solid_data.GetPoint(pointIds.GetId(1))
        point2 = solid_data.GetPoint(pointIds.GetId(2))
        T0 = temp_data.GetValue(pointIds.GetId(0))
        T1 = temp_data.GetValue(pointIds.GetId(1))
        T2 = temp_data.GetValue(pointIds.GetId(2))
        TA_sum += cell.ComputeArea() * (T0 + T1 + T2) / 3
    T_mean = TA_sum / A_sum
    return (Q_dot_sum, T_mean)

# ------------------------------------------------------------------------------
# Extract the data from the sims

data = []
sims = subprocess.run(['ls'], cwd='./simulations', stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True).stdout.split("\n")
failed_sims = []
for sim in sims:
    try:
        with open('./simulations/' + sim + '/job_summary.txt', 'r') as file:
            contents = file.read().split(':')[1].split('\n')[1:-1]
            inp_vars = [item.split('=')[0] for item in contents]
            inp_vals = np.array([item.split('=')[1] for item in contents]).astype(float)
        with open('./simulations/' + sim + '/run_cht.csv', 'r') as file:
            header = file.readline().replace(" ", "").replace('"', '').replace('\n', '').split(',')
            last_line = None
            for line in file: last_line = line
            conv_results = last_line.replace(" ", "").replace("\n", "").split(',')
            conv_results = np.array(conv_results).astype(float)
            res_dict = dict(zip(['SIM_ID'] + inp_vars + header,
                [sim] + np.hstack((inp_vals, conv_results)).tolist()))
        if res_dict['bgs[Rho][0]'] > CONV_TOL: # didn't converge yet
            failed_sims.append(sim)
        else:
            Q_dot, T_avg = process_vtm_data('./simulations/' + sim + '/flow.vtm')
            res_dict['Q_dot'] = Q_dot
            res_dict['T_avg'] = T_avg
            data.append(res_dict)
    except:
        failed_sims.append(sim)

df = pd.DataFrame(data)
x_labels = {
    'Fidelity' : 'model_index',
    'Mach' : 'Mach',
    'AoA' : 'AoA',
    'T_wall' : 'T_wall',
}
y_labels = {
    'Q_dot' : 'Q_dot',
    'T_avg' : 'T_avg',
    }
data_order = list(x_labels.values()) + list(y_labels.values())
df = df[list(x_labels.keys()) + list(y_labels.keys())]
df.Fidelity = abs(df.Fidelity - df.Fidelity.max() - 1)
df = df.rename(columns=x_labels)
df = df.rename(columns=y_labels)
df[data_order]

# Randomly split out some validation data, then save
imax = df.model_index.max()
inds = df.index[df.model_index == imax]
n_validate = 0
val_inds = np.random.choice(inds, n_validate, replace=False)
df_val = df.loc[val_inds]
df_train = df.drop(val_inds)
x_labels_csv = list(x_labels.values())
x_labels_csv.remove('model_index')
df_train.to_csv('train_data_wedge_cht.csv', index=False, header=False)
header = ['# x_labels: ' + ', '.join(x_labels_csv) + '\n',
        '# y_labels: ' + ', '.join(y_labels.values()) + '\n',
        '# params: \n',
        '# header: ' + ', '.join(data_order) + '\n']
with open('train_data_wedge_cht.csv', 'r') as file:
    existing_lines = file.readlines()
all_lines = header + existing_lines
with open('train_data_wedge_cht.csv', 'w') as file:
    file.writelines(all_lines)

# # Split out some validation data later if need
# df_val.to_csv('validate_data_wedge_cht.csv', index=False, header=False)
# header = ['# x_labels: ' + ', '.join(x_labels_csv) + '\n',
        # '# y_labels: ' + ', '.join(y_labels.values()) + '\n',
        # '# header: ' + ', '.join(data_order) + '\n']
# with open('validate_data_wedge_cht.csv', 'r') as file:
    # existing_lines = file.readlines()
# all_lines = header + existing_lines
# with open('validate_data_wedge_cht.csv', 'w') as file:
    # file.writelines(all_lines)

