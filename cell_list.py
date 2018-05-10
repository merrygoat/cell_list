import numpy as np
from scipy.spatial.distance import cdist
import math


def read_xyz_file(filename, dimensions):

    print("Reading data from XYZ file.")

    particle_positions = []
    frame_number = 0
    line_number = 0
    with open(filename, 'r') as input_file:
        for line in input_file:
            if line_number == 0:
                # Check for blank line at end of file
                if line != "":
                    frame_particles = int(line)
                    particle_positions.append(np.zeros((frame_particles, dimensions)))
            elif line_number == 1:
                comment = line
            else:
                particle_positions[frame_number][line_number-2] = line.split()[1:]
            line_number += 1
            # If we have reached the last particle in the frame, reset counter for next frame
            if line_number == (frame_particles + 2):
                line_number = 0
                frame_number += 1

    print("XYZ read complete.")

    return particle_positions


def count_bonded_particles(particle_coordinates):
    # A simple order parameter. Counts number of bonds where a bond is particle distance < 1.1
    bond_length = 1.1
    bond_length_squared = bond_length ** 2

    distance_matrix = cdist(particle_coordinates[0], particle_coordinates[0], metric='sqeuclidean')
    bond_matrix = distance_matrix < bond_length_squared
    num_overlaps = np.count_nonzero(bond_matrix)

    return num_overlaps


def get_cell_size(particle_coordinates, correlation_length):
    # Given a set of particle coordinates and a correlation length, generate cells

    spatial_dimensions = 3
    box_size = []
    num_cells = []
    cell_size = []

    for dimension in range(spatial_dimensions):
        box_size.append(max(particle_coordinates[:, dimension]) - min(particle_coordinates[:, dimension]))
        num_cells.append(math.ceil(box_size[dimension] / correlation_length))
        cell_size.append(box_size[dimension] / num_cells[dimension])

    return num_cells, cell_size


def get_scalar_cell_index(cell_indices, num_cells):
    x_index = cell_indices[0]
    y_index = cell_indices[1]
    z_index = cell_indices[2]
    n_cells_x = num_cells[0]
    n_cells_y = num_cells[1]
    n_cells_z = num_cells[2]

    return ((x_index + n_cells_x) % n_cells_x) * n_cells_y * n_cells_z + \
           ((y_index + n_cells_y) % n_cells_y) * n_cells_z + \
           ((z_index + n_cells_z) % n_cells_z)


def get_cell_index(particle_position, cell_lengths):
    x_index = int(particle_position[0] / cell_lengths[0])
    y_index = int(particle_position[1] / cell_lengths[1])
    z_index = int(particle_position[2] / cell_lengths[2])
    return [x_index, y_index, z_index]


