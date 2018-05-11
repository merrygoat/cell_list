import numpy as np
from scipy.spatial.distance import cdist, squareform
import math
import coordinate_methods


def count_bonded_particles(particle_coordinates, bond_length):
    # A simple order parameter. Counts number of bonds where a bond is particle distance < 1.1
    bond_length_squared = bond_length ** 2

    distance_matrix = squareform(cdist(particle_coordinates, particle_coordinates, metric='sqeuclidean'))
    bond_matrix = distance_matrix <= bond_length_squared
    num_overlaps = np.count_nonzero(bond_matrix)

    return num_overlaps


def get_cell_size(particle_coordinates, correlation_length):
    # Given a set of particle coordinates and a correlation length, generate cells

    spatial_dimensions = 3
    max_coord = [0] * spatial_dimensions
    min_coord = [100] * spatial_dimensions
    box_size = []
    num_cells = []
    cell_size = []

    # Loop through all the cells to find the highest and lowest coodinates
    for frame in particle_coordinates:
        for dimension in range(spatial_dimensions):
            if max(frame[:, dimension]) > max_coord[dimension]:
                max_coord[dimension] = max(frame[:, dimension])
            if min(frame[:, dimension]) < min_coord[dimension]:
                min_coord[dimension] = min(frame[:, dimension])

    # Using the highest and lowest coordinates, generate cell numbers and sizes
    for dimension in range(spatial_dimensions):
        box_size.append(max_coord[dimension] - min_coord[dimension])
        num_cells.append(math.ceil(box_size[dimension] / correlation_length))
        cell_size.append(box_size[dimension] / num_cells[dimension])

    return num_cells, cell_size, box_size


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


def get_vector_cell_index(particle_position, cell_lengths):
    x_index = int(particle_position[0] / cell_lengths[0])
    y_index = int(particle_position[1] / cell_lengths[1])
    z_index = int(particle_position[2] / cell_lengths[2])
    return [x_index, y_index, z_index]


def setup_cell_list(particle_coordinates, cell_size, num_cells):
    heads = []
    particle_links = []
    total_cells = 1

    for dimension in num_cells:
        total_cells = total_cells * dimension

    for frame_index, frame in enumerate(particle_coordinates):
        heads.append([-1] * total_cells)
        particle_links.append([-1] * len(frame))
        for particle_index, particle in enumerate(frame):
            vector_index = get_vector_cell_index(particle, cell_size)
            scalar_index = get_scalar_cell_index(vector_index, num_cells)
            if heads[frame_index][scalar_index] == -1:
                heads[frame_index][scalar_index] = particle_index
            else:
                particle_links[frame_index][particle_index] = heads[frame_index][scalar_index]
                heads[frame_index][scalar_index] = particle_index
    return heads, particle_links


def loop_over_inner_cells(num_cells):
    # A generator to return the index of the inner cell
    for x_cell in range(num_cells[0]):
        for y_cell in range(num_cells[1]):
            for z_cell in range(num_cells[2]):
                yield [x_cell, y_cell, z_cell]


def loop_over_neighbour_cells(vector_cell_index, num_cells):
    # A generator to return the index of neighbouring cells
    neighbour_vector_index = []
    for x_neighbour in range(vector_cell_index[0] - 1, vector_cell_index[0] + 1):
        for y_neighbour in range(vector_cell_index[1] - 1, vector_cell_index[1] + 1):
            for z_neighbour in range(vector_cell_index[2] - 1, vector_cell_index[2] + 1):
                neighbour_vector_index = [x_neighbour, y_neighbour, z_neighbour]
                # Correct neighbour index for boundaries
                for dimension in range(3):
                    if neighbour_vector_index[dimension] > num_cells[dimension]:
                        neighbour_vector_index[dimension] -= num_cells[dimension]
                    if neighbour_vector_index[dimension] < 0:
                        neighbour_vector_index[dimension] += num_cells[dimension]
                yield neighbour_vector_index


def main(xyz_file_name):
    particle_coordinates = coordinate_methods.read_xyz_file(xyz_file_name, 3)
    num_cells, cell_size, box_size = get_cell_size(particle_coordinates, 1)
    particle_coordinates = coordinate_methods.wrap_coordinates(particle_coordinates, box_size)
    cell_heads, links = setup_cell_list(particle_coordinates, cell_size, num_cells)

    for frame in particle_coordinates:
        for inner_cell_vector_index in loop_over_inner_cells(num_cells):
            for neighbour_vector_index in loop_over_neighbour_cells(inner_cell_vector_index, num_cells):
                pass  # do science


if __name__ == '__main__':
    main("sample_configuration.xyz")
