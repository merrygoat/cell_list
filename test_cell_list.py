import cell_list
import math
import numpy as np


def test_xyz_read():
    # Read the sample xyz file. Check coordinates are correct.
    coordinates = cell_list.read_xyz_file("sample_configuration.xyz", 3)

    assert len(coordinates) == 1
    assert len(coordinates[0]) == 125


def test_cell_size():
    # Given some coordinates, check the correct number of cells are generated.

    # Unit cell of size 10, 7, 5
    coordinates = np.array([[0, 0, 0], [0, 0, 5], [0, 7, 0], [0, 7, 5], [10, 0, 0], [10, 0, 5], [10, 7, 0], [10, 7, 5]])

    num_cells, cell_size = cell_list.get_cell_size(coordinates, 1)
    assert num_cells[0] == 10
    assert num_cells[1] == 7
    assert num_cells[2] == 5
    assert math.isclose(cell_size[0], 1)
    assert math.isclose(cell_size[1], 1)
    assert math.isclose(cell_size[2], 1)


def test_get_scalar_index():
    # Given cell indices and cell dimensions get a scalar cell index

    assert cell_list.get_scalar_cell_index([0, 0, 0], [10, 10, 10]) == 0
    assert cell_list.get_scalar_cell_index([0, 0, 1], [10, 10, 10]) == 1
    assert cell_list.get_scalar_cell_index([0, 1, 0], [10, 10, 10]) == 10
    assert cell_list.get_scalar_cell_index([1, 0, 0], [10, 10, 10]) == 100
    assert cell_list.get_scalar_cell_index([0, 0, 9], [10, 10, 10]) == 9
    assert cell_list.get_scalar_cell_index([0, 9, 0], [10, 10, 10]) == 90
    assert cell_list.get_scalar_cell_index([9, 0, 0], [10, 10, 10]) == 900
    assert cell_list.get_scalar_cell_index([1, 1, 1], [5, 5, 5]) == 31


def test_get_cell_index():
    # Given a particle position and the size of the cells, find the cell indices of the particle

    assert cell_list.get_cell_index([0, 1.5, 2.5], [1, 1, 1]) == [0, 1, 2]