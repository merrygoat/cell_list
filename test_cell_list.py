import cell_list
import coordinate_methods
import math
import numpy as np

class TestCoordinateMethods:
    @staticmethod
    def test_xyz_read():
        # Read the sample xyz file. Check coordinates are correct.
        coordinates = coordinate_methods.read_xyz_file("sample_configuration.xyz", 3)

        assert len(coordinates) == 1
        assert len(coordinates[0]) == 125

    @staticmethod
    def test_coordinate_wrapping():
        # Check sample coordaintes are coorectly wrapped.
        wrapped_coordinates = coordinate_methods.wrap_coordinates([np.array([[0, 0, 0], [11, -1, 0], [11, 11, 12], [-1, -1, -1]])], [10, 10, 10])

        assert np.array_equal(np.array([[0, 0, 0], [1, 9, 0], [1, 1, 2], [9, 9, 9]]), wrapped_coordinates[0]) == True

class TestCellListMethods:
    @staticmethod
    def test_count_bonded_particles():
        particle_list = np.array([[0, 0, 0], [0, 0, 5], [0, 7, 0], [0, 7, 5], [10, 0, 0], [10, 0, 5], [10, 7, 0], [10, 7, 5]])

        assert cell_list.count_bonded_particles(particle_list, bond_length=5) == 4
        assert cell_list.count_bonded_particles(particle_list, bond_length=10) == 16

    @staticmethod
    def test_cell_size():
        # Given some coordinates, check the correct number of cells are generated.

        # Unit cell of size 10, 7, 5
        coordinates = np.array([[0, 0, 0], [0, 0, 5], [0, 7, 0], [0, 7, 5], [10, 0, 0], [10, 0, 5], [10, 7, 0], [10, 7, 5]])

        num_cells, cell_size, box_size = cell_list.get_cell_size(coordinates, 1)
        assert num_cells[0] == 10
        assert num_cells[1] == 7
        assert num_cells[2] == 5
        assert math.isclose(cell_size[0], 1)
        assert math.isclose(cell_size[1], 1)
        assert math.isclose(cell_size[2], 1)
        assert box_size == [10, 7, 5]

    @staticmethod
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

    @staticmethod
    def test_get_cell_index():
        # Given a particle position and the size of the cells, find the cell indices of the particle

        assert cell_list.get_cell_index([0, 1.5, 2.5], [1, 1, 1]) == [0, 1, 2]
