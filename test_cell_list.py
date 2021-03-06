import cell_list
import coordinate_methods
import math
import numpy as np


class TestCoordinateMethods:
    @staticmethod
    def test_xyz_read():
        # Read the sample xyz file. Check coordinates are correct.
        coordinates = coordinate_methods.read_xyz_file("sample_configurations/sample_configuration.xyz", 3)

        assert len(coordinates) == 1
        assert len(coordinates[0]) == 59


class TestCellListMethods:
    # cubic unit cell of size [10, 7, 5]
    sample_coordinates = [np.array([[0, 0, 0], [0, 0, 5], [0, 7, 0], [0, 7, 5], [10, 0, 0], [10, 0, 5], [10, 7, 0], [10, 7, 5]])]

    def test_get_simple_overlaps(self):
        assert cell_list.get_simple_overlaps(self.sample_coordinates, bond_length=5) == 4
        assert cell_list.get_simple_overlaps(self.sample_coordinates, bond_length=10) == 16

    def test_get_cell_list_overlaps(self):
        assert cell_list.get_cell_list_overlaps(self.sample_coordinates, bond_length=5) == -1
        real_coorindates = coordinate_methods.read_xyz_file("sample_configurations/sample_configuration.xyz", 3)
        simple_overlap_count = cell_list.get_simple_overlaps(real_coorindates, bond_length=1)
        cell_overlap_count = cell_list.get_cell_list_overlaps(real_coorindates, bond_length=1)[0]
        assert simple_overlap_count == cell_overlap_count

    @staticmethod
    def test_large_system():
        real_coorindates = coordinate_methods.read_xyz_file("sample_configurations/large_configuration.xyz", 3)
        simple_overlap_count = cell_list.get_simple_overlaps(real_coorindates, bond_length=1)
        cell_overlap_count = cell_list.get_cell_list_overlaps(real_coorindates, bond_length=1)[0]
        assert simple_overlap_count == cell_overlap_count

    def test_cell_size(self):
        # Given some coordinates, check the correct number of cells are generated.
        num_cells, cell_size, box_size = cell_list.get_cell_size(self.sample_coordinates, correlation_length=2, pbcs=0)
        assert num_cells[0] == 6
        assert num_cells[1] == 4
        assert num_cells[2] == 3
        assert math.isclose(cell_size[0], 2)
        assert math.isclose(cell_size[1], 7/3)
        assert math.isclose(cell_size[2], 2.5)
        assert box_size == [10, 7, 5]

    @staticmethod
    def test_get_scalar_cell_index():
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
    def test_get_vector_cell_index():
        # Given a particle position and the size of the cells, find the cell indices of the particle
        assert cell_list.get_vector_cell_index([0, 1.5, 2.5], [1, 1, 1]) == [0, 1, 2]

    @staticmethod
    def test_setup_cell_list():
        # Given some particles and some cells make the linked list and cell heads
        heads, links = cell_list.setup_cell_list([np.array([[0.5, 1.5, 0.5], [1.5, 1.5, 1.5], [0.5, 1.25, 0.5], [0.8, 0.3, 1.6]])], [1, 1, 1], [2, 2, 2])

        assert np.array_equal(heads, [[-1, 3, 2, -1, -1, -1, -1, 1]])
        assert np.array_equal(links, [[-1, -1, 0, -1]])

    @staticmethod
    def test_loop_over_inner_cells():
        # Given a number of cells, loop through them sequentially
        index_list = []
        for index in cell_list.loop_over_inner_cells([2, 2, 2]):
            index_list.append(index)
        assert index_list == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

    @staticmethod
    def test_loop_over_neighbour_cells():
        # Given a cell to start from and the total number of cells list the neighbours of the cell
        index_list = []
        for index in cell_list.loop_over_neighbour_cells([1, 1, 1], [3, 3, 3]):
            index_list.append(index)
        assert index_list == [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2],
                              [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2],
                              [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]
        index_list = []
        for index in cell_list.loop_over_neighbour_cells([2, 2, 2], [3, 3, 3]):
            index_list.append(index)
        assert index_list == [[1, 1, 1], [1, 1, 2], [1, 1, 0], [1, 2, 1], [1, 2, 2], [1, 2, 0], [1, 0, 1], [1, 0, 2], [1, 0, 0],
                              [2, 1, 1], [2, 1, 2], [2, 1, 0], [2, 2, 1], [2, 2, 2], [2, 2, 0], [2, 0, 1], [2, 0, 2], [2, 0, 0],
                              [0, 1, 1], [0, 1, 2], [0, 1, 0], [0, 2, 1], [0, 2, 2], [0, 2, 0], [0, 0, 1], [0, 0, 2], [0, 0, 0]]
