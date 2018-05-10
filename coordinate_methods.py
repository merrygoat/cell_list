import numpy as np

def read_xyz_file(filename, dimensions):
    """
    A simple XYZ file reader. Assumes file is well formed. Not failsafe for ill formed files.
    :param filename: The name of the xyz file to read
    :param dimensions: The number of spatial dimensions in the xyz file
    :return: A list of length A of B by dimensions numpy arrays where A is the number of frames
     and B is the number of particles in each frame.
    """
    print("Reading data from XYZ file.")

    particle_positions = []
    frame_number = 0
    line_number = 0
    frame_particles = 0
    with open(filename, 'r') as input_file:
        for line in input_file:
            if line_number == 0:
                # Check for blank line at end of file
                if line != "":
                    frame_particles = int(line)
                    particle_positions.append(np.zeros((frame_particles, dimensions)))
            elif line_number == 1:
                pass
            else:
                particle_positions[frame_number][line_number-2] = line.split()[1:]
            line_number += 1
            # If we have reached the last particle in the frame, reset counter for next frame
            if line_number == (frame_particles + 2):
                line_number = 0
                frame_number += 1

    print("XYZ read complete.")

    return particle_positions


def wrap_coordinates(particle_coordinates, box_size):
    """Wrap coordinates into the box of size [[0, Lenx], [0, Leny], [0, Lenz]].
     This is important for the cell indexing which needs all coords > 0 and < len.

    :param particle_coordinates: an A by B by d array where A is num frames, B is num particles
     and d is the number of dimensions
    :param box_size: A d member tuple, the size of the box in each dimension.
    :return: An A by B by d array of wrapped particle coordinates
    """

    for frame_num in range(len(particle_coordinates)):
        for particle_num in range(len(particle_coordinates[frame_num])):
            for dimension in range(len(particle_coordinates[frame_num][particle_num])):
                while particle_coordinates[frame_num][particle_num][dimension] > box_size[dimension]:
                    particle_coordinates[frame_num][particle_num][dimension] -= box_size[dimension]
                while particle_coordinates[frame_num][particle_num][dimension] < 0:
                    particle_coordinates[frame_num][particle_num][dimension] += box_size[dimension]

    return particle_coordinates
