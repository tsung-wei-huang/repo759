import numpy as np
import matplotlib.pyplot as plt
"""
N-body Simulation

Modified by Jie Tong (jtong36@wisc.edu)

Original work:
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""


def getAcc(pos, mass, G, softening):
    """
    Calculate the acceleration on each particle due to Newton's Law
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """

    N = pos.shape[0]
    a = np.zeros((N, 3))

    for i in range(N):
        for j in range(N):
            if i != j:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2) ** (-1.5)
                a[i, 0] += G * (dx * inv_r3) * mass[j, 0]
                a[i, 1] += G * (dy * inv_r3) * mass[j, 0]
                a[i, 2] += G * (dz * inv_r3) * mass[j, 0]

    return a


def main():
    """N-body simulation"""

    # Simulation parameters
    N = 100  # Number of particles
    t = 0  # current time of the simulation
    tEnd = 10.0  # time at which simulation ends
    dt = 0.01  # timestep
    softening = 0.1  # softening length
    G = 1.0  # Newton's Gravitational Constant
    plotRealTime = True  # switch on for plotting as the simulation goes along
    board_size = 4

    # Generate Initial Conditions
    np.random.seed(17)  # set the random number generator seed

    mass = np.random.rand(N, 1)  # random mass
    pos = np.random.randn(N, 3)  # random positions
    vel = np.random.randn(N, 3)  # random velocities
    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # save particle orbits for plotting trails
    pos_save = np.zeros((N, 3, Nt + 1))
    pos_save[:, :, 0] = pos

    # prep figure
    fig = plt.figure(figsize=(8, 10), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:3, 0])

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # ensure particles stay within the board limits
        pos[pos > board_size] = board_size
        pos[pos < -board_size] = -board_size

        # update accelerations
        acc = getAcc(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt / 2.0

        # update time
        t += dt

        # save positions for plotting trail
        pos_save[:, :, i + 1] = pos

        # plot in real time
        if plotRealTime or (i == Nt - 1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_save[:, 0, max(i - 50, 0) : i + 1]
            yy = pos_save[:, 1, max(i - 50, 0) : i + 1]
            plt.scatter(xx, yy, s=1, color=[0.7, 0.7, 1])
            plt.scatter(pos[:, 0], pos[:, 1], s=10, color="blue")
            ax1.set(xlim=(-board_size, board_size), ylim=(-board_size, board_size))
            ax1.set_aspect("equal", "box")
            ax1.set_xticks(range(-board_size, board_size + 1))
            ax1.set_yticks(range(-board_size, board_size + 1))

            plt.pause(0.001)
    # print(pos)
    # Save figure
    plt.savefig("nbody-python.png", dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
