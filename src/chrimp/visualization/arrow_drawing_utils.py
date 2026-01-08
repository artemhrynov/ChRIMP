import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def quadratic_bezier(P0, P1, P2, t):
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2


def cubic_bezier(P0, P1, P2, P3, t):
    return (
        (1 - t) ** 3 * P0
        + 3 * (1 - t) ** 2 * t * P1
        + 3 * (1 - t) * t**2 * P2
        + t**3 * P3
    )


def aa_control_points(p1, p2, reverse=False):
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    l = np.linalg.norm(p2 - p1)
    h = (-1 if reverse else 1) * l / 4

    # Imaginary point in the middle of the line at a height h
    im_point_rho = l / 2
    im_point_phi = h

    im_point_x = (
        p1[0] + np.cos(angle) * im_point_rho - np.sin(angle) * im_point_phi
    )
    im_point_y = (
        p1[1] + np.sin(angle) * im_point_rho + np.cos(angle) * im_point_phi
    )

    im_point = np.array([im_point_x, im_point_y])

    control_point = 2 * im_point - 0.5 * (p1 + p2)
    return control_point, im_point


def atom_atom_attack(
    p1,
    p2,
    color="k",
    show_ps=False,
    show_control_points=False,
    ax=None,
    reverse=False,
):
    epsilon = 0.05
    s = np.linspace(0 + epsilon, 1 - epsilon, 100)

    control_point, im_point = aa_control_points(p1, p2, reverse=reverse)

    x = quadratic_bezier(p1[0], control_point[0], p2[0], s)
    y = quadratic_bezier(p1[1], control_point[1], p2[1], s)

    if ax is None:
        ax = plt.gca()

    if show_ps:
        ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c="r")
    if show_control_points:
        ax.scatter(control_point[0], control_point[1], c="b")
        ax.scatter(im_point[0], im_point[1], c="g")
    ax.plot(x[:-1], y[:-1], c=color)
    ax.annotate(
        "",
        xy=(x[-1], y[-1]),
        xytext=(x[-2], y[-2]),
        arrowprops=dict(arrowstyle="->", color=color, lw=2),
        zorder=4,
    )


def baa_control_points(p1, p2, p3):
    ba_vec = p1 - p2
    bc_vec = p3 - p2

    angle_1 = np.arctan2(bc_vec[1], bc_vec[0])
    angle_2 = np.arctan2(ba_vec[1], ba_vec[0])
    l = np.linalg.norm(ba_vec)

    control_point_1 = np.array(
        [p2[0] - np.cos(angle_1) * 2 * l, p2[1] - np.sin(angle_1) * 2 * l]
    )
    control_point_2 = np.array(
        [p2[0] - np.cos(angle_2) * 2 * l, p2[1] - np.sin(angle_2) * 2 * l]
    )

    return control_point_1, control_point_2


def bond_atom_atom_attack(
    p1, p2, p3, color="k", show_ps=False, show_control_points=False, ax=None
):

    # If p2 == p3, p3 is part of the bond
    if np.allclose(p2, p3):
        p_mid = 0.5 * (p1 + p2)
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        d = np.linalg.norm(p2 - p1)
        h = max([0.7, d / 5])

        dx = -np.sin(angle) * h
        dy = np.cos(angle) * h

        new_p3 = np.array([p3[0] + dx, p3[1] + dy])

        atom_atom_attack(
            p_mid,
            new_p3,
            color=color,
            show_ps=show_ps,
            show_control_points=show_control_points,
            ax=ax,
            reverse=False,
        )
        return

    # The bond between p1 and p2 attacks p3 through p2
    epsilon = 0.02
    s = np.linspace(0 + epsilon, 1 - (epsilon / 2), 100)

    p_mid = 0.5 * (p1 + p2)

    control_point_1, control_point_2 = baa_control_points(p_mid, p2, p3)

    x = cubic_bezier(
        p_mid[0], control_point_1[0], control_point_2[0], p3[0], s
    )
    y = cubic_bezier(
        p_mid[1], control_point_1[1], control_point_2[1], p3[1], s
    )

    if ax is None:
        ax = plt.gca()

    if show_ps:
        ax.scatter([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], c="r")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c="r")
    if show_control_points:
        ax.scatter(
            [control_point_1[0], control_point_2[0]],
            [control_point_1[1], control_point_2[1]],
            c="b",
        )
    ax.plot(x[:-1], y[:-1], c=color, zorder=4)
    ax.annotate(
        "",
        xy=(x[-1], y[-1]),
        xytext=(x[-2], y[-2]),
        arrowprops=dict(arrowstyle="->", color=color, lw=2),
        zorder=4,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation

    np.random.seed(22)

    # Set up the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Initial positions
    p0 = np.array([0, 0])  # Attacking atom
    p1 = np.array([0, -1])  # Bond start
    p2 = np.array([0, 1])  # Bond end

    p3 = np.array([2.5, 0])  # Target atom, will move

    # Function to update the frames
    def update(frame):
        # Clear the axes
        for ax in axes:
            ax.clear()
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)

        # Move the target point slightly
        delta = np.random.normal(scale=1, size=2)
        p3_moved = p3 + delta

        new_coord = np.random.uniform(-5, 5, 2)
        p3_moved = new_coord

        # Left plot: atom attacks atom
        ax1 = axes[0]
        atom_atom_attack(
            p0, p3_moved, show_ps=True, show_control_points=False, ax=ax1
        )
        ax1.set_title("Atom attacks Atom")

        # Right plot: bond attacks atom
        ax2 = axes[1]
        bond_atom_atom_attack(
            p1, p2, p3_moved, show_ps=True, show_control_points=False, ax=ax2
        )
        ax2.set_title("Bond attacks Atom")

    # Create the animation
    ani = FuncAnimation(fig, update, frames=100, interval=2000)

    plt.show()

