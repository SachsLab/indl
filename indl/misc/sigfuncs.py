__all__ = ['sigmoid', 'minimum_jerk']


import numpy as np


def sigmoid(x, A=0, K=1, C=1, Q=1, B=1, v=1, x_offset=0):
    """
    Returns f(x) = A + (K - A) / ((C + Q * np.exp(-B*(x - x_offset)))**(1/v))
    This is a generalized logistic function.
    The default arguments reduce this to a simple sigmoid:
        f(x) = 1 / (1 + np.exp(-x))
    """
    y = A + (K - A) / ((C + Q * np.exp(-B*(x - x_offset)))**(1/v))
    return y


def minimum_jerk(x, a0=0, af=1, degree=1, duration=None):
    """
    A 1-D trajectory that minimizes jerk (3rd time-derivative of position).
    A minimum jerk trajectory is considered "smooth" and to be a feature of natural limb movements.
    https://storage.googleapis.com/wzukusers/user-31382847/documents/5a7253343814f4Iv6Hnt/minimumjerk.pdf

    A minimum-jerk trajectory is defined by:
        trajectory = a0 + (af - a0) * (10*(dx.^3) - 15*(dx.^4) + 6*(dx.^5));
    where a0 is the resting position, and af is the finishing position.

    duration is assumed to be the extent of x but it may be overidden.
    """
    x = np.array(x)[:, None]
    assert(x.size == x.shape[0]), f"x must be 1D trajectory, found {x.shape}"

    a0 = np.atleast_2d(a0)
    af = np.atleast_2d(af)

    if duration is None:

        sorted_x = np.sort(x, axis=0)
        dx = np.diff(x, axis=0)
        duration = sorted_x[-1] - sorted_x[0] + np.min(dx)

    dx = x / duration

    if degree not in [1, 2]:
        # Default - no derivative
        k0 = a0
        x1 = 10 * dx**3
        x2 = -15 * dx**4
        x3 = 6 * dx**5
    elif degree == 1:
        # Velocity
        k0 = np.zeros_like(a0)
        x1 = 30 * dx**2
        x2 = -60 * dx**3
        x3 = 30 * dx**4
    elif degree == 2:
        # Acceleration
        k0 = np.zeros_like(a0)
        x1 = 60 * dx
        x2 = -180 * dx**2
        x3 = 120 * dx**3

    return k0 + (af - a0) * (x1 + x2 + x3)
