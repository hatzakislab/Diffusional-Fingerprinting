"""
Library for simulation of diffusion types

Henrik Dahl Pinholt
"""
import numpy as np
from tqdm import tqdm
from fbm import fgn, times
# import multiprocess as mp
import matplotlib.pyplot as plt


def SquareDist(x0, x1, y0, y1):
    """Computes the squared distance between the two points (x0,y0) and (y1,y1)

    Returns
    -------
    float
        squared distance between the two input points

    """
    return (x1 - x0) ** 2 + (y1 - y0) ** 2


def msd(x, y, frac):
    """Computes the mean squared displacement (msd) for a trajectory (x,y) up to
    frac*len(x) of the trajectory.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    frac : float in [0,1]
        Fraction of trajectory duration to compute msd up to.

    Returns
    -------
    iterable of lenght int(len(x)*frac)
        msd for the trajectory

    """
    N = int(len(x) * frac)
    msd = []
    for lag in range(1, N):
        msd.append(
            np.mean(
                [
                    SquareDist(x[j], x[j + lag], y[j], y[j + lag])
                    for j in range(len(x) - lag)
                ]
            )
        )
    return np.array(msd)


def Get_params(numparams, dt, D):
    """Generate a random set of parameters within the bounds presented in Phys. Rev. E 100, 032410 2019.
    For each generation, four parameter sets are generated. One for normal diffusion, confined diffusion,
    directed motion and fractional brownian motion.

    Parameters
    ----------
    numparams : int
        Number of parameter sets to generate for each diffusion type.
    dt : float > 0
        time-step to use for the trace generation.
    D : float > 0
        Diffusion constant to use for the trace generation.

    Returns
    -------
    numpy ndarray of shape (numparams,13)
        parameter sets for the numparams trajectories of each diffusion type.
        1. Duration of each normal difffusion trace
        2. Duration of each anomalous difffusion trace
        3. Duration of each confined difffusion trace
        4. Duration of each directed motion trace
        5. Diffusion constant for each normal diffusion trace
        6. Time increment for each normal diffusion trace
        7. r_c, confinement radius for confined diffusion traces
        8. v, average velocity of persistent motion traces
        9. alpha, alpha parameter in fractional brownian motion simulation
        10. sigmaND, localization errors for normal diffusion
        11. sigmaCD, localization errors for confined diffusion
        12. sigmaDM, localization errors for directed motion
        13. sigmaND, localization errors for normal diffusion
    """
    # bounds from table 1 Kowalek et al 2020
    Nmin, Nmax = 30, 600
    Bmin, Bmax = 1, 6
    Rmin, Rmax = 1, 17
    alphamin, alphamax = 0.3, 0.7
    Qmin, Qmax = 1, 9

    # Gen parameters
    Q = np.random.uniform(Qmin, Qmax, size=numparams)
    Q1, Q2 = Q, Q

    NsND = np.random.randint(Nmin, Nmax + 1, size=numparams)
    NsAD = np.random.randint(Nmin, Nmax + 1, size=numparams)
    NsCD = np.random.randint(Nmin, Nmax + 1, size=numparams)
    NsDM = np.random.randint(Nmin, Nmax + 1, size=numparams)
    TDM = NsDM * dt

    B = np.random.uniform(Bmin, Bmax, size=numparams)
    r_c = np.sqrt(D * NsCD * dt / B)  # solving for r_c in eq. 8 Kowalek

    R = np.random.uniform(Rmin, Rmax, size=numparams)
    v = np.sqrt(R * 4 * D / TDM)  # solving for v in eq. 7 Kowalek

    alpha = np.random.uniform(alphamin, alphamax, size=numparams)

    # Compute sigma for ND, AD, CD from eq. 12 Kowalek
    sigmaND = np.sqrt(D * dt) / Q1
    sigmaAD = np.sqrt(D * dt) / Q1
    sigmaCD = np.sqrt(D * dt) / Q1

    # Compute sigma for DM from eq. 12 Kowalek
    sigmaDM = np.sqrt(D * dt + v ** 2 * dt ** 2) / Q2

    return np.array(
        [
            NsND,
            NsAD,
            NsCD,
            NsDM,
            D * np.ones(numparams),
            dt * np.ones(numparams),
            r_c,
            v,
            alpha,
            sigmaND,
            sigmaAD,
            sigmaCD,
            sigmaDM,
        ]
    ).T


def Gen_normal_diff(D, dt, sigma1s, Ns, withlocerr=True):
    """Generate a set of normal diffusion traces

    Parameters
    ----------
    D : float
        Diffusion constant.
    dt : float
        Time step for each increment.
    sigma1s : list-like
        Localization errors for each trace.
    Ns : list-like of integers
        Duration of each trace.
    withlocerr : Boolean
        Wether to simulate the trace with localization errors or not.

    Returns
    -------
    list of length len(Ns)
        list containing the two-dimensional simulated trajectories as an array of
        shape (N,2) where N is the duration of the trajectory.

    """
    traces = []
    for n, sig in zip(Ns, sigma1s):
        xsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n)
        ysteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n)
        x, y = (
            np.concatenate([[0], np.cumsum(xsteps)]),
            np.concatenate([[0], np.cumsum(ysteps)]),
        )
        if withlocerr:
            x_noisy, y_noisy = (
                x + np.random.normal(0, sig, size=x.shape),
                y + np.random.normal(0, sig, size=y.shape),
            )
            traces.append(np.array([x_noisy, y_noisy]).T)
        else:
            traces.append(np.array([x, y]).T)
    return traces


def Gen_directed_diff(D, dt, vs, sigmaDM, Ns, beta_set=None, withlocerr=True):
    """Generate a set of directed motion  traces.

    Parameters
    ----------
    D : float
        Diffusion constant.
    dt : float
        Time step for each increment.
    vs : float
        Average drift speed.
    sigmaDM : list-like
        Localization errors for each trace.
    Ns : list-like of integers
        Duration of each trace.
    beta_set : list-like of floats
        Drift angle in the 2D plane.
    withlocerr : Boolean
        Wether to simulate the trace with localization errors or not.

    Returns
    -------
    list of length len(Ns)
        list containing the two-dimensional simulated trajectories as an array of
        shape (N,2) where N is the duration of the trajectory.

    """
    traces = []
    for v, n, sig in zip(vs, Ns, sigmaDM):
        if beta_set is None:
            beta = np.random.uniform(0, 2 * np.pi)
        else:
            beta = beta_set
        dx, dy = v * dt * np.cos(beta), v * dt * np.sin(beta)

        xsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n) + dx
        ysteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n) + dy

        x, y = (
            np.concatenate([[0], np.cumsum(xsteps)]),
            np.concatenate([[0], np.cumsum(ysteps)]),
        )
        if withlocerr:
            x_noisy, y_noisy = (
                x + np.random.normal(0, sig, size=x.shape),
                y + np.random.normal(0, sig, size=y.shape),
            )
            traces.append(np.array([x_noisy, y_noisy]).T)
        else:
            traces.append(np.array([x, y]).T)
    return traces


def _Take_subdiff_step(x0, y0, D, dt, r_c, nsubsteps=100):
    """Compute the step for a confined diffusing particle.
    The step is computed by propagating the particle for nsubsteps as a normal
    random walker with a reduced timestep and including a reflective circular boundary of radius r_c.
    The final step is then taken as the step from initial to final position.

    Parameters
    ----------
    x0 : float
        Initial x-coordinate.
    y0 : float
        Initial y-coordinate.
    D : float
        Diffusion constant.
    dt : float
        Time step.
    r_c : float
        Confinement radius beyond which no motion can occur.
    nsubsteps : int
        Number of substeps to take in computing the step.

    Returns
    -------
    tuple of length 2
        final x and y coordinates for a single step

    """
    dt_prim = dt / nsubsteps
    for i in range(nsubsteps):
        x1, y1 = (
            x0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
            y0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
        )
        if np.sqrt(x1 ** 2 + y1 ** 2) < r_c:
            x0, y0 = x1, y1
    return x1, y1

def Gen_confined_diff(D, dt, r_cs, sigmaCD, Ns, withlocerr=True, multiprocess=True):
    """Generate confined diffusion trajectories.

    Parameters
    ----------
    D : float
        Diffusion constant.
    dt : float
        Time step for each increment.
    r_cs : list-like of floats
        Confinement radii beyond which no motion can occur.
    sigmaCD : list-like of floats >0
        Localization errors for each trace.
    Ns : list-like of integers
        Duration of each trace.
    withlocerr : Boolean
        Wether to simulate the traces with localization errors or not.
    multiprocess : Boolean
        Wether to use multiprocessing to generate the traces in parallel.

    Returns
    -------
    list of length len(Ns)
        list containing the two-dimensional simulated trajectories as an array of
        shape (N,2) where N is the duration of the trajectory.

    """
    def get_trace(x):
        D, dt, r_c, sig, n = x
        xs, ys = [], []
        x0, y0 = 0, 0
        for i in range(n + 1):
            xs.append(x0)
            ys.append(y0)
            x0, y0 = _Take_subdiff_step(x0, y0, D, dt, r_c)
        x, y = np.array(xs), np.array(ys)
        if withlocerr:
            x_noisy, y_noisy = (
                x + np.random.normal(0, sig, size=x.shape),
                y + np.random.normal(0, sig, size=y.shape),
            )
        else:
            x_noisy, y_noisy = x, y
        return np.array([x_noisy, y_noisy]).T

    args = [(D, dt, r, sig, N) for r, sig, N in zip(r_cs, sigmaCD, Ns)]

    if multiprocess:
        print('closed mulitprocessing')
        # with mp.Pool(mp.cpu_count()) as p:
        #     traces = p.map(get_trace, args)
    else:
        traces = []
        for i in range(len(Ns)):
            traces.append(get_trace(args[i]))
    return traces


def Gen_anomalous_diff(D, dt, alphs, sigmaAD, Ns, withlocerr=True):
    """Generate traces of anomalous diffusion with fractional brownian motion.

    Parameters
    ----------
    D : float
        Diffusion constant.
    dt : float
        Time step for each increment.
    alphs : list-like of floats
        Alpha scaling for the trajectories.
    sigmaAD : list-like of floats
        Localization errors for each trace..
    Ns : list-like of integers
        Duration of each trace.
    withlocerr : Boolean
        Wether to simulate the traces with localization errors or not.

    Returns
    -------
    list of length len(Ns)
        list containing the two-dimensional simulated trajectories as an array of
        shape (N,2) where N is the duration of the trajectory.

    """
    Hs = alphs / 2
    traces = []
    for n, sig, H in zip(Ns, sigmaAD, Hs):
        n = int(n)
        stepx, stepy = (
            np.sqrt(2 * D * dt) * fgn(n=n, hurst=H, length=n, method="daviesharte"),
            np.sqrt(2 * D * dt) * fgn(n=n, hurst=H, length=n, method="daviesharte"),
        )
        x, y = (
            np.concatenate([[0], np.cumsum(stepx)]),
            np.concatenate([[0], np.cumsum(stepy)]),
        )
        x_noisy, y_noisy = (
            x + np.random.normal(0, sig, size=x.shape),
            y + np.random.normal(0, sig, size=y.shape),
        )
        if withlocerr:
            traces.append(np.array([x_noisy, y_noisy]).T)
        else:
            traces.append(np.array([x, y]).T)
    return traces

#
# def _Update_binary_state(state_n, p_shift):
#     """Update internal state in a binary HMM.
#
#     Parameters
#     ----------
#     state_n : in
#         Current HMM state$.
#     p_shift : float in [0,1]
#         Transition probability.
#
#     Returns
#     -------
#     int
#         output state.
#
#     """
#     randnum = np.random.uniform()
#     states = np.array([0, 1])
#     if randnum <= p_shift:
#         return states[states != state_n][0]
#     else:
#         return states[states == state_n][0]
#
#
# def Gen_binary_state_series(p_shift, N):
#     """Generate a binary state series of HMM states given symmetric transition probabilities.
#
#     Parameters
#     ----------
#     p_shift : float in [0,1]
#         Transition probability.
#     N : integer
#         Duration of trajectory.
#
#     Returns
#     -------
#     list-like
#         HMM state trajectory.
#
#     """
#     start_state = np.random.randint(2)
#     statehist = np.zeros(N)
#     statehist[0] = start_state
#     for i in range(1, N):
#         statehist[i] = _Update_binary_state(statehist[i - 1], p_shift)
#     return statehist
#
#
# def Gen_binary_state_series_uneven(p_shift1, p_shift2, N):
#     """Generate a binary series of HMM states where each state has a different transition probability.
#
#     Parameters
#     ----------
#     p_shift1 : float in [0,1]
#         Transition probability for state 0.
#     p_shift2 : float in [0,1]
#         Transition probability for state 1.
#     N : int
#         Duration of trajectory.
#
#     Returns
#     -------
#     list-like
#         HMM state trajectory.
#
#     """
#     start_state = np.random.randint(2)
#     statehist = np.zeros(N)
#     statehist[0] = start_state
#     for i in range(1, N):
#         if statehist[i - 1] == 0:
#             statehist[i] = _Update_binary_state(statehist[i - 1], p_shift1)
#         else:
#             statehist[i] = _Update_binary_state(statehist[i - 1], p_shift2)
#     return statehist
#
#
# def Get_shifting_diff(diff_gens, params, state, lens, plot=False):
#     """Generates binary state-shifting diffusion between the two trace generators given
#     in diff_gens.
#
#     Parameters
#     ----------
#     diff_gens : list of length 2.
#         Should contain two of the four possible trace-generators:
#         - Gen_normal_diff
#         - Gen_directed_diff
#         - Gen_confined_diff
#         - Gen_anomalous_diff.
#     params : list of length 2.
#         Should contain the parameters to .
#     state : type
#         Description of parameter `state`.
#     lens : type
#         Description of parameter `lens`.
#     plot : type
#         Description of parameter `plot`.
#
#     Returns
#     -------
#     type
#         Description of returned object.
#
#     """
#     diff1, diff2 = [diff_gens[i](**params[i]) for i in range(2)]
#     if len(state) == 1 and state[0] == 0:
#         trace = diff1[0][:-1]
#     elif len(state) == 1 and state[0] == 1:
#         trace = diff2[0][:-1]
#     else:
#         c0, c1 = 0, 0
#         stepsx, stepsy = [], []
#         for s in state:
#             trace_choice = [diff1, diff2][s][[c0, c1][s]]
#             step_comp = trace_choice[1:] - trace_choice[:-1]
#             stepsx += list(step_comp[:, 0])
#             stepsy += list(step_comp[:, 1])
#             # steps = np.concatenate([steps, step_comp])
#             if s == 0:
#                 c0 += 1
#             else:
#                 c1 += 1
#
#         trace = np.concatenate(
#             [
#                 np.array([[0, 0]]),
#                 np.array([np.cumsum(stepsx), np.cumsum(stepsy)]).T[:-1],
#             ]
#         )
#     if len(trace) != np.sum(lens):
#         # print(diff1, diff2, trace, cropped_diff1, cropped_diff2)
#         print(diff1, diff2, trace)  # , cropped_diff1, cropped_diff2)
#         raise ValueError(
#             f"{len(trace)},{state},{lens},{[len(i) for i in diff1],[len(i) for i in diff2]}"
#         )
#         # raise ValueError(
#         #     f"{len(trace)},{state},{c0,c1},{lens},{[len(i) for i in diff1],[len(i) for i in diff2]},{[len(i) for i in cropped_diff1],[len(i) for i in cropped_diff2]}"
#         # )
#     trace = np.array(trace)
#     if plot:
#         SLS = np.sqrt(np.sum((trace[1:] - trace[:-1]) ** 2, axis=1))
#         frames = np.arange(len(SLS))
#         n = 0
#         cols = ["dimgrey", "darkred"]
#         fig, ax = plt.subplots(1, 2, figsize=(8, 4))
#
#         for s, l in zip(state, lens):
#             ax[0].plot(
#                 trace[:, 0][n : n + l + 1], trace[:, 1][n : n + l + 1], c=cols[s]
#             )
#             ax[1].plot(
#                 frames[n : np.min([n + l + 1, len(SLS)])],
#                 SLS[n : np.min([n + l + 1, len(SLS)])],
#                 "o",
#                 c=cols[s],
#             )
#             n += l
#         plt.show()
#     return trace
