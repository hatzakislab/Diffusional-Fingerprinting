"""
Library for performing machine learning tasks on diffusional fingerprints

Henrik Dahl Pinholt
"""
import matplotlib
from matplotlib import animation
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, svm, metrics

from pomegranate import *
import pandas as pd


def histogram(
    data,
    range=(None, None),
    bins=None,
    remove0=False,
    plot=True,
    savefig="",
    labels=None,
    bars=False,
    ax=None,
    color="r",
    alpha=1,
    remove_num=None,
    calibration=None,
    legend=None,
    logbin=False,
    normalize=False,
    elinewidth=5,
    capsize=4,
):
    """Get data for histogram
    --------
    Params
    --------
    data: 1D ndarray
        - Raw Data
    xmin: float
        - Minimum value for data
    xmax: float
        - Maximum value for data
    nbins: integer
        - Number of bins in histogram
    plot: Boolean
        - Wether to plot histogram
    savefig: string
        - Wether to save the plotted histogram
    labels: dictionary
        - Values to pass to ax.set()
    legend: optional, str
        - What to call the plot in a legend
    logbin: optional, Boolean
        - wether to bin the histogram logarithmically
    """
    xmin, xmax = range
    if xmin is None:
        xmin = np.min(data)
    if xmax is None:
        xmax = np.max(data)
    if bins is None:
        bins = np.max([int(len(data) / 15), 5])
    if logbin:
        N = bins
        inputs = np.logspace(np.log10(xmin), np.log10(xmax), N)
        if normalize:
            hist = np.histogram(data, bins=inputs, normed=True)
            hist2 = np.histogram(data, bins=inputs)
        else:
            hist = np.histogram(data, bins=inputs)
    else:
        if normalize:
            hist = np.histogram(data, bins=bins, range=(xmin, xmax), normed=True)
            hist2 = np.histogram(data, bins=bins, range=(xmin, xmax))
        else:
            hist = np.histogram(data, bins=bins, range=(xmin, xmax))
    counts, bin_edges = hist
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    if calibration is None:
        mask1 = (xmin < bin_centers) & (bin_centers <= xmax)
        if remove0:
            mask2 = counts > 0
            if remove_num is not None:
                mask3 = counts > remove_num
                mask_final = mask1 & mask2 & mask3
            else:
                mask_final = mask1 & mask2
        else:
            mask_final = mask1
        x, y, sy = (
            bin_centers[mask_final],
            counts[mask_final],
            np.sqrt(counts[mask_final]),
        )
        if normalize:
            counts1, bin_edges1 = hist2
            bin_centers1 = 0.5 * (bin_edges1[1:] + bin_edges1[:-1])
            mask1 = (xmin < bin_centers1) & (bin_centers1 <= xmax)
            if remove0:
                mask2 = counts1 > 0
                if remove_num is not None:
                    mask3 = counts > remove_num
                    mask_final = mask1 & mask2 & mask3
                else:
                    mask_final = mask1 & mask2
            else:
                mask_final = mask1
            x1, y1, sy1 = (
                bin_centers1[mask_final],
                counts1[mask_final],
                np.sqrt(counts1[mask_final]),
            )
            relerr = y1 / sy1
            sy = y / relerr
    else:
        counts = counts - calibration(bin_centers)
        mask1 = (xmin < bin_centers) & (bin_centers <= xmax)
        if remove0:
            mask2 = counts > 0
            if remove_num is not None:
                mask3 = counts > remove_num
                mask_final = mask1 & mask2 & mask3
            else:
                mask_final = mask1 & mask2
        else:
            mask_final = mask1
        x, y, sy = (
            bin_centers[mask_final],
            counts[mask_final],
            np.sqrt(counts[mask_final]),
        )
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        if logbin is None:
            Binwidth = (xmax - xmin) / bins
        else:
            Binwidth = (bin_edges[1:] - bin_edges[:-1])[mask_final]
        if not bars:
            if legend is None:
                ax.errorbar(
                    x,
                    y,
                    yerr=sy,
                    xerr=Binwidth / 2,
                    linestyle="",
                    ecolor=color,
                    fmt=".",
                    mfc=color,
                    mec=color,
                    capsize=capsize,
                    elinewidth=elinewidth,
                )
            else:
                ax.errorbar(
                    x,
                    y,
                    yerr=sy,
                    xerr=Binwidth / 2,
                    linestyle="",
                    ecolor=color,
                    fmt=".",
                    mfc=color,
                    mec=color,
                    capsize=capsize,
                    label=legend,
                    elinewidth=elinewidth,
                )
        else:
            if legend is None:
                ax.bar(
                    x,
                    y,
                    width=Binwidth,
                    color=color,
                    yerr=sy,
                    capsize=capsize,
                    alpha=alpha,
                    error_kw={"elinewidth": elinewidth},
                )
            else:
                ax.bar(
                    x,
                    y,
                    width=Binwidth,
                    color=color,
                    yerr=sy,
                    capsize=capsize,
                    alpha=alpha,
                    label=legend,
                    error_kw={"elinewidth": elinewidth},
                )
        if np.abs(np.mean(Binwidth)) < 100:
            ax.set(xlabel="x", ylabel=f"Frequency / {np.abs(np.mean(Binwidth)):4.2e}")
        else:
            ax.set(xlabel="x", ylabel=f"Frequency / {np.abs(np.mean(Binwidth)):4.2e}")
        if labels is not None:
            ax.set(**labels)
        if logbin:
            ax.set_xscale("log")
        if legend is not None:
            ax.legend()
        if savefig != "":
            plt.savefig(savefig + ".pdf")
        if ax is None:
            plt.show()
    return x, y, sy


def radarplot(values, colors, labels, savefig=False, show=True):
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    # -----------------------------------------------------------------------------
    # Copyright (C) 2011  Nicolas P. Rougier
    #
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are met:
    #
    # * Redistributions of source code must retain the above copyright notice, this
    #   list of conditions and the following disclaimer.
    #
    # * Redistributions in binary form must reproduce the above copyright notice,
    #   this list of conditions and the following disclaimer in the documentation
    #   and/or other materials provided with the distribution.
    #
    # * Neither the name of the glumpy Development Team nor the names of its
    #   contributors may be used to endorse or promote products derived from this
    #   software without specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    # POSSIBILITY OF SUCH DAMAGE.
    #
    # -----------------------------------------------------------------------------
    import numpy as np
    import matplotlib
    import matplotlib.path as path
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.cm as cm

    print(len(values))
    # Data to be represented
    # ----------
    properties = ["Precision", "Recall", "F1"]
    # ----------
    # Choose some nice colors
    matplotlib.rc("axes", facecolor="white")
    # Make figure background the same colors as axes
    fig = plt.figure(figsize=(10, 10), facecolor="white")

    # Use a polar axes
    axes = plt.subplot(111, polar=True)

    for v, n, l in zip(values, colors, labels):
        v = v
        # Set ticks to the number of properties (in radians)
        t = np.arange(0, 2 * np.pi, 2 * np.pi / len(properties))
        plt.xticks(t, [])

        # Set yticks from 0 to 10
        plt.yticks(np.linspace(0, 1, 11))

        # Draw polygon representing values
        points = [(x, y) for x, y in zip(t, v)]
        points.append(points[0])
        points = np.array(points)
        codes = (
            [path.Path.MOVETO]
            + [path.Path.LINETO] * (len(v) - 1)
            + [path.Path.CLOSEPOLY]
        )
        _path = path.Path(points, codes)
        _patch = patches.PathPatch(
            _path, fill=True, color=n, linewidth=0, alpha=0.4, label=l
        )
        axes.add_patch(_patch)
        _patch = patches.PathPatch(_path, fill=False, linewidth=2)
        axes.add_patch(_patch)

        # Draw circles at value points
        plt.scatter(
            points[:, 0],
            points[:, 1],
            linewidth=2,
            s=50,
            color="white",
            edgecolor="black",
            zorder=10,
        )

    # Set axes limits
    plt.ylim(0, 1)

    # Draw ytick labels to make sure they fit properly
    for i in range(len(properties)):
        angle_rad = i / float(len(properties)) * 2 * np.pi
        angle_deg = i / float(len(properties)) * 360
        ha = "right"
        if angle_rad < np.pi / 2 or angle_rad > 3 * np.pi / 2:
            ha = "left"
        plt.text(
            angle_rad,
            1.1,
            properties[i],
            size=14,
            horizontalalignment=ha,
            verticalalignment="center",
        )

        # A variant on label orientation
        #    plt.text(angle_rad, 11, properties[i], size=14,
        #             rotation=angle_deg-90,
        #             horizontalalignment='center', verticalalignment="center")

    # Done
    plt.legend()
    if savefig != False:
        plt.savefig(savefig + ".pdf", facecolor="white", dpi=500)
    if show:
        plt.show()


def Meancalc(data, targetnum, n_comp, groups):
    """calculates mean of all target groups
    Parameters
    ------------
    data : np.array shape(N_datapoints,N_features)
        input data, rows are datapoints, and columns are features
    targetnum : np.array of ints
        datapoint labels
    n_comp :

    """
    import numpy as np

    means = np.zeros((n_comp, groups))
    nums = np.zeros(groups)
    for i in range(len(targetnum)):
        means[:, targetnum[i]] += data[i]
        nums[targetnum[i]] += 1
    for i in range(n_comp):
        means[i] /= nums
    return means


def Rfinder(X, targetnum, means, sumlist, frac, chull=False, vertices=False):
    """Finds the convex hull of a fraction of the points found from going away
    from the meancenter and counting points returns the radius of the sphere
    that contains frac of points from mean"""
    from scipy.spatial import ConvexHull

    # import and grab target group points
    t = targetnum
    triangles, vlist, simplist, rlist, dlist, Tgroups = [], [], [], [], [], []
    for i in range(len(sumlist)):
        Tgroups.append(X[t == i])
    for i in range(len(sumlist)):
        d = np.array(
            [
                np.sqrt(
                    np.sum(
                        [
                            (Tgroups[i][j, v] - means[:, i][v]) ** 2
                            for v in range(len(X[0]))
                        ]
                    )
                )
                for j in range(len(Tgroups[i]))
            ]
        )

        sorter = np.argsort(d)

        pnum = int(sumlist[i] * frac)
        rlist.append(d[sorter][pnum])
        # Generate convex hull from points up til pnum
        data = np.array(Tgroups[i][sorter][:pnum])
        if len(data) > 0:
            Chull = ConvexHull(data)
            triangles.append(Chull.simplices)
            vlist.append(Chull.vertices)
            dlist.append(data)
        else:
            return "No points"
    if vertices:
        return triangles, dlist, rlist, vlist
    else:
        return triangles, dlist, rlist


SMALL_SIZE = 15
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


class ML:
    def __init__(self, X, y, center=True):
        """Initialize ML object
        ---------
        Params
        ---------
        X: ndarray of shape (n_features,n_samples)
            -Input training data
        y: ndarray of shape(n_samples)
            -Input labels for training data
        """
        self.X = X
        self.stringy = y
        # save list of unique labels
        # Convert to number labels as well
        tlist = pd.unique(y)
        self.to_string = dict(zip(range(len(tlist)), tlist))  # Make int->string dict
        self.to_int = {v: k for k, v in self.to_string.items()}  # Make string->int dict
        self.y = np.array([self.to_int[t] for t in y])  # Convert input
        self.unique = pd.unique(self.y)
        self.stringyunique = pd.unique(self.stringy)
        if center:
            self.Center()

    def Center(self, center=True, verbose=False):
        """Center the column means of the data (important for dimension reduction)
        ---------
        Params
        ---------
        verbose: Boolean
            - Wether to print progress
        center: Boolean
            - Wether to center internal X (true) or return centered data (false)
        ---------
        Returns (if center=False)
        ---------
        X: ndarray
            - Column centered data
        T: StandardScaler object
            - Object used to do the rescaling of original data
        """
        if verbose:
            s = np.sum(np.mean(self.X, axis=1))
            print("Centering column means, current colmean sum is %.3f" % s)
        scaler = StandardScaler()
        scaler.fit(self.X)
        X = scaler.transform(self.X)
        if center:
            self.X = X
            self.scaler = scaler
        else:
            return X, scaler
        if verbose:
            s = sum([np.mean(self.X[:, i]) for i in range(len(self.X[0, :]))])
            print("sum of colmeans is now %.3f" % s)

    def Reduce(self, method, n_components=2, reduce=True, verbose=False):
        """Reduce dimensionality of the data
        ---------
        Params
        ---------
        method: string {pca","lin"} or pca or lin object with .transform() property
            - type of transformation, or the object to transform with
        n_components: integer, default=2
            - Number of components to transform to, has no effect if a method
              input is a transformation object
        reduce: Boolean
            - Wether to reduce internal X (true) or return reduced data (false)
        verbose: Boolean
            - Wether to print progress
        ---------
        Returns (if reduce=False)
        ---------
        X: ndarray
            - Reduced data
        T: pca or LinearDiscriminantAnalysis object
            - Object used to do the rescaling of original data
        """
        if method == "pca":
            if verbose:
                print(
                    "----Transforming data to a %dD hyperplane with PCA-----"
                    % n_components
                )
            L = PCA(n_components=n_components)
            X = L.fit_transform(self.X)
        elif method == "Tsne":
            from sklearn.manifold import TSNE

            L = TSNE(n_components=n_components, perplexity=50, verbose=1)
            X = L.fit_transform(self.X)
        elif method == "lin":
            if verbose:
                print(
                    "----Transforming data to a %dD hyperplane with LDA-----"
                    % n_components
                )
            L = LinearDiscriminantAnalysis(n_components=n_components)
            X = L.fit_transform(self.X, self.y)
        elif type(method.transform) == type(self.Center):
            L = method
            X = L.transform(self.X)
        else:
            raise ValueError("type has to be lin or pca or have .transform property")
        if verbose:
            print("Transformation done")
        # Decide to reduce or not and save or return transformation
        self.T = L
        if reduce:
            self.X = X
        else:
            return X, L

    def GetSumlist(self):
        """Returns a list of the number of items for each class in data"""
        sumlist = np.zeros(len(self.unique))
        # Count number times every name appears in the target
        counts = pd.Series(self.y).value_counts().values
        names = pd.Series(self.stringy).value_counts().index.values
        sorted = np.argsort(names)
        names, counts = names[sorted], counts[sorted]
        # Add counts to sumlist in same order as in self.unique
        for i in range(len(counts)):
            sumlist[self.to_int[names[i]]] = counts[i]
        return sumlist

    def ProjectPlot(
        self,
        savefig="",
        xlims=[1],
        ylims=[2],
        zlims=[2],
        yscale="linear",
        frac=0.68,
        spheres=False,
        lines=False,
        points=False,
        skip=1,
        s=0.01,
        lw=0.03,
        chull=True,
        alpha=0.6,
        With2D=False,
        axis=None,
        animate=False,
        verbose=True,
        frames=360,
        colors=None,
        legend=True,
        Get_mesh=False,
    ):
        """Plots the input data X if it is 2 or 3 dimensional
        ----------
        Params
        ----------
        savefig:  string
            - filename for saved plots
        xlims:  itterable 1x2:
            - xlims of plot
        ylims:  itterable 1x2:
            - ylims of plot
        zlims:  itterable 1x2:
            - zlims of plot (if 3D)
        yscale:  string
            - scale on y-axis ("log", "linear" (default))
        frac:  float in range[0,1]
            - fraction of points to include in convex hull
              default -> 0.68
        spheres:  boolean
            - wether to plot spheres containing frac points from the mean in 3D
              default -> False
        lines:  boolean
            - wether to plot lines from cluster mean to point
              default -> False
        points: Boolean
            - wether to plot datapoints
              default -> False
        skip: integer
            - What lines to skip (for example 5 skips every fitfth line)
              default -> 1 (none)
        s: float,
            - markersize in scatterplot
              default -> 0.01
        lw: float,
            - linewidth for lines
              default -> 0.03
        chull: boolean
            - wether to plot chulls of frac points from mean
              default -> True
        alpha: float [0,1]
            - alpha of the chulls/spheres
              default -> 0.6
        With2D: boolean
            - Wether to plot with 2D projections in 3D
              default -> False
        axis: axis object, optional
            - Axis for which to plot on
        verbose: boolean
            - wether to print stuff along the way
        animate: string
            - string to save animated rotation of the 3D plot
        frames: int
            - number of frames to animate if animate is set to a string
        """
        import numpy as np

        X, Y = self.X, self.stringy
        t = self.y
        sumlist = self.GetSumlist()
        n_components = len(X[0, :])
        m = Meancalc(X, t, n_components, len(sumlist))
        # plot if dimension = 2
        if colors is None:
            colors = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        if n_components == 2:
            if len(xlims) == 1:
                xlims = [np.min(X[:, 0]), np.max(X[:, 0])]
            if len(ylims) == 1:
                ylims = [np.min(X[:, 1]), np.max(X[:, 1])]
            """make histogram for each lipase"""
            ncols = math.ceil(len(self.stringyunique) ** (0.5))
            if len(self.stringyunique) ** (0.5) % ncols == 0.0:
                nrows = ncols
            else:
                nrows = ncols - 1
            fig1, ax1 = plt.subplots(ncols, nrows, figsize=(12, 8))
            if type(ax1) == type(np.array([])):
                for i, a in zip(self.stringyunique, ax1.flatten()):
                    hist = a.hist2d(
                        X[:, 0][Y == i],
                        X[:, 1][Y == i],
                        bins=100,
                        range=[xlims, ylims],
                        cmin=1,
                    )
                    a.set(xlabel="1st axis", ylabel="2nd axis", title=i)
                    a.grid()
                    cbar = fig1.colorbar(hist[-1], ax=a)
                    cbar.set_label(f"Frequency", rotation=270)
            else:
                hist = ax1.hist2d(
                    X[:, 0], X[:, 1], bins=100, range=[xlims, ylims], cmin=1
                )
                ax1.set(xlabel="1st axis", ylabel="2nd axis", title=Y[0])
                ax1.grid()
                cbar = fig1.colorbar(hist[-1], ax=ax1)
                cbar.set_label(f"Frequency", rotation=270)
            fig1.tight_layout()
            """make scatter plot"""
            taken = []
            if verbose:
                print("plotting scatterplot")
            # check for added axis
            if axis is None:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            else:
                ax = axis
            if chull:
                triangles, dlist, rlist, verts = Rfinder(
                    X, t, m, sumlist, frac, vertices=True
                )
                for tris, c, d in zip(verts, colors[: len(triangles)], dlist):
                    ax.fill(d[tris][:, 0], d[tris][:, 1], c, alpha=alpha)
            # Itterate through all targets in the data
            for name in self.stringyunique:
                number = self.to_int[name]
                if points:
                    ax.scatter(
                        X[Y == name][::skip][:, 0],
                        X[Y == name][::skip][:, 1],
                        s=s,
                        marker=".",
                        label=name,
                        c=colors[number],
                    )
                if lines:
                    if verbose:
                        print("plotting lines")
                    # Print % done
                    # if i % (int(len(X)/10)) == 0 and i != 0 and verbose:
                    # print ("%d %% done" % (int((i*100)/float(len(X)))))
                    ax.scatter(m[:, number][0], m[:, number][1], s=1, c=colors[number])
                    for i in range(len(X)):
                        if i % skip == 0:
                            a = [m[:, number][0], X[i, :][0]]
                            b = [m[:, number][1], X[i, :][1]]
                            ax.plot(a, b, linewidth=lw, c=colors[number])
            # Fix plot to look pretty
            if legend:
                lgnd = ax.legend(loc=2, borderaxespad=0.0)
            # set size on points in legend
            for i in range(len(lgnd.legendHandles)):
                lgnd.legendHandles[i]._sizes = [20]
            # Set limits and labels
            ax.set_xlim(xlims[0], xlims[1])
            ax.set_ylim(ylims[0], ylims[1])
            ax.set_xlabel("1st axis")
            ax.set_ylabel("2nd axis")
            ax.set_yscale(yscale)
            # Save if needed
            if savefig != "":
                plt.savefig(savefig, dpi=700, bbox_inches="tight")
                fig1.savefig(savefig + "Hist", dpi=700)
            plt.tight_layout()
            if axis is None:
                plt.show()
        # Plot as 3D surface/graph if n_components is 3
        if n_components == 3:
            import numpy as np
            from mpl_toolkits.mplot3d import Axes3D

            # Initialize/calculate important variables
            if axis == None:
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = axis

            taken = []
            if colors is None:
                colors = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
            print("plotting")
            # Itterate through all points in the reduced data
            for i in range(len(X)):
                # Print % done
                if i % (int(len(X) / 10)) == 0 and i != 0:
                    print("%d %% done" % (int((i * 100) / float(len(X)))))
                # Check if it is the first time this lipase is plotted
                if t[i] in taken:
                    # if not, no label and keep plotting
                    label = None
                else:
                    label = self.y[i]
                    taken.append(t[i])
                    ax.scatter(
                        m[:, t[i]][0],
                        m[:, t[i]][1],
                        m[:, t[i]][2],
                        s=s,
                        c=colors[t[i]],
                        label=self.to_string[label],
                    )
                    triangles, dlist, rlist = Rfinder(X, t, m, sumlist, frac)
                    if spheres:
                        # If spheres, plot it around the mean centers based
                        # on calculated radius
                        r = rlist[t[i]]
                        u = np.linspace(0, 2 * np.pi, 10)
                        v = np.linspace(0, np.pi, 10)
                        x = r * np.outer(np.cos(u), np.sin(v)) + m[:, t[i]][0]
                        y = r * np.outer(np.sin(u), np.sin(v)) + m[:, t[i]][1]
                        z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + m[:, t[i]][2]
                        # Plot the surface
                        ax.plot_surface(
                            x,
                            y,
                            z,
                            rstride=1,
                            cstride=1,
                            color=colors[t[i]],
                            alpha=alpha,
                            linewidth=0,
                        )
                    if chull:
                        for tr, c, d in zip(triangles, colors[: len(triangles)], dlist):
                            x, y, z = d[:, 0], d[:, 1], d[:, 2]
                            ax.plot_trisurf(x, y, tr, z, color=c, alpha=alpha)
                # If points and lines needed plot them
                if i % skip == 0 and points:
                    ax.scatter(
                        X[i, :][0],
                        X[i, :][1],
                        X[i, :][2],
                        s=s,
                        marker=".",
                        c=colors[t[i]],
                    )
                if i % skip == 0 and lines:
                    a = [m[:, t[i]][0], X[i, :][0]]
                    b = [m[:, t[i]][1], X[i, :][1]]
                    c = [m[:, t[i]][2], X[i, :][2]]
                    ax.plot(a, b, c, linewidth=lw, c=colors[t[i]])
            if With2D:
                twoX = X[:, :-1]
                for i in range(len(twoX)):
                    # Print % done
                    if i % (int(len(twoX) / 10)) == 0 and i != 0:
                        print("%d %% done" % (int((i * 100) / float(len(twoX)))))
                    # Check if it is the first time this lipase is plotted
                    if t[i] in taken:
                        # if not, no label and keep plotting
                        label = None
                    else:
                        # If so, add label and plot the mean center with label
                        if self.train:
                            # If so, add label and plot the mean center with label
                            label = self.converter[t[i]]
                        else:
                            label = "Input point"
                        taken.append(t[i])
                        ax.scatter(
                            m[:, t[i]][0],
                            m[:, t[i]][1],
                            zdir="z",
                            zs=-8,
                            s=1,
                            c=colors[t[i]],
                            label=label,
                        )
                    if i % skip == 0:
                        # If points and lines needed plot them
                        ax.scatter(
                            twoX[i, :][0],
                            twoX[i, :][1],
                            s=s,
                            marker=".",
                            c=colors[t[i]],
                            zdir="z",
                            zs=-8,
                        )
                        a = [m[:, t[i]][0], twoX[i, :][0]]
                        b = [m[:, t[i]][1], twoX[i, :][1]]
                        ax.plot(a, b, linewidth=lw, c=colors[t[i]], zdir="z", zs=-8)

            # Make plot look sexy
            if len(xlims) > 1:
                ax.set_xlim(xlims[0], xlims[1])
            if len(ylims) > 1:
                ax.set_ylim(ylims[0], ylims[1])
            if len(zlims) > 1:
                ax.set_zlim(zlims[0], zlims[1])
            ax.set_xlabel("1st axis")
            ax.set_ylabel("2nd axis")
            ax.set_zlabel("3rd axis")
            ax.set_yscale(yscale)
            plt.tight_layout()
            if legend:
                lgnd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                for i in range(len(lgnd.legendHandles)):
                    lgnd.legendHandles[i]._sizes = [20]
            if savefig != "":
                plt.savefig(savefig, dpi=700, bbox_inches="tight")
            if type(animate) == type("hi"):
                if legend:
                    lgnd = ax.legend(loc=2)
                    for i in range(len(lgnd.legendHandles)):
                        lgnd.legendHandles[i]._sizes = [20]

                def animater(i):
                    if i % 50 == 0:
                        print(i)
                    ax.view_init(elev=10.0, azim=360 * i / frames)

                anim = animation.FuncAnimation(
                    fig, animater, interval=1, blit=False, frames=frames, repeat=False
                )
                anim.save(animate + ".mp4", fps=30, extra_args=["-vcodec", "libx264"])
            else:
                if axis is None:
                    plt.show()
            if n_components == 2:
                return
            if n_components == 3 and Get_mesh:
                return triangles, dlist

    def Train(
        self,
        verbose=True,
        crossval=True,
        algorithm="Fisher",
        plot=True,
        boundaryplot=False,
        bins=500,
        labelsL=None,
        labelsR=None,
        savefig="",
        ret=False,
        bounds=None,
        f1=False,
    ):
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn import datasets, svm, metrics
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        import imp
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import balanced_accuracy_score

        X, y = self.X, self.y
        classnames = [self.to_string[0], self.to_string[1]]
        if algorithm == "RBF":
            from sklearn import svm
            from sklearn.model_selection import GridSearchCV

            parameters = {"C": [0.01, 1, 10], "gamma": [1e-1, 1, 1e1]}
            svc = svm.SVC(kernel="rbf", probability=True)
            clf = GridSearchCV(svc, parameters, cv=5, verbose=3, n_jobs=4)
            clf.fit(X, y)
            scores = clf.predict_proba(X)[:, 0]
            if plot and len(self.unique) == 2:
                fpr, tpr, thresholds = metrics.roc_curve(y, scores)
                x0, y0, sy0 = histogram(
                    scores[y == 0], bins=bins, plot=False, remove0=True
                )
                x1, y1, sy1 = histogram(
                    scores[y == 1], bins=bins, plot=False, remove0=True
                )
                fig2, ax = plt.subplots(1, 2, figsize=(12, 8))
                ax[0].errorbar(
                    x0, y0, yerr=sy0, color="b", label=classnames[0], capsize=2
                )
                ax[0].errorbar(
                    x1, y1, yerr=sy1, color="r", label=classnames[1], capsize=2
                )
                if labelsL is None:
                    ax[0].set(
                        xlabel="Predicted probability",
                        ylabel=f"frequency / {1/500:4.2E}",
                        title="Efficiency of the Decision Tree",
                    )
                else:
                    ax[0].set(**labelsL)
                ax[0].legend()

                if ax is None:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax[1].plot(
                    tpr,
                    fpr,
                    color="darkorange",
                    lw=3,
                    label="ROC curve (area = %0.2f)" % metrics.auc(tpr, fpr),
                )
                ax[1].plot([0, 1], [0, 1], color="navy", linewidth=3, linestyle="--")
                if labelsR is None:
                    ax[1].set(
                        xlabel="True Positive Rate",
                        ylabel="False Positve Rate",
                        title=f"ROC for Decision Tree classifier (area = {metrics.auc(tpr,fpr):4.4f})",
                    )
                else:
                    ax[1].set(**labels)
                ax[1].legend()
                if savefig != "":
                    plt.savefig(savefig + ".pdf", dpi=500)
                plt.show()
        if algorithm == "neighbors":
            # parameters = {'n_neighbors':[5,10,30,40]}
            # clf = GridSearchCV(KNeighborsClassifier(), parameters, cv=5)
            clf = KNeighborsClassifier(n_neighbors=10, algorithm="kd_tree")
            clf.fit(X, y)
            print(clf.get_params)
            if plot and len(self.unique) == 2:
                fpr, tpr, thresholds = metrics.roc_curve(y, scores)
                x0, y0, sy0 = histogram(
                    scores[y == 0], bins=bins, plot=False, remove0=True
                )
                x1, y1, sy1 = histogram(
                    scores[y == 1], bins=bins, plot=False, remove0=True
                )
                fig2, ax = plt.subplots(1, 2, figsize=(12, 8))
                ax[0].errorbar(
                    x0, y0, yerr=sy0, color="b", label=classnames[0], capsize=2
                )
                ax[0].errorbar(
                    x1, y1, yerr=sy1, color="r", label=classnames[1], capsize=2
                )
                if labelsL is None:
                    ax[0].set(
                        xlabel="Predicted probability",
                        ylabel=f"frequency / {1/500:4.2E}",
                        title="Efficiency of the Decision Tree",
                    )
                else:
                    ax[0].set(**labelsL)
                ax[0].legend()

                if ax is None:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax[1].plot(
                    tpr,
                    fpr,
                    color="darkorange",
                    lw=3,
                    label="ROC curve (area = %0.2f)" % metrics.auc(tpr, fpr),
                )
                ax[1].plot([0, 1], [0, 1], color="navy", linewidth=3, linestyle="--")
                if labelsR is None:
                    ax[1].set(
                        xlabel="True Positive Rate",
                        ylabel="False Positve Rate",
                        title=f"ROC for Decision Tree classifier (area = {metrics.auc(tpr,fpr):4.4f})",
                    )
                else:
                    ax[1].set(**labels)
                ax[1].legend()
                if savefig != "":
                    plt.savefig(savefig + ".pdf", dpi=500)
                plt.show()
        elif algorithm == "Boost":
            from sklearn.experimental import enable_hist_gradient_boosting
            from sklearn.ensemble import HistGradientBoostingClassifier

            clf = HistGradientBoostingClassifier(max_iter=1000, verbose=0)
            clf.fit(X, y)
            scores = clf.predict_proba(X)[:, 0]
            if plot and len(self.unique) == 2:
                print("computing roc curve")
                fpr, tpr, thresholds = metrics.roc_curve(y, scores)
                x0, y0, sy0 = histogram(
                    scores[y == 0], bins=bins, plot=False, remove0=True
                )
                x1, y1, sy1 = histogram(
                    scores[y == 1], bins=bins, plot=False, remove0=True
                )
                print("making seperation plot")
                fig2, ax = plt.subplots(1, 2, figsize=(12, 8))
                ax[0].errorbar(
                    x0, y0, yerr=sy0, color="b", label=classnames[0], capsize=2
                )
                ax[0].errorbar(
                    x1, y1, yerr=sy1, color="r", label=classnames[1], capsize=2
                )
                if labelsL is None:
                    ax[0].set(
                        xlabel="Predicted probability",
                        ylabel=f"frequency / {1/500:4.2E}",
                        title="Efficiency of the Decision Tree",
                    )
                else:
                    ax[0].set(**labelsL)
                ax[0].legend()
                print("plotting ROC curve")
                if ax is None:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax[1].plot(
                    tpr,
                    fpr,
                    color="darkorange",
                    lw=3,
                    label="ROC curve (area = %0.2f)" % metrics.auc(tpr, fpr),
                )
                ax[1].plot([0, 1], [0, 1], color="navy", linewidth=3, linestyle="--")
                if labelsR is None:
                    ax[1].set(
                        xlabel="True Positive Rate",
                        ylabel="False Positve Rate",
                        title=f"ROC for Decision Tree classifier (area = {metrics.auc(tpr,fpr):4.4f})",
                    )
                else:
                    ax[1].set(**labels)
                ax[1].legend()
                if savefig != "":
                    plt.savefig(savefig + ".pdf", dpi=500)
                print("showing")
                plt.show()
        elif algorithm == "Tree":
            clf = DecisionTreeClassifier()
            clf.fit(X, y)
            scores = clf.predict_proba(X)[:, 0]
            if plot and len(self.unique) == 2:
                fpr, tpr, thresholds = metrics.roc_curve(y, scores)
                x0, y0, sy0 = histogram(
                    scores[y == 0], bins=bins, plot=False, remove0=True
                )
                x1, y1, sy1 = histogram(
                    scores[y == 1], bins=bins, plot=False, remove0=True
                )
                fig2, ax = plt.subplots(1, 2, figsize=(12, 8))
                ax[0].errorbar(
                    x0, y0, yerr=sy0, color="b", label=classnames[0], capsize=2
                )
                ax[0].errorbar(
                    x1, y1, yerr=sy1, color="r", label=classnames[1], capsize=2
                )
                if labelsL is None:
                    ax[0].set(
                        xlabel="Predicted probability",
                        ylabel=f"frequency / {1/500:4.2E}",
                        title="Efficiency of the Decision Tree",
                    )
                else:
                    ax[0].set(**labelsL)
                ax[0].legend()

                if ax is None:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax[1].plot(
                    tpr,
                    fpr,
                    color="darkorange",
                    lw=3,
                    label="ROC curve (area = %0.2f)" % metrics.auc(tpr, fpr),
                )
                ax[1].plot([0, 1], [0, 1], color="navy", linewidth=3, linestyle="--")
                if labelsR is None:
                    ax[1].set(
                        xlabel="True Positive Rate",
                        ylabel="False Positve Rate",
                        title=f"ROC for Decision Tree classifier (area = {metrics.auc(tpr,fpr):4.4f})",
                    )
                else:
                    ax[1].set(**labels)
                ax[1].legend()
                if savefig != "":
                    plt.savefig(savefig + ".pdf", dpi=500)
                plt.show()
        elif algorithm == "Logistic":
            clf = LogisticRegression(max_iter=10000)
            clf.fit(X, y)
            if plot and len(self.unique) == 2:
                scores1 = clf.predict_proba(X)[:, 1]
                scores2 = clf.predict_proba(X)[:, 0]
                fpr, tpr, thresholds = metrics.roc_curve(y, scores2)
                x0, y0, sy0 = histogram(
                    scores1[y == 0], bins=bins, plot=False, remove0=True
                )
                x1, y1, sy1 = histogram(
                    scores1[y == 1], bins=bins, plot=False, remove0=True
                )
                fig2, ax = plt.subplots(1, 2, figsize=(12, 8))
                ax[0].errorbar(
                    x0, y0, yerr=sy0, color="b", label=classnames[0], capsize=2
                )
                ax[0].errorbar(
                    x1, y1, yerr=sy1, color="r", label=classnames[1], capsize=2
                )
                if labelsL is None:
                    ax[0].set(
                        xlabel="Predicted probability",
                        ylabel=f"frequency / {1/500:4.2E}",
                        title="Efficiency of the Logistic Regression",
                    )
                else:
                    ax[0].set(**labelsL)
                ax[0].legend()

                if ax is None:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax[1].plot(
                    tpr,
                    fpr,
                    color="darkorange",
                    lw=3,
                    label="ROC curve (area = %0.2f)" % metrics.auc(tpr, fpr),
                )
                ax[1].plot([0, 1], [0, 1], color="navy", linewidth=3, linestyle="--")
                if labelsR is None:
                    ax[1].set(
                        xlabel="True Positive Rate",
                        ylabel="False Positve Rate",
                        title=f"ROC for Logistic Regression classifier (area = {metrics.auc(tpr,fpr):4.4f})",
                    )
                else:
                    ax[1].set(**labels)
                ax[1].legend()
                if savefig != "":
                    plt.savefig(savefig + ".pdf", dpi=500)
        elif algorithm == "RF":
            X, y = self.X, self.y
            clf = RandomForestClassifier(oob_score=True, n_estimators=20)
            clf = clf.fit(X, y)
            if plot and len(self.unique) == 2:
                scores = clf.oob_decision_function_[
                    ~np.isnan(clf.oob_decision_function_[:, 0])
                ][:, 0]
                fpr, tpr, thresholds = metrics.roc_curve(
                    y[~np.isnan(clf.oob_decision_function_[:, 0])], scores
                )
                scoremask = y[~np.isnan(clf.oob_decision_function_[:, 0])] == 0
                x0, y0, sy0 = histogram(scores[scoremask], bins=bins, plot=False)
                x1, y1, sy1 = histogram(scores[~scoremask], bins=bins, plot=False)
                fig2, ax = plt.subplots(1, 2, figsize=(12, 8))
                ax[0].errorbar(
                    x0, y0, yerr=sy0, color="b", label=classnames[0], capsize=2
                )
                ax[0].errorbar(
                    x1, y1, yerr=sy1, color="r", label=classnames[1], capsize=2
                )
                if labelsL is None:
                    ax[0].set(
                        xlabel="Out of bag error",
                        ylabel=f"frequency / {1/500:4.2E}",
                        title="Efficiency of the Random forest",
                    )
                else:
                    ax[0].set(**labelsL)
                ax[0].legend()

                if ax is None:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax[1].plot(
                    tpr,
                    fpr,
                    color="darkorange",
                    lw=3,
                    label="ROC curve (area = %0.2f)" % metrics.auc(tpr, fpr),
                )
                ax[1].plot([0, 1], [0, 1], color="navy", linewidth=3, linestyle="--")
                if labelsR is None:
                    ax[1].set(
                        xlabel="True Positive Rate",
                        ylabel="False Positve Rate",
                        title=f"ROC for Random Forest classifier (area = {metrics.auc(tpr,fpr):4.4f})",
                    )
                else:
                    ax[1].set(**labels)
                ax[1].legend()
                if savefig != "":
                    plt.savefig(savefig + ".pdf", dpi=500)
        elif algorithm == "Fisher":
            clf = LinearDiscriminantAnalysis(solver="svd")
            clf.fit(X, y)
            if plot and len(self.unique) == 2:
                scores = clf.decision_function(X)
                fpr, tpr, thresholds = metrics.roc_curve(y, scores * -1)
                discr = clf.transform(X)
                bacgr, sig = discr[y == 0], discr[y == 1]
                x0, y0, sy0 = histogram(bacgr, bins=bins, plot=False)
                x1, y1, sy1 = histogram(sig, bins=bins, plot=False)
                # Plot ROC
                fig2, ax = plt.subplots(1, 2, figsize=(12, 8))
                ax[0].errorbar(x0, y0, yerr=sy0, color="b", label=classnames[0])
                ax[0].errorbar(x1, y1, yerr=sy1, color="r", label=classnames[1])
                ax[0].set(
                    xlabel="Fisher discriminant",
                    ylabel=f"frequency / {1/500:4.2E}",
                    title="Efficiency of the FDA",
                )
                ax[0].legend()
                if ax is None:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax[1].plot(
                    tpr,
                    fpr,
                    color="darkorange",
                    lw=3,
                    label="ROC curve (area = %0.2f)" % metrics.auc(tpr, fpr),
                )
                ax[1].plot([0, 1], [0, 1], color="navy", linewidth=3, linestyle="--")
                ax[1].set(
                    xlabel="True Positive Rate",
                    ylabel="False Positve Rate",
                    title=f"ROC for Fisher classifier (area = {metrics.auc(tpr,fpr):4.4f})",
                )
                ax[1].legend()
                if savefig != "":
                    plt.savefig(savefig + "ROC.pdf", dpi=500)
        else:
            raise (ValueError("No such algorithm: " + algorithm))
        self.clf = clf
        if boundaryplot:
            print("Making boundaryplot")
            mlobj = ML(X, y)
            Xraw = np.copy(X)
            if X.shape[-1] != 2:
                mlobj.Reduce("pca", reduce=True, n_components=2)
                pX = mlobj.X
            else:
                pX = X
            # Plot decision map
            if bounds is None:
                xmin, xmax, ymin, ymax = (
                    min(pX[:, 0]),
                    max(pX[:, 0]),
                    min(pX[:, 1]),
                    max(pX[:, 1]),
                )
            else:
                xmin, xmax, ymin, ymax = bounds
            x, ys = np.linspace(xmin, xmax, num=10), np.linspace(ymin, ymax, num=10)
            xx, yy = np.meshgrid(x, ys)
            P = np.zeros((len(xx), len(yy)))
            fig3, ax3 = plt.subplots(1, 2, figsize=(12, 8))
            print("Predicting scatterdata")
            totlen = len(x) * len(ys)
            for i in range(len(xx)):
                for j in range(len(yy)):
                    if (i * len(y) + j) % int(totlen / 20) == 0 and (
                        i * len(y) + j
                    ) != 0:
                        print(str((i * len(y) + j) / int(totlen)) + " % done")
                    point = [x[i], ys[j]]
                    if X.shape[-1] != 2:
                        point = mlobj.T.inverse_transform(point)
                    if len(self.unique) == 2:
                        P[i, j] = self.clf.predict_proba(np.array([point]))[:, 0]
                    else:
                        P[i, j] = self.clf.predict(np.array([point]))
            print("Plotting predictionmap")
            if len(self.unique) == 2:
                cont = ax3[1].imshow(
                    P.T,
                    extent=(xmin, xmax, ymin, ymax),
                    aspect="auto",
                    alpha=0.4,
                    cmap="cool",
                )
            else:
                from matplotlib.colors import LinearSegmentedColormap

                colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"][
                    : len(self.unique)
                ]
                cm = LinearSegmentedColormap.from_list(
                    "name", colors, N=len(self.unique)
                )
                cont = ax3[1].imshow(
                    P,
                    extent=(xmin, xmax, ymin, ymax),
                    aspect="auto",
                    alpha=0.4,
                    cmap=cm,
                )
            cbar = fig3.colorbar(cont, ax=ax3[1], orientation="vertical", pad=0.25)
            ax3[1].set(xlabel="1st axis", ylabel="2nd axis", title="ML predictions")
            print("Plotting points")
            mlobj.ProjectPlot(
                lines=False,
                axis=ax3[0],
                skip=2,
                chull=False,
                points=True,
                xlims=(xmin, xmax),
                ylims=(ymin, ymax),
                lw=0.2,
                s=0.7,
            )
            if len(self.unique) == 2:
                cbar.set_label("% Active")
            else:
                cbar.set_ticks(range(len(self.unique)))
                cbar.set_label("Predicted label")
            fig3.tight_layout()

            if savefig != "":
                plt.savefig(savefig)
            print("showing")
            plt.show()
        # Train on self data
        if verbose:
            print("-----------Training----------- ")
        if crossval:
            if verbose:
                print("Cross validating classifier")
            if f1:
                scores = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted")
            else:
                scores = cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy")
            if verbose:
                print("Accuracy: %0.2f (+/- %0.5f)" % (scores.mean(), scores.std()))
            self.Accuracy = scores.mean()
            self.Accuracyvar = scores.std()
        if ret:
            return fpr, tpr, thresholds, [x0, y0, sy0], [x1, y1, sy1]

    def Predict(
        self,
        inputML,
        label1=None,
        decfunc=False,
        label2=None,
        plot=False,
        verbose=False,
        name="Predicted",
    ):
        """Predicts the type of the points (true gives real type as int)
        based on ML given classifier (You have to have trained the model first)
        """
        # Project points to internal space
        # Center mean and var same as self
        inputML.X = self.scaler.transform(inputML.X)
        # Reduce data dim
        try:
            inputML.X = self.T.transform(inputML.X)
        except:
            pass
        if verbose:
            print("predicting lipase activity")
        try:
            predict = self.clf.predict(inputML.X)
            if not decfunc:
                probact = self.clf.predict_proba(inputML.X)[:, 1] * 100
            else:
                probact = self.clf.decision_function(inputML.X)
        except AttributeError:
            self.Train(plot=False, crossval=False, verbose=False)
            predict = self.clf.predict(inputML.X)
            if not decfunc:
                probact = self.clf.predict_proba(inputML.X)[:, 1] * 100
            else:
                probact = self.clf.decision_function(inputML.X)
        mean = [np.mean(p) for p in probact]
        var = [np.std(p) / 2 for p in probact]
        if verbose:
            print("Output format: prediction, %%%s " % (self.GetTlist()[1]))

        if plot and len(self.X[0]) == 2:
            n_bins = 30
            lipases = []
            fig, ax = plt.subplots(1, 2, figsize=(12, 8))
            ax1, ax2 = ax[0], ax[1]
            labels1 = "avg active frac" + " (%.01f+-%.01f)" % (mean[0], var[0])
            if label1 is not None:
                histogram(probact, bins=n_bins, plot=True, ax=ax1, labels=labels1)
            else:
                histogram(probact, bins=n_bins, plot=True, ax=ax1)
            if label2 is not None:
                histogram(predict, bins=2, plot=True, ax=ax2, labels=labels2, bars=True)
            else:
                histogram(predict, bins=2, plot=True, ax=ax2, bars=True)
            ax1.set(xlabel="Active fraction", title=labels1)
            ax2.set(xlabel="experiment_type", title="Predicted activity")
            plt.tight_layout()
            plt.show()
        return predict, probact

    def Feature_rank(self, savefig=None, show=False, nbins=30, numfeats=5, names=None):
        X, y = self.X, self.y
        Xdat, ydat = self.X, self.y
        if names is None:
            if X.shape[-1] == 17:
                names = np.array(
                    [
                        "alpha",
                        "D",
                        "Pval",
                        "Efficiency",
                        "Dimension",
                        "Gaussianity",
                        "Kurtosis",
                        "MSDratio",
                        "Trappedness",
                        "T0",
                        "T1",
                        "T2",
                        "T3",
                        "<tau>",
                        "N",
                        "meanSL",
                        "meanMSD",
                    ]
                )
            else:
                names = np.arange(X.shape[-1])
        from matplotlib.colors import LinearSegmentedColormap

        colors = [
            matplotlib.colors.to_rgb("dimgrey"),
            matplotlib.colors.to_rgb("darkred"),
        ]  # R -> G -> B
        cbins = 2  # Discretizes the interpolation into bins
        cmap_name = "my_list"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=cbins)
        norm = matplotlib.colors.Normalize(vmin=-10.0, vmax=10.0)
        numfeats = numfeats
        learn = ML(self.X, self.y)
        learn.Reduce("lin", n_components=1)
        learn.clf = learn.T
        sort = np.argsort(np.abs(learn.clf.coef_[0]))
        normweight = np.abs(learn.clf.coef_[0][sort])[::-1][:numfeats] / np.max(
            np.abs(learn.clf.coef_[0][sort])[::-1][:numfeats]
        )

        #
        # colors = ["darkred","darkgreen","dimgrey"]
        nbins = nbins
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        spacing = (1) / (numfeats)

        ax.set(
            xlim=(-0.1, 1.3),
            xticks=np.linspace(0, 1, num=10),
            xticklabels=[f"{i:4.1f}" for i in np.linspace(0, 1, num=10)],
            yticks=np.linspace(0, 1, num=numfeats, endpoint=False),
            xlabel="Normalized Featurevalue",
            yticklabels=names[sort][::-1][:numfeats][::-1],
            ylim=(-1 / (2 * numfeats), 1 - 1 / (2 * numfeats)),
        )  # ,
        # ylabel="Feature name")

        for n, i, top, nimp in zip(
            range(numfeats),
            np.linspace(0, 1, num=numfeats, endpoint=False)[::-1],
            sort[::-1],
            normweight,
        ):
            imp = 1
            L3_dat = Xdat[:, top][ydat == 0]
            Nat_dat = Xdat[:, top][ydat == 1]

            minL3, maxL3 = np.min(L3_dat), np.max(L3_dat)
            minNat, maxNat = np.min(Nat_dat), np.max(Nat_dat)
            minT, maxT = np.min([minL3, minNat]), np.max([maxL3, maxNat])
            bw = (maxT - minT) / nbins

            L3x, L3y, L3sy = histogram(
                L3_dat, bins=nbins, range=(minT, maxT), plot=False, normalize=True
            )
            Natx, Naty, Natsy = histogram(
                Nat_dat, bins=nbins, range=(minT, maxT), plot=False, normalize=True
            )
            diff = L3y - Naty
            normdiff = spacing * diff / (np.max([np.abs(diff)])) / 2
            x = np.linspace(0, imp, num=nbins)
            y = np.linspace(i - spacing / 2, i + spacing / 2, num=nbins)
            X, Y = np.meshgrid(x, y)
            unitynormed_diff = normdiff / np.max(np.abs(normdiff))
            Z = np.array([diff] * len(x))
            map = ax.imshow(
                Z,
                extent=[0, imp, (i - spacing / 2), (i + spacing / 2)],
                interpolation="gaussian",
                cmap=cm,
                norm=norm,
            )

            topcurv = np.max([i + normdiff, i * np.ones(nbins)], axis=0)
            botcurv = np.min([i + normdiff, i * np.ones(nbins)], axis=0)

            def return_closest(n, set):
                return min(set, key=lambda x: abs(x - n))

            topcurv = [val for val in topcurv for _ in (0, 1)]
            botcurv = [val for val in botcurv for _ in (0, 1)]
            topcurv = [val for val in topcurv for _ in (0, 1)]
            botcurv = [val for val in botcurv for _ in (0, 1)]
            topcurv = [val for val in topcurv for _ in (0, 1)]
            botcurv = [val for val in botcurv for _ in (0, 1)]

            ax.fill_between(
                np.linspace(0, imp, num=nbins * 8),
                topcurv,
                i * np.ones(nbins * 8) + spacing / 2,
                color="w",
            )
            ax.fill_between(
                np.linspace(0, imp, num=nbins * 8),
                botcurv,
                i * np.ones(nbins * 8) - spacing / 2,
                color="w",
            )

            ax.plot(np.linspace(0, imp), np.ones(50) * (i), "k", linewidth=1)
            ax.text(imp + 0.1, i, f"{nimp:4.2f}", fontsize=20)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        if savefig is not None:
            fig.savefig(savefig + ".pdf")
        if show:
            plt.show()
