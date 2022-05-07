from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import numpy as np
"""
First-order smooth spline
"""
def SmoothSpline(x, y, Xpred, weight, smooth=0.1):
    spl = UnivariateSpline(x, y, w=weight, ext=3)
    spl.set_smoothing_factor(smooth)
    return spl(Xpred)


"""
Savitzky-Golay filter (first-order filter)
"""
def SGFilter(y, order=1, WindowLength=51):
    return savgol_filter(y, WindowLength, order)

"""
Main function for getting initial velocities
"""
def InitialVelocity(specturm):
    ############################################################
    # get the points with maximum semblance at each time point
    ############################################################
    SelectedRow = np.where(np.max(specturm, axis=1) > 0)[0]
    MaxIndex = np.argmax(specturm[SelectedRow, :], axis=1)
    MaxPoints = np.array([SelectedRow, MaxIndex], dtype=np.float32).T
    ValueList = specturm[SelectedRow, MaxIndex]
    #######################
    # first order spline
    #######################
    SmoothV = SmoothSpline(MaxPoints[:, 0], MaxPoints[:, 1], np.arange(specturm.shape[0]), ValueList, smooth=0.1)

    ############################################################
    # Savitzky-Golay filter (first-order filter)
    ############################################################
    InitialV = SGFilter(SmoothV, order=1, WindowLength=51)
    return InitialV