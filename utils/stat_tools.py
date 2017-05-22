from math import copysign

import numpy as np


def standard_normal_distr_quantile(p):
    """Calculate percentile of the standard normal distribution. P(X<=x)=p.
    See https://en.wikipedia.org/wiki/Probit for details"""

    # calculation is complicated and requires extra libraries, so use pre-computed values when
    # applicable (values from Menezes and Gurker)

    a = [.4000, .2500, .2000, 0.1500, 0.1000, 0.0500, 0.025, 0.0100, 0.0050, 0.0025, 0.0010, 0.0005]
    x = [.2533, .6745, .8416, 1.0364, 1.2816, 1.6449, 1.960, 2.3263, 2.5758, 2.8070, 3.0902, 3.2905]

    try:
        p_x = round((p if p < .5 else 1 - p), 4)  # round to fix rounding errors
        return copysign(x[a.index(p_x)], p - .5)
    except ValueError:
        raise ValueError("This quantile is not available in the pre-computed values")  # not found


def chi_squared_distr_quantile(p, dof=1):
    """Calculate percentile of the chi-squared distribution. P(X<=x)=p.
    See https://en.wikipedia.org/wiki/Probit for details"""

    # calculation is complicated and requires extra libraries, so use pre-computed values when
    # applicable (values from Menezes)

    # values for p = 0.900 0.950 0.975 0.090 0.995 0.999
    p_values = [.900, .950, .975, .990, .995, .999]
    percentiles = {
        1: [2.7055, 3.8415, 5.0239, 6.6349, 7.8794, 10.8276],
        2: [4.6052, 5.9915, 7.3778, 9.2103, 10.5966, 13.8155],
        3: [6.2514, 7.8147, 9.3484, 11.3449, 12.8382, 16.2662],
        4: [7.7794, 9.4877, 11.1433, 13.2767, 14.8603, 18.4668],
        5: [9.2364, 11.0705, 12.8325, 15.0863, 16.7496, 20.5150],
        6: [10.6446, 12.5916, 14.4494, 16.8119, 18.5476, 22.4577],
        7: [12.0170, 14.0671, 16.0128, 18.4753, 20.2777, 24.3219],
        8: [13.3616, 15.5073, 17.5345, 20.0902, 21.9550, 26.1245],
        9: [14.6837, 16.9190, 19.0228, 21.6660, 23.5894, 27.8772],
        10: [15.9872, 18.3070, 20.4832, 23.2093, 25.1882, 29.5883],
        11: [17.2750, 19.6751, 21.9200, 24.7250, 26.7568, 31.2641],
        12: [18.5493, 21.0261, 23.3367, 26.2170, 28.2995, 32.9095],
        13: [19.8119, 22.3620, 24.7356, 27.6882, 29.8195, 34.5282],
        14: [21.0641, 23.6848, 26.1189, 29.1412, 31.3193, 36.1233],
        15: [22.3071, 24.9958, 27.4884, 30.5779, 32.8013, 37.6973],
        16: [23.5418, 26.2962, 28.8454, 31.9999, 34.2672, 39.2524],
        17: [24.7690, 27.5871, 30.1910, 33.4087, 35.7185, 40.7902],
        18: [25.9894, 28.8693, 31.5264, 34.8053, 37.1565, 42.3124],
        19: [27.2036, 30.1435, 32.8523, 36.1909, 38.5823, 43.8202],
        20: [28.4120, 31.4104, 34.1696, 37.5662, 39.9968, 45.3147],
        21: [29.6151, 32.6706, 35.4789, 38.9322, 41.4011, 46.7970],
        22: [30.8133, 33.9244, 36.7807, 40.2894, 42.7957, 48.2679],
        23: [32.0069, 35.1725, 38.0756, 41.6384, 44.1813, 49.7282],
        24: [33.1962, 36.4150, 39.3641, 42.9798, 45.5585, 51.1786],
        25: [34.3816, 37.6525, 40.6465, 44.3141, 46.9279, 52.6197],
        26: [35.5632, 38.8851, 41.9232, 45.6417, 48.2899, 54.0520],
        27: [36.7412, 40.1133, 43.1945, 46.9629, 49.6449, 55.4760],
        28: [37.9159, 41.3371, 44.4608, 48.2782, 50.9934, 56.8923],
        29: [39.0875, 42.5570, 45.7223, 49.5879, 52.3356, 58.3012],
        30: [40.2560, 43.7730, 46.9792, 50.8922, 53.6720, 59.7031],
        31: [41.4217, 44.9853, 48.2319, 52.1914, 55.0027, 61.0983],
        63: [77.7454, 82.5287, 86.8296, 92.0100, 95.6493, 103.4424],
        127: [147.8048, 154.3015, 160.0858, 166.9874, 171.7961, 181.9930],
        255: [284.3359, 293.2478, 301.1250, 310.4574, 316.9194, 330.5197],
        511: [552.3739, 564.6961, 575.5298, 588.2978, 597.0978, 615.5149],
        1023: [1081.3794, 1098.5208, 1113.5334, 1131.1587, 1143.2653, 1168.4972]
    }

    try:
        index = p_values.index(p)
    except ValueError:
        raise ValueError("This quantile is not available in the pre-computed values")  # not found

    return percentiles[dof][index]


def boxplot_params(data):
    """Parameters for a box plot, as per J.W.Tukey."""
    q1, q2, q3 = map(float, np.percentile(data, [25, 50, 75], interpolation='midpoint'))

    h = 1.5 * (q3 - q1)
    lf, uf = q1 - h, q3 + h
    # out_below = filter(lambda x: x < lf, data)

    # separate data
    outliers = []
    lw = None
    uw = None
    for d in data:
        if lf <= d <= uf:
            if lw is None or d < lw:
                lw = d
            if uw is None or d > uw:
                uw = d
        else:
            outliers.append(d)
    if lw is None:
        lw = lf
    if uw is None:
        uw = uf

    return dict(lq=q1, med=q2, uq=q3, h=h, lf=lf, uf=uf, lw=lw, uw=uw, out=outliers)
