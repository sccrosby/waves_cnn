from datetime import datetime, timedelta


def matlab_datenum_to_py_date(matlab_datenum):
    """
    Converts Matlab sequential datetimes to Python date times
    Args:
        matlab_datenum:

    Returns:
    """
    # http://sociograph.blogspot.com/2011/04/how-to-avoid-gotcha-when-converting.html
    python_date_time = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)
    return python_date_time


def matlab_datenum_nparray_to_py_date(matlab_datenum_array):
    for i in range(len(matlab_datenum_array)):
        matlab_datenum_array[i] = matlab_datenum_to_py_date(matlab_datenum_array[i])
    return matlab_datenum_array