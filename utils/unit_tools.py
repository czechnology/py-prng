from math import log10


def nicer_time(t, unit='s'):
    """Print time in a nice form"""

    if t == 0:
        return "0s"

    t = float(t)
    if unit == 'us':
        seconds = t / 1000000
    elif unit == 'ms':
        seconds = t / 1000
    elif unit == 's':
        seconds = t
    elif unit == 'm':
        seconds = t * 60
    elif unit == 'h':
        seconds = t * 3600

    elif unit == 'd':
        seconds = t * 86400
    elif unit == 'w':
        seconds = t * 604800
    else:
        raise ValueError("Unknown time unit " + unit)

    dec_places = int(max(3 - log10(seconds), 0))

    minutes, seconds = divmod(seconds, 60)
    time_str = str(round(seconds, dec_places if dec_places > 0 else None)) + 's'
    if minutes <= 0:
        return time_str

    hours, minutes = divmod(minutes, 60)
    time_str = str(int(minutes)) + 'm ' + time_str
    if hours <= 0:
        return time_str

    days, hours = divmod(hours, 24)
    time_str = str(int(hours)) + 'h ' + time_str
    if days <= 0:
        return time_str

    weeks, days = divmod(days, 7)
    time_str = str(int(days)) + 'd ' + time_str
    if weeks <= 0:
        return time_str

    time_str = str(int(weeks)) + 'w ' + time_str
    return time_str
