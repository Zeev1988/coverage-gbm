"""Coordinates and radar geometry.

Units: angles in radians in every data table (the product contract); degrees only in
the command-line tools and the report. Positions are converted to a local flat frame
(x east, y north, metres) around an origin = mean radar position. Elevation angles
use the standard 4/3-earth refraction model.
"""
import numpy as np

EARTH_RADIUS_M = 6_371_000.0
K_FACTOR = 4.0 / 3.0


class Frame:
    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.c = np.cos(self.lat0)

    def xy(self, lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        return (lon - self.lon0) * EARTH_RADIUS_M * self.c, (lat - self.lat0) * EARTH_RADIUS_M

    def latlon(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return y / EARTH_RADIUS_M + self.lat0, x / (EARTH_RADIUS_M * self.c) + self.lon0

    def to_dict(self):
        return {"lat0": self.lat0, "lon0": self.lon0}


def polar(dx, dy, alt_target, alt_sensor):
    """slant range (m), azimuth (rad, from north clockwise, [0, 2pi)), elevation (rad)."""
    ground = np.hypot(dx, dy)
    dh = np.asarray(alt_target, float) - alt_sensor
    slant = np.hypot(ground, dh)
    az = np.arctan2(dx, dy) % (2 * np.pi)
    drop = ground ** 2 / (2 * K_FACTOR * EARTH_RADIUS_M)
    el = np.arctan2(dh - drop, ground)
    return slant, az, el


def plot_xy(rng, az, el, sx, sy):
    """Plot position in the local frame from the radar's own polar measurement."""
    g = np.asarray(rng, float) * np.cos(np.asarray(el, float))
    return sx + g * np.sin(az), sy + g * np.cos(az)


def angle_diff(a, b):
    """|a - b| wrapped to [0, pi]"""
    return np.abs((np.asarray(a) - b + np.pi) % (2 * np.pi) - np.pi)
