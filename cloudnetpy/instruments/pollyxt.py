"""Module with a class for PollyXT Raman Polarization lidar."""
from collections import namedtuple
import netCDF4
import numpy as np
import numpy.ma as ma
from cloudnetpy.instruments.ramanlidar import Lidar
from cloudnetpy import utils

instrument_info = namedtuple('instrument_info',
                             ['calibration_factor',
                              'overlap_function_params',
                              'is_range_corrected'])

# TODO: should be a separate config file or accessible over http api
LIDAR_INFO = {
    'punta-arenas': instrument_info(
        calibration_factor=1.0,
        overlap_function_params=None,
        is_range_corrected=True),
}


class PollyXT(Lidar):
    """Class for  PollyXT Raman Polarization lidar."""
    def __init__(self, file_name, site_name, wavelength, tilt_angle):
        super().__init__(file_name)
        self.model = 'PollyXT'
        self.wavelength = wavelength
        self.tilt_angle = tilt_angle
        self.dataset = netCDF4.Dataset(self.file_name)
        self.variables = self.dataset.variables
        # n_range_gates:  Estimates saturated profiles using the variance of the n top range gates
        # variance_limt:  --------------------------------"---------------------------------------
        # saturation_noise: Removes low values in saturated profiles above peak,
        # noise_min: (noise_mimimum, noise_minimum_smoothed)
        self.noise_params = (70, 2e-14, 0.3e-0, (5e-7, 8e-7))
        self.calibration_info = _read_calibration_info(site_name)

    def read_ceilometer_file(self):
        """Reads data and metadata from PollyXT netCDF file."""
        self.range = self._calc_range()
        self.time = self._convert_time()
        self.date = self._read_date()
        self.backscatter = self._convert_backscatter()
        self.metadata = self._read_metadata()

    def _calc_range(self):
        """Assumes 'range' means the upper limit of range gate."""
        # Note: height above instrument is used, convert to 'above sea level'
        ceilo_range = self._getvar('height')
        return ceilo_range - utils.mdiff(ceilo_range)/2

    def _convert_time(self):
        time = self.variables['time']
        try:
            assert all(np.diff(time) > 0)
        except AssertionError:
            raise RuntimeError('Inconsistent lidar time stamps.')
        if max(time) > 24:
            time = utils.seconds2hours(time)
        return time

    def _read_date(self):
        # use date specification in filename rather than non-existing year, month, day parameter
        return [int(self.file_name[:4]), int(self.file_name[5:7]), int(self.file_name[8:10])]

    def _convert_backscatter(self):
        """Steps to convert PollyXT SNR to raw beta."""
        beta_raw = self._getvar(f'attenuated_backscatter_{self.wavelength}nm')
        if not self.calibration_info.is_range_corrected:
            beta_raw *= self.range ** 2
        overlap_function = _get_overlap(self.range, self.calibration_info)
        beta_raw /= overlap_function
        beta_raw *= self.calibration_info.calibration_factor
        return beta_raw

    def _getvar(self, *args):
        """Reads data of variable (array or scalar) from netcdf-file."""
        for arg in args:
            if arg in self.variables:
                var = self.variables[arg]
                return var[0] if utils.isscalar(var) else var[:]

    def _read_metadata(self):
        meta = {'tilt_angle': self.tilt_angle}
        return meta


def _get_overlap(range_ceilo, calibration_info):
    """Approximative overlap function."""
    params = calibration_info.overlap_function_params or (0, 1)
    return utils.array_to_probability(range_ceilo, *params)


def _read_calibration_info(site_name):
    if 'punta' in site_name.lower():
        return LIDAR_INFO['punta-arenas']
    elif 'mace' in site_name.lower():
        return LIDAR_INFO['mace-head']
    else:
        return LIDAR_INFO[site_name.lower()]
