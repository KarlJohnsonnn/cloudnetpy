"""Module for reading and processing PollyXT (processed PollyNET files)"""
import linecache
import numpy as np
from cloudnetpy.instruments.pollyxt import PollyXT
from cloudnetpy import utils, output, CloudnetArray
from cloudnetpy.metadata import MetaData


def lidar2chunk(input_file, site_meta):
    """Converts PollyNET files **att_bsc** files into calibrated netCDF file.

    This function reads PollyNET *_att_bsc.nc files writes the data into netCDF file.
    Three variants of the attenuated backscatter for a specific wavelength
    (312, 523, 1064) are saved in the file:

        1. Raw backscatter, `beta_raw`
        2. Signal-to-noise screened backscatter, `beta`
        3. SNR-screened backscatter with smoothed weak background, `beta_smooth`

    Args:
        input_file (str): Raman Lidar file name. For Vaisala it is a text file,
            for Jenoptik it is a netCDF file.
        site_meta (dict): Dictionary containing information about the
            site. Required key value pairs are `name` and `altitude`
            (metres above mean sea level).

    Raises:
        RuntimeError: Failed to read or process raw Raman Lidar data.

    Examples:
        >>> from cloudnetpy.instruments import lidar2chunk
        >>> site_meta = {'name': 'Punta-Arenas', 'altitude':9}
        >>> lidar2nc('pollynet_att_bsc.nc', '.nc', site_meta)

    """
    lidar = _initialize_lidar(input_file, site_meta['name'], site_meta['wavelength'], site_meta['tilt_angle'])
    lidar.read_lidarmeter_file()
    beta_variants = lidar.calc_beta()
    _append_data(lidar, beta_variants)
    _append_height(lidar, site_meta['altitude'])
    output.update_attributes(lidar.data, ATTRIBUTES)
    return lidar, site_meta['name']


def lidar2nc(input_file, output_file, site_meta, keep_uuid=False):
    """Converts PollyNET files **att_bsc** files into calibrated netCDF file.

    OUTER LOOP concatenating multiple files
    Args:
        input_file (str or list): Raman Lidar file name or list of file names. For Vaisala it is a text file,
            for Jenoptik it is a netCDF file.
        site_meta (dict): Dictionary containing information about the
            site. Required key value pairs are `name` and `altitude`
            (metres above mean sea level).
        output_file (str): Output file name, e.g. 'lidar.nc'.
        keep_uuid (bool, optional): If True, keeps the UUID of the old file,
            if that exists. Default
    Returns:
        str: UUID of the generated file.is False when new UUID is generated.
    """
    if isinstance(input_file, str):
        input_file = [input_file]
    elif isinstance(input_file, list):
        input_file = sorted(input_file)
    else:
        raise RuntimeError('Error: Unknown lidar files.')

    lidar_list = []
    for file_ in input_file:
        lidar, site_meta['name'] = lidar2chunk(file_, site_meta)
        lidar_list.append(lidar)

    return _save_lidar(lidar_list, output_file, site_meta['name'], keep_uuid)

def _initialize_lidar(file, site_name, wavelength, tilt_angle):
    model = _find_lidar_model(file)
    if model == 'pollyxt':
        return PollyXT(file, site_name, wavelength, tilt_angle)
    raise RuntimeError('Error: Unknown lidar model.')


def _find_lidar_model(file):
    if file.endswith('nc'):
        if 'att_bsc' in file.lower():  # name convention for pollynet files
            return 'pollyxt'
    return None


def _append_height(lidar, site_altitude):
    """Finds height above mean sea level."""
    tilt_angle = np.median(lidar.metadata['tilt_angle'])
    height = utils.range_to_height(lidar.range, tilt_angle)
    height += float(site_altitude)
    lidar.data['height'] = CloudnetArray(height, 'height')


def _append_data(lidar, beta_variants):
    """Add data and metadata as CloudnetArray's to lidar.data attribute."""
    for data, name in zip(beta_variants, ('beta_raw', 'beta', 'beta_smooth')):
        lidar.data[name] = CloudnetArray(data, name)
    for field in ('range', 'time'):
        lidar.data[field] = CloudnetArray(getattr(lidar, field), field)
    for field, data in lidar.metadata.items():
        first_element = data if utils.isscalar(data) else data[0]
        if not isinstance(first_element, str):  # String array writing not yet supported
            lidar.data[field] = CloudnetArray(np.array(lidar.metadata[field],
                                                       dtype=float), field)
    if hasattr(lidar, 'wavelength'):
        lidar.data['wavelength'] = CloudnetArray(lidar.wavelength, 'wavelength', 'nm')


def _save_lidar(file_list, output_file, location, keep_uuid):
    """Saves the Raman Lidar netcdf-file."""
    anker = 0

    file_data = {key: var for key, var in file_list[anker].data.items()}
    file_data['time'].data = np.ma.concatenate([file_list[i].data['time'].data for i in range(len(file_list))])

    # because polly hast >30 seconds time resolution
    args = (file_data['time'].data, file_data['range'].data)
    argsnew = (np.linspace(args[0][0], args[0][-1], 2 * args[0].size), file_data['range'].data)
    file_data['time'].data = np.ma.masked_array(argsnew[0])

    dims = {'time': argsnew[0].size, 'range': len(file_list[anker].range)}

    for var in ['beta', 'beta_raw', 'beta_smooth']:
        file_beta = [file.data[var].data for file in file_list if var in file.data.keys()]
        if len(file_beta) > 0:
            file_data[var].data = utils.interpolate_2d_masked(np.concatenate(file_beta), args, argsnew)

    rootgrp = output.init_file(output_file, dims, file_data, keep_uuid)
    uuid = rootgrp.file_uuid
    output.add_file_type(rootgrp, 'lidar')
    if hasattr(file_list[anker], 'dataset'):
        output.copy_variables(file_list[anker].dataset, rootgrp, ('wavelength',))
    rootgrp.title = f"Raman Lidar file from {location}"
    rootgrp.year, rootgrp.month, rootgrp.day = file_list[anker].date
    rootgrp.location = location
    rootgrp.history = f"{utils.get_time()} - Raman Lidar file created"
    rootgrp.source = file_list[anker].model
    output.add_references(rootgrp)
    rootgrp.close()
    return uuid

ATTRIBUTES = {
    'beta': MetaData(
        long_name='Attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment='Range corrected, SNR screened, attenuated backscatter.'
    ),
    'beta_raw': MetaData(
        long_name='Raw attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment="Range corrected, attenuated backscatter."
    ),
    'beta_smooth': MetaData(
        long_name='Smoothed attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment=('Range corrected, SNR screened backscatter coefficient.\n'
                 'Weak background is smoothed using Gaussian 2D-kernel.')
    ),
    'scale': MetaData(
        long_name='Scale',
        units='%',
        comment='100 (%) is normal.'
    ),
    'software_level': MetaData(
        long_name='Software level ID',
        units='',
    ),
    'laser_temperature': MetaData(
        long_name='Laser temperature',
        units='C',
    ),
    'window_transmission': MetaData(
        long_name='Window transmission estimate',
        units='%',
    ),
    'tilt_angle': MetaData(
        long_name='Tilt angle from vertical',
        units='degrees',
    ),
    'laser_energy': MetaData(
        long_name='Laser pulse energy',
        units='%',
    ),
    'background_light': MetaData(
        long_name='Background light',
        units='mV',
        comment='Measured at internal ADC input.'
    ),
    'backscatter_sum': MetaData(
        long_name='Sum of detected and normalized backscatter',
        units='sr-1',
        comment='Multiplied by scaling factor times 1e4.',
    ),
    'range_resolution': MetaData(
        long_name='Range resolution',
        units='m',
    ),
    'number_of_gates': MetaData(
        long_name='Number of range gates in profile',
        units='',
    ),
    'unit_id': MetaData(
        long_name='Raman Lidar unit number',
        units='',
    ),
    'message_number': MetaData(
        long_name='Message number',
        units='',
    ),
    'message_subclass': MetaData(
        long_name='Message subclass number',
        units='',
    ),
    'detection_status': MetaData(
        long_name='Detection status',
        units='',
        comment='From the internal software of the instrument.'
    ),
    'warning': MetaData(
        long_name='Warning and Alarm flag',
        units='',
        definition=('\n'
                    'Value 0: Self-check OK\n'
                    'Value W: At least one warning on\n'
                    'Value A: At least one error active.')
    ),
    'warning_flags': MetaData(
        long_name='Warning flags',
        units='',
    ),
    'receiver_sensitivity': MetaData(
        long_name='Receiver sensitivity',
        units='%',
        comment='Expressed as % of nominal factory setting.'
    ),
    'window_contamination': MetaData(
        long_name='Window contamination',
        units='mV',
        comment='Measured at internal ADC input.'
    )
}
