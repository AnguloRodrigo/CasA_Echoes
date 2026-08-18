"""
Extract flux profiles from infrared echoes found in JWST NIRCam images.

Author: Rodrigo Angulo
Code helped by Claude (Opus)
"""


import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from jwst.datamodels import ImageModel


class NIRCamProfileExtractor:
    
    def __init__(self, filename):
        """Load JWST image data."""
        self.model = ImageModel(filename)
        self.data = self.model.data.astype(float)
        self.err = self.model.err.astype(float) if self.model.err is not None else None
        self.dq = self.model.dq if self.model.dq is not None else None
        self.wcs = self.model.meta.wcs
        
        self.pixel_scale = self._compute_pixel_scale()
        
        # self.roll_angle = self.model.meta.wcsinfo.roll_ref  # degrees
        # self.roll_angle = self.model.meta.wcsinfo.PA_APER
        # self.roll_angle = self._compute_roll_angle()
        self.roll_angle = fits.getval(filename, 'PA_APER', ext=1)
        
        self._print_info()
    
    def _compute_pixel_scale(self):
        """Compute pixel scale from WCS."""
        x0, y0 = self.data.shape[1] // 2, self.data.shape[0] // 2
        coord1 = self.wcs.pixel_to_world(x0, y0)
        coord2 = self.wcs.pixel_to_world(x0 + 1, y0)
        return coord1.separation(coord2).arcsec
    
    def _compute_roll_angle(self):
        """Compute roll angle from WCS by checking North direction."""
        x0, y0 = self.data.shape[1] // 2, self.data.shape[0] // 2
        
        # Get sky coords at center and slightly north in pixel coords
        coord_center = self.wcs.pixel_to_world(x0, y0)
        coord_up = self.wcs.pixel_to_world(x0, y0 + 10)
        
        # Position angle from center to "up" in pixels
        pa_of_pixel_y = coord_center.position_angle(coord_up).deg
        
        # Roll angle: how much is +y rotated from North
        # If pa_of_pixel_y = 0, then +y points North, roll = 0
        return pa_of_pixel_y
    
    def _print_info(self):
        """Print useful information."""
        print('-------------------------------')
        print(f"File: {self.model.meta.filename}")
        print(f"Filter: {self.model.meta.instrument.filter}")
        print(f"Image shape: {self.data.shape}")
        print(f"Pixel scale: {self.pixel_scale:.4f} arcsec/pixel")
        print(f"Roll angle: {self.roll_angle:.2f} degrees")
        print(f"Obs date: {self.model.meta.observation.date}")
        print('-------------------------------')
    
    def close(self):
        self.model.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def sky_pa_to_detector_pa(self, sky_pa):
        """
        Convert position angle on sky to position angle in detector coordinates.
        Parameter -- sky_pa : float; Position angle on sky in degrees (East of North)
        Returns -- detector_pa : float; Position angle in detector coordinates (counterclockwise from +y)
        """
        # The roll angle tells us how much the detector is rotated relative to sky
        detector_pa = self.roll_angle - sky_pa
        return detector_pa
    
    def detector_pa_to_sky_pa(self, detector_pa):
        """Convert detector PA to sky PA."""
        return self.roll_angle - detector_pa
    
    def get_pixel_coords(self, ra, dec):
        """Convert world to pixel coordinates."""
        coord = SkyCoord(ra, dec, unit='deg')
        return self.wcs.world_to_pixel(coord)
    
    def get_world_coords(self, x, y):
        """Convert pixel to world coordinates."""
        return self.wcs.pixel_to_world(x, y)
    
    def extract_profile_sky(self, center_ra, center_dec, length_arcsec, width_arcsec, sky_pa=0, n_samples=None, mask_dq=True):
        """
        Extract profile with position angle defined on sky.
        Converts sky PA to detector PA, ensuring consistent extraction across epochs.
        
        Parameters --
        center_ra, center_dec : float; Center coordinates in degrees
        length_arcsec : float; Length of rectangle in arcsec
        width_arcsec : float; Width of rectangle in arcsec
        sky_pa : float; Position angle ON SKY in degrees (East of North)
        n_samples : int, optional; Number of samples along profile
        mask_dq : bool; Mask bad pixels from DQ array
        
        Returns -- dict with profile data and metadata
        """
        # Convert sky PA to detector PA for this specific image
        detector_pa = self.sky_pa_to_detector_pa(sky_pa)
        
        # Get pixel coordinates of center
        center_x, center_y = self.get_pixel_coords(center_ra, center_dec)
        center_x, center_y = float(center_x), float(center_y)
        
        # Convert dimensions to pixels
        length_pix = length_arcsec / self.pixel_scale
        width_pix = width_arcsec / self.pixel_scale
        
        if n_samples is None:
            n_samples = int(np.ceil(length_pix))
        
        n_width = int(np.ceil(width_pix))
        
        # Coordinates relative to center
        l_coords = np.linspace(-length_pix / 2, length_pix / 2, n_samples)
        w_coords = np.linspace(-width_pix / 2, width_pix / 2, n_width)
        
        L, W = np.meshgrid(l_coords, w_coords)
        
        # Rotate using DETECTOR PA
        angle_rad = np.radians(detector_pa + 180.)     # personal choice -- flip box for extraction -- just want echo moving left to right of box
        
        X = center_x + L * np.sin(angle_rad) + W * np.cos(angle_rad)
        Y = center_y + L * np.cos(angle_rad) - W * np.sin(angle_rad)
        
        # Prepare data
        data_to_extract = self.data.copy()
        if mask_dq and self.dq is not None:
            data_to_extract[self.dq > 0] = np.nan
        
        # Extract
        extracted = map_coordinates(data_to_extract, [Y, X], order=1, mode='constant', cval=np.nan)
        
        # Error propagation
        if self.err is not None:
            extracted_err = map_coordinates(self.err, [Y, X], order=1, mode='constant', cval=np.nan)
            flux_err = np.sqrt(np.nansum(extracted_err**2, axis=0))
        else:
            flux_err = None
        
        # Sum along width
        flux = np.nansum(extracted, axis=0)
        xax_arcsec = l_coords * self.pixel_scale
        
        result_dict = {
                       'profile_data': {
                                        'xax_arcsec': xax_arcsec,
                                        'flux': flux,
                                        'flux_err': flux_err,
                                        'extracted_2d': extracted,   # Claude saved this, might be useful to look at
                                       },
                       'profile_box': {
                                       'center_x': center_x,
                                       'center_y': center_y,
                                       'center_ra': center_ra,
                                       'center_dec': center_dec,
                                       'sky_pa': sky_pa,
                                       'detector_pa': detector_pa,
                                       'length_arcsec': length_arcsec,
                                       'width_arcsec': width_arcsec,
                                       'length_pix': length_pix,
                                       'width_pix': width_pix,
                                      },
                       'fits_info': {
                                     'filename': self.model.meta.filename,
                                     'obs_date': self.model.meta.observation.date_end,
                                     'filter': self.model.meta.instrument.filter,
                                     'roll_angle': self.roll_angle,
                                     'pixel_scale': self.pixel_scale,
                                     'flux_units': str(self.model.meta.bunit_data),
                                    },
                      }
        return(result_dict)
    
    def get_rectangle_corners(self, profile_result):
        """Get corner coordinates using detector PA."""
        
        profile_box = profile_result['profile_box']
        
        cx = profile_box['center_x']
        cy = profile_box['center_y']
        length = profile_box['length_pix']
        width = profile_box['width_pix']
        # Use detector PA for plotting on image
        angle_rad = np.radians(profile_box['detector_pa'] + 180.)    # personal choice -- want box flipped for readout
        
        hl, hw = length / 2, width / 2
        corners_local = np.array([[-hl, -hw], [+hl, -hw], [+hl, +hw], [-hl, +hw], [-hl, -hw]])
        
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        corners_x = cx + corners_local[:, 0] * sin_a + corners_local[:, 1] * cos_a
        corners_y = cy + corners_local[:, 0] * cos_a - corners_local[:, 1] * sin_a
        
        return corners_x, corners_y
    
    def plot_extraction(self, profile_result, vmin=None, vmax=None, figsize=(14, 5), zoom_factor=1.5, just_show=False):
        """Visualize extraction."""

        profile_data = profile_result['profile_data']
        profile_box = profile_result['profile_box']
        fits_info = profile_result['fits_info']
        
        fig = plt.figure(figsize=figsize)
        
        ax1 = fig.add_subplot(131)
        if vmin is None or vmax is None:
            vmin, vmax = np.nanpercentile(self.data, [5, 95])
        
        ax1.imshow(self.data, origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
        
        corners_x, corners_y = self.get_rectangle_corners(profile_result)
        ax1.plot(corners_x, corners_y, 'r-', lw=2)
        ax1.plot(profile_box['center_x'], profile_box['center_y'], 'r+', ms=10, mew=2)
        
        # Add North arrow
        self._add_north_arrow(ax1, profile_result, zoom_factor=zoom_factor)
        
        cx, cy = profile_box['center_x'], profile_box['center_y']
        max_dim = max(profile_box['length_pix'], profile_box['width_pix']) * zoom_factor
        ax1.set_xlim(cx - max_dim, cx + max_dim)
        ax1.set_ylim(cy - max_dim, cy + max_dim)
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_title(f"Sky PA={profile_box['sky_pa']:.1f}° (Det PA={profile_box['detector_pa']:.1f}°)")
        
        # 2D extraction
        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(profile_data['extracted_2d'], origin='lower', aspect='auto', cmap='viridis', extent=[profile_data['xax_arcsec'][0], profile_data['xax_arcsec'][-1], -profile_box['width_arcsec'] / 2, +profile_box['width_arcsec'] / 2])
        ax2.set_xlabel('Position along profile (arcsec)')
        ax2.set_ylabel('Width (arcsec)')
        ax2.set_title('Extracted Region')
        plt.colorbar(im2, ax=ax2, label=fits_info['flux_units'])
        
        # 1D profile
        ax3 = fig.add_subplot(133)
        pos = profile_data['xax_arcsec']
        flux = profile_data['flux']
        
        if profile_data['flux_err'] is not None:
            ax3.errorbar(pos, flux, yerr=profile_data['flux_err'], fmt='b-', lw=1, elinewidth=0.5)
        else:
            ax3.plot(pos, flux, 'b-', lw=1)
        
        ax3.axvline(0, color='gray', ls='--', lw=0.5)
        ax3.set_xlabel('Position along profile (arcsec)')
        ax3.set_ylabel(f"Flux ({fits_info['flux_units']})")
        ax3.set_title(f"{fits_info['obs_date']}")
        
        plt.tight_layout()
        if just_show is False:
            return fig, (ax1, ax2, ax3)
        elif just_show is True:
            plt.show()
    
    def _add_north_arrow(self, ax, profile_result, arrow_length=20, zoom_factor=1.5):
        """Add North-East compass to image."""

        profile_box = profile_result['profile_box']
        
        cx, cy = profile_box['center_x'], profile_box['center_y']
        
        # Arrow base position (offset from center)
        max_dim = max(profile_box['length_pix'], profile_box['width_pix']) * zoom_factor
        base_x = cx - max_dim * 0.7
        base_y = cy + max_dim * 0.7
        
        # North direction in detector (use roll angle)
        roll_rad = np.radians(self.roll_angle)
        
        # North is at angle roll_angle from +y axis
        north_dx = arrow_length * np.sin(roll_rad)
        north_dy = arrow_length * np.cos(roll_rad)
        
        # East is 90 degrees from North
        east_dx = arrow_length * np.cos(roll_rad)
        east_dy = -arrow_length * np.sin(roll_rad)
        
        ax.arrow(base_x, base_y, north_dx, north_dy, head_width=3, head_length=2, fc='white', ec='white', lw=1.5)
        ax.arrow(base_x, base_y, east_dx, east_dy, head_width=3, head_length=2, fc='white', ec='white', lw=1.5)
        
        ax.text(base_x + north_dx * 1.3, base_y + north_dy * 1.3, 'N', color='white', fontsize=10, ha='center', va='center', fontweight='bold')
        ax.text(base_x + east_dx * 1.3, base_y + east_dy * 1.3, 'E', color='white', fontsize=10, ha='center', va='center', fontweight='bold')


def plot_multi_epoch_comparison(profiles, figsize=(12, 8)):
    """Compare profiles across epochs."""
    
    n_epochs = len(profiles)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Top: All profiles overlaid
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_epochs))
    
    for i, (profile, color) in enumerate(zip(profiles, colors)):
        
        profile_data = profile['profile_data']
        fits_info = profile['fits_info']
        
        label = f"{fits_info['obs_date']} (roll={fits_info['roll_angle']:.1f}°)"
        ax1.plot(profile_data['xax_arcsec'], profile_data['flux'], color=color, lw=1.5, label=label)
    
    ax1.set_ylabel(f"Flux ({profiles[0]['fits_info']['flux_units']})")
    ax1.set_title(f"Multi-epoch profiles at Sky PA = {profiles[0]['profile_box']['sky_pa']:.1f}°")
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax1.axvline(0, color='gray', ls='--', lw=0.5)
    
    # Bottom: Waterfall plot
    ax2 = axes[1]
    for i, (profile, color) in enumerate(zip(profiles, colors)):
        
        profile_data = profile['profile_data']
        fits_info = profile['fits_info']
        
        offset = i * np.nanmax(profile_data['flux']) * 0.3
        ax2.plot(profile_data['xax_arcsec'], profile_data['flux'] + offset, color=color, lw=1)
        ax2.text(profile_data['xax_arcsec'][-1] * 1.02, profile_data['flux'][-1] + offset, fits_info['obs_date'], fontsize=8, va='center')
    
    ax2.set_xlabel('Position along profile (arcsec)')
    ax2.set_ylabel('Flux + offset')
    ax2.set_title('Waterfall plot')
    
    plt.tight_layout()
    return fig, axes


def save_profile_fits(profile_result, output_filename):
    """
    Save extracted profile to a FITS file with data and metadata.
    """

    profile_data = profile_result['profile_data']
    profile_box = profile_result['profile_box']
    fits_info = profile_result['fits_info']
    
    # Create primary HDU with metadata
    primary_hdu = fits.PrimaryHDU()
    header = primary_hdu.header
    
    # Store scalar metadata in header
    header['CENT_X'] = (profile_box['center_x'], 'Profile box center X pixel')
    header['CENT_Y'] = (profile_box['center_y'], 'Profile box center Y pixel')
    header['CENT_RA'] = (profile_box['center_ra'], 'Profile box center RA [deg]')
    header['CENT_DEC'] = (profile_box['center_dec'], 'Profile box center Dec [deg]')
    header['SKY_PA'] = (profile_box['sky_pa'], 'Profile box sky position angle [deg]')
    header['DET_PA'] = (profile_box['detector_pa'], 'Profile box detector position angle [deg]')
    header['LENGTH'] = (profile_box['length_arcsec'], 'Profile box length [arcsec]')
    header['WIDTH'] = (profile_box['width_arcsec'], 'Profile box width [arcsec]')
    header['LENGT_PX'] = (profile_box['length_pix'], 'Profile box length [pix]')
    header['WIDTH_PX'] = (profile_box['width_pix'], 'Profile box width [pix]')
    
    header['FILENAME'] = (fits_info['filename'], 'Image filename')
    header['OBSDATE'] = (fits_info['obs_date'], 'Date of observation')
    header['FILTER'] = (fits_info['filter'], 'Filter')
    header['ROLL_ANG'] = (fits_info['roll_angle'], 'Image aperture position angle [deg]')
    header['BUNIT'] = (fits_info['flux_units'], 'Flux units')
    header['PIXSCALE'] = (fits_info['pixel_scale'], 'Pixel scale [arcsec/pix]')
    
    # Create table with 1D profile data
    profile_table = Table()
    profile_table['xax_arcsec'] = profile_data['xax_arcsec']
    profile_table['flux'] = profile_data['flux']
    if profile_data['flux_err'] is not None:
        profile_table['flux_err'] = profile_data['flux_err']
    
    table_hdu = fits.BinTableHDU(profile_table, name='PROFILE')
    
    # Create image HDU with 2D extracted data
    image_hdu = fits.ImageHDU(profile_data['extracted_2d'], name='EXTRACTED_2D')
    
    # Combine into HDU list
    hdul = fits.HDUList([primary_hdu, table_hdu, image_hdu])
    
    # Write to file
    hdul.writeto(output_filename, overwrite=True)
    print(f"Saved profile to {output_filename}")


def load_profile_fits(filename):
    """
    Load extracted profile from FITS file.
    """

    def load_fits_native_endian(filename):
        """Load FITS data with native byte order."""
        with fits.open(filename) as hdul:
            data = hdul['SCI'].data.astype(hdul['SCI'].data.dtype.newbyteorder('='))
        return data
    
    with fits.open(filename) as hdul:
        header = hdul[0].header
        
        # Read 1D profile
        profile_table = Table.read(hdul['PROFILE'])
        
        # Read 2D extracted data
        extracted_2d = hdul['EXTRACTED_2D'].data
        
        # Reconstruct result dictionary
        result = {
                  'profile_data': {
                                     'xax_arcsec': np.array(profile_table['xax_arcsec'], dtype='<f8'),
                                     'flux': np.array(profile_table['flux'], dtype='<f8'),
                                     'flux_err': np.array(profile_table['flux_err'], dtype='<f8') if 'flux_err' in profile_table.colnames else None,
                                     'extracted_2d': extracted_2d
                                    },
                  'profile_box': {
                                  'center_x': header['CENT_X'],
                                  'center_y': header['CENT_Y'],
                                  'center_ra': header['CENT_RA'],
                                  'center_dec': header['CENT_DEC'],
                                  'sky_pa': header['SKY_PA'],
                                  'detector_pa': header['DET_PA'],
                                  'length_arcsec': header['LENGTH'],
                                  'width_arcsec': header['WIDTH'],
                                  'length_pix': header['LENGT_PX'],
                                  'width_pix': header['WIDTH_PX']
                                 },
                   'fits_info': {
                                 'filename': header['FILENAME'],
                                 'obs_date': header['OBSDATE'],
                                 'filter': header['FILTER'],
                                 'roll_angle': header['ROLL_ANG'],
                                 'pixel_scale': header['PIXSCALE'],
                                 'flux_units': header['BUNIT']
                                },
                 }
    
    return(result)

