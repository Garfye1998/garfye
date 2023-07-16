import numpy as np
from astropy.table import QTable, Table, Column
#import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from astropy.visualization import simple_norm
import os
import scipy.ndimage as ndi
import statmorph_egs as statmorph
#import statmorph_ori
import photutils
from photutils.segmentation import SegmentationImage
from photutils.aperture import CircularAperture
from photutils.aperture import EllipticalAperture
import astropy.io.fits as fits
from astropy.utils import lazyproperty

from astropy.nddata import Cutout2D
#from astropy.cosmology import WMAP9 as cosmo
#import astropy.units as u


def seg_map(image):
    threshold = photutils.detect_threshold(image, 1.2)
    npixels = 10  # minimum number of connected pixels
    segment_map = photutils.detect_sources(image, threshold, npixels)
       
    
    try:
        #segm = photutils.deblend_sources(image, segm_ds, npixels)
        segm = photutils.deblend_sources(image, segment_map,npixels=10, nlevels=4, contrast=0.25)
        for i in range(0,segm.areas.shape[0]):
            segmap = segm.data == i + 1
            segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
            segmap = segmap_float > 0.35
            aa = int(image.shape[0]/2)
            bb = int(image.shape[1]/2)
            if np.count_nonzero(segmap[aa-2:aa+3,bb-2:bb+3]) > 1:
                seg_map = segmap        
                break
        seg_img = SegmentationImage(seg_map.astype(int))
    
    except: 
        seg_img =0
        
    return seg_img

def segmap_ori(image):
    threshold = photutils.detect_threshold(image, 1.2)
    npixels = 5  # minimum number of connected pixels
    segm = photutils.detect_sources(image, threshold, npixels)
    
        #segm = photutils.deblend_sources(image, segm_ds, npixels)
    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label
    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    seg_map = segmap_float > 0.3
    seg_img = SegmentationImage(seg_map.astype(int))
        
    return seg_img


class mymorph:
    def __init__(self, img_name, image, psf, mask, redshift, mass, sfr,band, type_sph,type_disk,type_irr,gal_sn):
        self.img_name = img_name
        self._image = image        
        self._psf = psf
        self._mask = mask
        self.redshift = redshift
        self.sfr = sfr
        self.mass = mass
        self.band = band
        self.type_sph =type_sph
        self.type_disk =type_disk
        self.type_irr =type_irr
        self.gal_sn = gal_sn
    
    @lazyproperty
    def _segmap(self):
        segmap = seg_map(self._image)            
        return segmap
    
    
    
    @lazyproperty
    def morph(self):
        '''find image in path, and do non-parametric measuring'''
        

        image = self._image    
        segmap = self._segmap               
        psf = self._psf
        mask = np.array(self._mask, dtype= bool)
        #print('%s_%0.1f starts'%(self.img_name,self.redshift))
        try:
            source_morphs = statmorph.source_morphology(image,segmap,mask=mask,gain=1, psf=psf)
            _morph = source_morphs[0]
        
        except:

            _morph = 0
            print('%s_%0.2f is failed'%(self.img_name,self.redshift))
            
                
        return _morph           
    
    
    @lazyproperty
    def save_fig(self):                
        morph = self.morph   
        mask = self._mask
        image = np.where(~mask+2,self._image,0)
        segmap = self._segmap
        font = { 'weight': 'normal', 'size':17}
        
                
        if self.morph == 0 or self.morph.r20 ==-99:
            fig = 0
            
        else:           
            cutout = self.morph._cutout_stamp_maskzeroed
            #fm1 = dict(stretch='log',log_a=10000,min_cut=0.001,max_cut=np.nanmax(image)/4)   
            fm1 = dict(stretch='asinh',min_cut=0.001,max_cut=np.nanmax(image)/4)
            fm2 = dict(stretch='asinh',min_cut=0.001,max_cut=np.nanmax(image)/2)
            aper20 = CircularAperture((morph.xc_asymmetry, morph.yc_asymmetry), morph.r20)
            aper80 = CircularAperture((morph.xc_asymmetry, morph.yc_asymmetry), morph.r80)
            a = morph.rpetro_ellip
            b = a / morph.elongation_asymmetry
            theta = morph.orientation_asymmetry
            aper_p = EllipticalAperture((morph.xc_asymmetry, morph.yc_asymmetry), a,b, theta=theta)
                        
            
            fig = plt.figure(figsize=(15,5))    
            ax = fig.add_subplot(131)
            plt.suptitle("%s_%s  z=%0.3f, m = %0.2f $lg M_\odot$"%(self.img_name, self.band, self.redshift,self.mass),fontsize=19)
            ax.imshow(image, origin='lower', cmap='Greys',norm=simple_norm(image,**fm1))
            aper20.plot(color='white',alpha=0.3)
            aper80.plot(color='red',alpha=0.3)
            aper_p.plot(color='yellow')
            ax.set_title("snp = %0.3f, snr = %0.3f"%(morph.sn_per_pixel,morph.snr2),font,loc='left')
            ax.set_xlabel('R_20 = %0.3f, R_80 = %0.3f'%(morph.r20,morph.r80),font)
            ax = fig.add_subplot(132)
            ax.imshow(segmap, origin='lower', cmap='gray')
            aper_p.plot(color='yellow')
            
            ax.set_xlabel('a: xc = %0.3f, yc = %0.3f'%(morph.xc_asymmetry,morph.yc_asymmetry),font)
            #ax.set_title("Regularized Segmap",font)
            ax = fig.add_subplot(133)
            ax.set_title("SFR = %0.4f"%(self.sfr),font,loc='left')
            ax.imshow(cutout, origin='lower', cmap='Greys',norm=simple_norm(image,**fm2))
            plt.close(fig)
        return fig
            #image_path = "result_image/"

            #if os.path.exists(image_path + "%s_%s.png"%(self.img_name, self.band)):
                #os.remove(image_path + "%s_%s.png"%(self.img_name, self.band))
            #fig.savefig(image_path + "%s_%s.png"%(self.img_name, self.band))
                                                  

    @lazyproperty
    def image_table(self):
        _table = Table(names=('id','band','x','y','redshift','mass','sfr','p_sph','p_disk','p_irr','gal_ser_n_h','Sersic_n','flag_n','snr',\
                       'r20','r50','re','r80','r_petro','ellip','C','A','S','Gini','M20','M20_c'),\
                       dtype=('S','S','f2','f2','f2','f2','f4','f2','f2','f2','f2','f2','I',\
                              'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4'))
                
        morph = self.morph 
        if self.morph == 0 or self.morph.r20 ==-99:
            _table.add_row((str(self.img_name),self.band,'%.3f'%(self.redshift),'%.2f'%(self.mass),self.sfr,\
                            '%.3f'%(self.type_sph),'%.3f'%(self.type_disk),'%.3f'%(self.type_irr),'%.3f'%(self.gal_sn),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))
        else:
            r20 = morph.r20#*0.06*u.arcsec/cosmo.arcsec_per_kpc_proper(self.redshift)/u.kpc #kpc
            r50 = morph.r50#*0.06*u.arcsec/cosmo.arcsec_per_kpc_proper(self.redshift)/u.kpc #kpc
            re = morph.sersic_rhal#f*0.06*u.arcsec/cosmo.arcsec_per_kpc_proper(self.redshift)/u.kpc #kpc
            r80 = morph.r80#*0.06*u.arcsec/cosmo.arcsec_per_kpc_proper(self.redshift)/u.kpc #kpc                    
            r_petro = morph.rpetro_circ #*0.06*u.arcsec/cosmo.arcsec_per_kpc_proper(self.redshift)/u.kpc #kpc

            _table.add_row((str(self.img_name),self.band,,'%.3f'%(self.redshift),'%.2f'%(self.mass),self.sfr,\
                            '%.3f'%(self.type_sph),'%.3f'%(self.type_disk),'%.3f'%(self.type_irr),'%.3f'%(self.gal_sn),\
                            '%.3f'%(morph.sersic_n),'%d'%(morph.flag_sersic),'%.2f'%(morph.snr2),r20,r50,re,r80,r_petro,\
                            morph.ellipticity_asymmetry))

            
        return _table
    
    @lazyproperty
    def worked(self):
        '''Judging whether the code is successful running.'''
        
        if self.morph == 0 or self.morph.r20 ==-99:
            do = 0
            
        else:
            do = 1
            
        return do 
                
    @lazyproperty    
    def Rp5(self):
        R_p = self.morph.rpetro_ellip
        if R_p <=5:
            return 0
        else:
            return 1
