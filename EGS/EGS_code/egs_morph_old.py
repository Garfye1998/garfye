import numpy as np
from astropy.table import QTable, Table, Column
#import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from astropy.visualization import simple_norm
import os
import scipy.ndimage as ndi
import statmorph#_me as statmorph
import photutils
from photutils.segmentation import SegmentationImage
from photutils.aperture import CircularAperture
import astropy.io.fits as fits
from astropy.utils import lazyproperty
import warnings

        
def open_image(input_image,band):
        
    img_path = "/home/hqzhang/Data_EGS/EGS_tot_v1/science_images/"
    
    image = fits.open(img_path + '%s/%s_%s.fits'%(input_image,input_image,band))[0].data
    
    mask_img = fits.open(img_path + '%s/%s_mask.fits'%(input_image,input_image))[0].data
    
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            if mask_img[i,j] == 1:
                image[i,j] = np.nan
        
    return image   

def seg_map(image):
    threshold = photutils.detect_threshold(image, 1.5)
    npixels = 5  # minimum number of connected pixels
    segm = photutils.detect_sources(image, threshold, npixels)
    
    try:
        #segm = photutils.deblend_sources(image, segm_ds, npixels)
        for i in range(0,segm.areas.shape[0]):
            segmap = segm.data == i + 1
            segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
            segmap = segmap_float > 0.4
            aa = int(image.shape[0]/2)
            bb = int(image.shape[1]/2)
            if np.count_nonzero(segmap[aa-2:aa+3,bb-2:bb+3]) > 1:
                seg_map = segmap        
                break
        seg_img = SegmentationImage(seg_map.astype(int))
    
    except: 
        seg_img =0
        
    return seg_img



def open_psf(band):    
    psf_path = "/home/wangjh/Mywork/EGS/PSF/"
    psf = fits.open(psf_path + "epsf_%s.fits"%(band))[0].data
    return psf
    

    
class mymorph:
    def __init__(self, image_name, redshift, mass):
        self.img_name = image_name
        #self.band = band
        self.redshift = redshift
        self.mass = mass
    
    @lazyproperty
    def image(self):        
        image = open_image(self.img_name,self._band)        
        return image    
    
    @lazyproperty
    def _segmap(self):
        return seg_map(self.image)
    
    @lazyproperty
    def _band(self):
        if self.redshift <= 0.5:
            band = 'v'
            
        elif 0.5 < self.redshift <= 1.0:
            band = 'i'
            
        elif 1.0 < self.redshift <= 2.0:    
            band = 'j'
            
        elif self.redshift > 2.0:
            band = 'h'
        
        return band
           
    @lazyproperty
    def _psf(self):
        psf = open_psf(self._band)               
        return psf   

    @lazyproperty
    def morph(self):
        '''find image in path, and do non-parametric measuring'''

        segmap = self._segmap               
        image = self.image
        psf = self._psf

        try:
            source_morphs = statmorph.source_morphology(image,segmap,gain=1, psf=psf)
            _morph = source_morphs[0]

        except:

            _morph = 0
            print('%s_%s is failed'%(self.img_name,self._band))
                
        return _morph    
    
    @lazyproperty
    def classify_galaxy_by_snp(self):
        snp = self.morph.sn_per_pixel
        
        if snp >= 0.2:
            i = 0
            #print('very good')
            
        if 0.15 <= snp < 0.2:
            i = 1
            #print('not bad')  
            
        if 0.10 <= snp < 0.15:
            i = 2
            #print('just ok') 
            
        if snp < 0.10:
            i = 3
            #print('bad')           

        #print(i)    
        return i 
    
    
    @lazyproperty
    def save_fig(self):     
        morph = self.morph        
        image = self.image
        segmap = self._segmap
        font = { 'weight': 'normal', 'size':17}
                
        if morph.r20 == -99.0:
            pass

        else:           
            fm1 = dict(stretch='asinh',asinh_a=0.01,min_cut=0.001,max_cut=np.nanmax(image)/2)    
            aper20 = CircularAperture((morph.xc_asymmetry, morph.yc_asymmetry), morph.r20)
            aper80 = CircularAperture((morph.xc_asymmetry, morph.yc_asymmetry), morph.r80)
            fig = plt.figure(figsize=(10,5))    
            ax = fig.add_subplot(121)
            plt.suptitle("%s_%s  z=%0.3f, m = %0.2f $lg M_\odot$"%(self.img_name, self._band, self.redshift,self.mass),fontsize=19)
            ax.imshow(image, origin='lower', cmap='Greys',norm=simple_norm(image,**fm1))
            aper20.plot(color='white')
            aper80.plot(color='red')
            ax.set_title("snp = %0.3f"%(morph.sn_per_pixel),font,loc='left')
            ax.set_xlabel('R_20 = %0.3f, R_80 = %0.3f'%(morph.r20,morph.r80),font)
            ax = fig.add_subplot(122)
            ax.imshow(segmap, origin='lower', cmap='gray')
            ax.set_xlabel('a: xc = %0.3f, yc = %0.3f'%(morph.xc_asymmetry,morph.yc_asymmetry),font)
            #ax.set_title("Regularized Segmap",font)

            image_path = "/home/wangjh/Mywork/result_image/"

            if self.classify_galaxy_by_snp == 0: 

                if os.path.exists(image_path + "label_0/%s_%s.png"%(self.img_name, self._band)):
                    os.remove(image_path + "label_0/%s_%s.png"%(self.img_name, self._band))
                fig.savefig(image_path + "label_0/%s_%s.png"%(self.img_name, self._band))


            if self.classify_galaxy_by_snp == 1: 

                if os.path.exists(image_path + "label_1/%s_%s.png"%(self.img_name, self._band)):
                    os.remove(image_path + "label_1/%s_%s.png"%(self.img_name, self._band))
                fig.savefig(image_path + "label_1/%s_%s.png"%(self.img_name, self._band))                

            if self.classify_galaxy_by_snp == 2: 

                if os.path.exists(image_path + "label_2/%s_%s.png"%(self.img_name, self._band)):
                    os.remove(image_path + "label_2/%s_%s.png"%(self.img_name, self._band))
                fig.savefig(image_path + "label_2/%s_%s.png"%(self.img_name, self._band))                

            if self.classify_galaxy_by_snp == 3: 

                if os.path.exists(image_path + "label_3/%s_%s.png"%(self.img_name, self._band)):
                    os.remove(image_path + "label_3/%s_%s.png"%(self.img_name, self._band))
                fig.savefig(image_path + "label_3/%s_%s.png"%(self.img_name, self._band))
    
    
    
    @lazyproperty
    def image_table(self):
        _table = Table(names=('id', 'band','redshift','mass','Sersic_n','r20"','r80"','C','A','S','Gini','M20','F(G, M20)','S(G, M20)'),\
                          dtype=('S','S','f2','f2','f2','f4','f4','f4','f4','f4','f4','f4','f4','f4'))
        
        morph = self.morph 
        #print(morph.flag_sersic)
        if morph.r20 == -99.0:
            _table.add_row(('%s'%(self.img_name),self._band,'%.3f'%(self.redshift),'%.2f'%(self.mass),\
                            None,None,None,None,None,None,None,None,None,None))
        else:            
            r20 = morph.r20 * 0.06 #arcsec 
            r80 = morph.r80 * 0.06 #arcsec 

            if morph.flag_sersic == 0:
                _table.add_row(('%s'%(self.img_name),self._band,'%.3f'%(self.redshift),'%.2f'%(self.mass),morph.sersic_n,r20,r80,\
                                morph.concentration, morph.asymmetry, morph.smoothness,\
                                morph.gini, morph.m20, morph.gini_m20_bulge,morph.gini_m20_merger))
            else:
                _table.add_row(('%s'%(self.img_name),self._band,'%.3f'%(self.redshift),'%.2f'%(self.mass),None,r20,r80,\
                                morph.concentration, morph.asymmetry, morph.smoothness,\
                                morph.gini, morph.m20, morph.gini_m20_bulge,morph.gini_m20_merger))                    
            
        return _table
    
    @lazyproperty
    def work(self):
        '''Judging whether the code is successful running.'''
        
        if self.morph == 0:
            do = 0
            
        else:
            do = 1
            
        return do 
            
            
        