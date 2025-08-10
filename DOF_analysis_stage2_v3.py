### Stage2 - for analyze and print results
### IL

import os

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm

from lmfit.models import GaussianModel, LinearModel, ConstantModel

from matplotlib_scalebar.scalebar import ScaleBar

#from DOF_main_and_config import *

def get_fit(y, x):
	
	#mod = GaussianModel() + LinearModel()
	mod = GaussianModel() + ConstantModel()
	#pars = mod.make_params(amplitude=50000, center=0, sigma=3, slope=0, intercept=500)
	pars = mod.make_params(amplitude=1e4, center=0, sigma=1, intercept=.5e3)
	##pars = mod.guess(y, x=x)

	result = mod.fit(y, pars, x=x)
	#print(result.fit_report())

	center = result.params['center'].value
	fwhm = result.params['fwhm'].value
	fwhm_err = result.params['fwhm'].stderr
	#print('center:	"{0}"'.format(center))
	#print('fwhm:	"{0} +/- {1}"'.format(fwhm, fwhm_err))
	
	#plt.plot(x, y, 'b.', label='input data')
	###plt.plot(x, result.init_fit, 'k--', label='input model')
	#plt.plot(x, result.best_fit, 'r-', label='best fit')
	#plt.legend()
	#plt.show()
	#plt.close()

	return center, fwhm, fwhm_err, result.best_fit

def get_pfit (x, y):
	z = np.polyfit(x, y, 7)
	#print (z)
	#print (-z[1]/2/z[2])
	p = np.poly1d(z)
	f = p(x)
	return f

def get_roi2 (image, image_path, ROI):	# open, crop and conver image to 2D array
	im = Image.open(image_path+image)					# open image
	#im.convert('L')									# 

	im_i = np.where(flist_plus[:,0] == '%s'%image)

	ROI_delta_y = round(float(flist_plus[im_i, 6]) + ROI[0])
	ROI_delta_x = round(float(flist_plus[im_i, 5]) + ROI[1])
	#print (ROI_delta_y, ROI_delta_x)
	ROI_2 = [ROI_delta_y - ROI2[0]/2, ROI_delta_x - ROI2[1]/2, ROI_delta_y + ROI2[0]/2, ROI_delta_x + ROI2[1]/2]		# new ROI2 with offset

	im = im.crop(ROI_2)									# crop image by ROI
	im = np.asarray(im)									# conver image to 2D array
	
	plt.imshow(im, interpolation='none', cmap='gray')	# cmap='viridis'/'nipy_spectral
	plt.tight_layout()
	plt.axis('off')
	plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89)
	scalebar = ScaleBar(pixel_size, 'um', location='lower right', color='w', frameon=False, fixed_value=5, border_pad=.5)
	plt.gca().add_artist(scalebar)
	plt.savefig('{0}_{1}_gray.png'.format(trash_path, image), dpi=500, bbox_inches = 'tight', pad_inches = 0)
	###plt.show()
	plt.show(block=False)
	plt.pause(1)
	plt.close()


	plt.imshow(im, interpolation='none')	# cmap='viridis'/'nipy_spectral
	plt.tight_layout()
	plt.axis('off')
	plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89, , format='%.0E')
	scalebar = ScaleBar(pixel_size, 'um', location='lower right', color='w', frameon=False, fixed_value=5, border_pad=.5)
	#scalebar = ScaleBar(pixel_size, 'um', location='lower right', color='w', frameon=False, fixed_value=5, border_pad=.5, font_properties={'size' : 15}) # font size tuning
	plt.gca().add_artist(scalebar)
	plt.savefig('{0}_{1}_col.png'.format(trash_path, image), dpi=500, bbox_inches = 'tight', pad_inches = 0)
	###plt.show()
	plt.show(block=False)
	plt.pause(1)
	plt.close()

	plt.imshow(im, interpolation='none')	# cmap='viridis'/'nipy_spectral
	plt.tight_layout()
	plt.axis('off')
	plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89, , format='%.0E')
	#scalebar = ScaleBar(pixel_size, 'um', location='lower right', color='w', frameon=False, fixed_value=5, border_pad=.5)
	#scalebar = ScaleBar(pixel_size, 'um', location='lower right', color='w', frameon=False, fixed_value=5, border_pad=.5, font_properties={'size' : 15}) # font size tuning
	#plt.gca().add_artist(scalebar)
	plt.savefig('{0}_{1}_col-scale.png'.format(trash_path, image), dpi=500, bbox_inches = 'tight', pad_inches = 0)
	###plt.show()
	plt.show(block=False)
	plt.pause(1)
	plt.close()

	# plt.imshow(im, interpolation='none', cmap='gray', norm=PowerNorm(gamma=1. / 2.))	# cmap='viridis'/'nipy_spectral
	# plt.tight_layout()
	# plt.axis('off')
	# plt.colorbar(pad=0.01)#shrink = 0.89, , format='%.0E')
	# scalebar = ScaleBar(pixel_size, 'um', location='lower right', color='w', frameon=False, fixed_value=5, border_pad=.5)
	# plt.gca().add_artist(scalebar)
	# plt.savefig('{0}_{1}_gray_norm-pow.png'.format(trash_path, image), dpi=500, bbox_inches = 'tight', pad_inches = 0)
	# ###plt.show()
	# plt.show(block=False)
	# plt.pause(1)
	# plt.close()

	plt.plot(caustic_x[:,im_i][:,0,0], caustic_h[:,im_i][:,0,0]/1e3, '.', color='#1f77b4ff')#, label='hor. line profile (exp.)')
	plt.plot(caustic_x[:,im_i][:,0,0], get_fit(caustic_h[:,im_i][:,0,0], caustic_x[:,im_i][:,0,0])[3]/1e3, color='#1f77b4ff', label='hor. FWHM - %.1f±%.1fμm'%(flist_plus[im_i, 1].astype(np.float64), fit_error))
	plt.plot(caustic_x[:,im_i][:,0,0], caustic_v[:,im_i][:,0,0]/1e3, '.', color='#ff7f0eff')#, label='ver. line profile (exp.)')
	plt.plot(caustic_x[:,im_i][:,0,0], get_fit(caustic_v[:,im_i][:,0,0], caustic_x[:,im_i][:,0,0])[3]/1e3, color='#ff7f0eff', label='ver. FWHM - %.1f±%.1fμm'%(flist_plus[im_i, 2].astype(np.float64), fit_error))
	plt.xlabel('transverse profile, μm', fontsize=12)
	plt.ylabel('intensity, a.u. (x$10^3$)', fontsize=12)
	plt.legend(loc=1, prop={'size': 10})
	plt.grid(True)
	plt.savefig('{0}_{1}_gauss_fit_min_hor_and_ver.lp.png'.format(trash_path, image), dpi=500, bbox_inches = 'tight')
	plt.show(block=False)
	plt.pause(1)
	plt.close()



	return print('final images was saved!\n')


### input par
###image_path = 'focus12/'
#image_path = 'focus/'
###trash_path = '_'+image_path
###pixel_size = 0.325				# in um!
###ROI = (700, 1050, 1300, 1550)	# ROI
###fine = 5						# fit loops = 10 is best fit
###ROI2 = [200, 200]				# ROI2 size for print image

# line profile settings
###lp_lenth = 180					# line profile, pix
###lp_width = 1					# in pix, not realized
###lp_num = lp_lenth*1				# line profile points number 
###fit_error = 0.2					# in um

###L2start	= 0.225					# start position in m!
#L2start	= 0.334					# start position in m!
###L2step	= 0.01					# step in m
###average = 1						# number of images in same position

### start from here
flist_plus	= np.load("%sdata_flist_plus.npy"%trash_path)
#flist_plus[:, 1:]	= flist_plus[:, 1:].astype(dtype = "float64")
lp			= np.load("%sdata_lp.npy"%trash_path)
l2			= np.load("%sdata_l2.npy"%trash_path)
#print (flist_plus)
np.savetxt("%s_data_flist_plus.csv"%trash_path, flist_plus, delimiter=",", fmt='%s')
#np.savetxt("%s_data_lp.csv"%trash_path, lp, delimiter=",", fmt='%s')


####

####

fwhm_h_min_i = np.argmin(flist_plus[:,1].astype(dtype = "float64"))
fwhm_v_min_i = np.argmin(flist_plus[:,2].astype(dtype = "float64"))
#print (fwhm_h_min_i, fwhm_v_min_i)

caustic_h = np.array(lp[:,0,:].T)
caustic_v = np.array(lp[:,1,:].T)
caustic_x = np.array(lp[:,2,:].T)

np.savetxt("%s_data_caustic_h.csv"%trash_path, caustic_h, delimiter=",", fmt='%s')
np.savetxt("%s_data_caustic_v.csv"%trash_path, caustic_v, delimiter=",", fmt='%s')
np.savetxt("%s_data_caustic_x.csv"%trash_path, caustic_x, delimiter=",", fmt='%s')


print ('Minimun focus size (FWHM)')
### Horizontal
print ('%s:	%.3f+/-%.3f um	->	min. hor. size'%(flist_plus[fwhm_h_min_i, 0], flist_plus[fwhm_h_min_i, 1].astype(np.float64), flist_plus[fwhm_h_min_i, 3].astype(np.float64)))
#L2h	= (fwhm_h_min_i//average)*L2step+L2start 
L2h	= l2[fwhm_h_min_i]
print ('L2 = %.3fm'%(L2h))
get_roi2(flist_plus[fwhm_h_min_i, 0], image_path, ROI)

### Vertical 
print ('%s:	%.3f+/-%.3f um	->	min. ver. size'%(flist_plus[fwhm_v_min_i, 0], flist_plus[fwhm_v_min_i, 2].astype(np.float64), flist_plus[fwhm_v_min_i, 4].astype(np.float64)))
#L2v	= (fwhm_v_min_i//average)*L2step+L2start 
L2v	= l2[fwhm_v_min_i]
print ('L2 = %.3fm'%(L2v))
get_roi2(flist_plus[fwhm_v_min_i, 0], image_path, ROI)

### find min (auto)

dof = []

caustic_h_min = []
caustic_v_min = []
caustic_x_min = []


for i in range(len(flist_plus[:,0])):
###	if (i+1)%average==0:
###		start = int(i-2)
###		stop = int(i)

		#print ("Hello")

###		fwhm_v_min_i = np.argmin(flist_plus[start:stop,2]) + start
		#fwhm_v_min_i = np.argmin(flist_plus[i,2].astype(dtype = "float64"))
		fwhm_v_min_i = i

		best_fhwm_h = float(flist_plus[fwhm_v_min_i, 1])
		best_fhwm_v = float(flist_plus[fwhm_v_min_i, 2])
		best_fwhm_x = l2[fwhm_v_min_i]
		best_fwhm = [best_fhwm_h, best_fhwm_v, best_fwhm_x]
		dof.append(best_fwhm)

		caustic_h_min.append(caustic_h[:, fwhm_v_min_i])
		caustic_v_min.append(caustic_v[:, fwhm_v_min_i])
		caustic_x_min.append(caustic_x[:, fwhm_v_min_i])


dof = np.array(dof).T
np.save("%sdata_dof"%trash_path, dof)
np.savetxt("%s_data_dof.csv"%trash_path, dof.T, delimiter=",")

caustic_h_min = np.array(caustic_h_min).T
caustic_v_min = np.array(caustic_v_min).T
caustic_x_min = np.array(caustic_x_min).T


# if image_path == 'focus/':
# 	dof = dof[:, 0:-3]
# 	caustic_h_min = caustic_h_min[:, 0:-3]
# 	caustic_v_min = caustic_v_min[:, 0:-3]
# 	caustic_x_min = caustic_x_min[:, 0:-3]
# else: pass


sorted_index = l2.argsort()
# print (sorted_index)
dof[0] = np.take_along_axis(dof[0], sorted_index, axis = 0)
dof[1] = np.take_along_axis(dof[1], sorted_index, axis = 0)
dof[2] = np.take_along_axis(dof[2], sorted_index, axis = 0)

# print (sorted_index.shape)
# print (dof[0].shape)
# print (caustic_h_min.shape)

sorted_index_2d = np.ones(caustic_h_min.shape)
sorted_index_2d = sorted_index_2d * sorted_index
sorted_index_2d = sorted_index_2d.astype(int)

# print (type(sorted_index_2d[0, 0]))
# print (sorted_index_2d.shape)
# print (sorted_index_2d)


caustic_h_min = np.take_along_axis(caustic_h_min, sorted_index_2d, axis = 1)
caustic_v_min = np.take_along_axis(caustic_v_min, sorted_index_2d, axis = 1)
#caustic_x_min = np.take_along_axis(caustic_x_min, sorted_index_2d, axis = 0)


plt.plot(dof[2], dof[0], '.', color='#1f77b4ff')
plt.plot(dof[2], get_pfit(dof[2], dof[0]), color='#1f77b4ff', label='hor. beam size (FWHM)')
plt.plot(dof[2], dof[1], '.', color='#ff7f0eff')
plt.plot(dof[2], get_pfit(dof[2], dof[1]), color='#ff7f0eff', label='ver. beam size (FWHM)')
plt.xlabel('$L_2$ distance, m', fontsize=12)
plt.ylabel('beam size (FWHM), μm', fontsize=12)
plt.legend(loc=1, prop={'size': 10})
plt.grid(True)
plt.savefig('{0}_DOF_min_hor_and_ver.lp.png'.format(trash_path), dpi=500, bbox_inches = 'tight')
plt.show(block=False)
plt.pause(1)
plt.close()

plt.imshow(caustic_h_min, extent=[dof[2][0],dof[2][-1], caustic_x_min[0, 0], caustic_x_min[-1, 0]], aspect='auto')#, interpolation='bicubic')
plt.xlabel('$L_2$ distance, m', fontsize=12)
plt.ylabel('transverse profile, μm', fontsize=12)
plt.legend(title='horizontal caustic', loc=1, prop={'size': 10}, frameon=False).get_title().set_color("white")
plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89, , format='%.0E')#shrink = 0.89)
plt.savefig('{0}_caustic_min_h.lp.png'.format(trash_path), dpi=500, bbox_inches = 'tight')
plt.show(block=False)
plt.pause(1)
plt.close()

plt.imshow(caustic_v_min, extent=[dof[2][0],dof[2][-1], caustic_x_min[0, 0], caustic_x_min[-1, 0]], aspect='auto')#, interpolation='bicubic')
plt.xlabel('$L_2$ distance, m', fontsize=12)
plt.ylabel('transverse profile, μm', fontsize=12)
plt.legend(title='vertical caustic', loc=1, prop={'size': 10}, frameon=False).get_title().set_color("white")
plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89, , format='%.0E')#shrink = 0.89)
plt.savefig('{0}_caustic_min_v.lp.png'.format(trash_path), dpi=500, bbox_inches = 'tight')
plt.show(block=False)
plt.pause(1)
plt.close()



plt.subplot(2, 1, 1)
plt.imshow(caustic_h_min, extent=[dof[2][0],dof[2][-1], caustic_x_min[0, 0], caustic_x_min[-1, 0]], aspect='auto')#, interpolation='bicubic')
#plt.xlabel('$L_2$ distance, m', fontsize=12)
plt.ylabel('transverse profile, μm', fontsize=12)
plt.legend(title='horizontal caustic', loc=1, prop={'size': 10}, frameon=False).get_title().set_color("white")
plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89, , format='%.0E')#shrink = 0.89)
plt.subplot(2, 1, 2)
plt.imshow(caustic_v_min, extent=[dof[2][0],dof[2][-1], caustic_x_min[0, 0], caustic_x_min[-1, 0]], aspect='auto')#, interpolation='bicubic')
plt.xlabel('$L_2$ distance, m', fontsize=12)
plt.ylabel('transverse profile, μm', fontsize=12)
plt.legend(title='vertical caustic', loc=1, prop={'size': 10}, frameon=False).get_title().set_color("white")
plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89, , format='%.0E')#shrink = 0.89)
plt.tight_layout()
plt.savefig('{0}_caustic_min_sum.lp.png'.format(trash_path), dpi=500, bbox_inches = 'tight')
plt.show(block=False)
plt.pause(1)
plt.close()

plt.subplot(2, 1, 1)
plt.imshow(caustic_h_min, extent=[dof[2][0],dof[2][-1], caustic_x_min[0, 0], caustic_x_min[-1, 0]], aspect='auto', interpolation='bicubic')
#plt.xlabel('$L_2$ distance, m', fontsize=12)
plt.ylabel('transverse profile, μm', fontsize=12)
plt.legend(title='horizontal caustic', loc=1, prop={'size': 10}, frameon=False).get_title().set_color("white")
plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89, , format='%.0E')#shrink = 0.89)
plt.subplot(2, 1, 2)
plt.imshow(caustic_v_min, extent=[dof[2][0],dof[2][-1], caustic_x_min[0, 0], caustic_x_min[-1, 0]], aspect='auto', interpolation='bicubic')
plt.xlabel('$L_2$ distance, m', fontsize=12)
plt.ylabel('transverse profile, μm', fontsize=12)
plt.legend(title='vertical caustic', loc=1, prop={'size': 10}, frameon=False).get_title().set_color("white")
plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89, , format='%.0E')#shrink = 0.89)
plt.tight_layout()
plt.savefig('{0}_caustic_min_sum_biq.lp.png'.format(trash_path), dpi=500, bbox_inches = 'tight')
plt.show(block=False)
plt.pause(1)
plt.close()

#text deleted
plt.subplot(2, 1, 1)
plt.imshow(caustic_h_min, extent=[dof[2][0],dof[2][-1], caustic_x_min[0, 0], caustic_x_min[-1, 0]], aspect='auto', interpolation='bicubic')
#plt.xlabel('$L_2$ distance, m', fontsize=12)
plt.ylabel('transverse profile, μm', fontsize=12)
#plt.legend(title='horizontal caustic', loc=1, prop={'size': 10}, frameon=False).get_title().set_color("white")
plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89, , format='%.0E')#shrink = 0.89)
plt.subplot(2, 1, 2)
plt.imshow(caustic_v_min, extent=[dof[2][0],dof[2][-1], caustic_x_min[0, 0], caustic_x_min[-1, 0]], aspect='auto', interpolation='bicubic')
plt.xlabel('$L_2$ distance, m', fontsize=12)
plt.ylabel('transverse profile, μm', fontsize=12)
#plt.legend(title='vertical caustic', loc=1, prop={'size': 10}, frameon=False).get_title().set_color("white")
plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89, , format='%.0E')#shrink = 0.89)
plt.tight_layout()
plt.savefig('{0}_caustic_min_sum_biq.lp-text.png'.format(trash_path), dpi=500, bbox_inches = 'tight')
plt.show(block=False)
plt.pause(1)
plt.close()


#print (dof[2][:])
#print(caustic_x_min[:, 0])
#print(caustic_x_min[45, 0])
#print(len(caustic_x_min[:, 0]))

#caustic_x_cen = int(len(caustic_x_min[:, 0]) / 2)
#print (caustic_x_cen)

#print (caustic_h_min[caustic_x_cen - 1, :])
#print (caustic_h_min[caustic_x_cen - 1 : caustic_x_cen + 1, :])
#print (np.average(caustic_h_min[caustic_x_cen - 1 : caustic_x_cen + 1, :], axis=0))

caustic_x_cen = int(len(caustic_x_min[:, 0]) / 2)
caustic_x_aver = 30

plt.subplot(2, 1, 1)
plt.plot(dof[2][:], np.average(caustic_h_min[caustic_x_cen - caustic_x_aver : caustic_x_cen + caustic_x_aver, :], axis=0))
#plt.xlabel('$L_2$ distance, m', fontsize=12)
plt.ylabel('transverse profile, μm', fontsize=12)
plt.legend(title='horizontal caustic', loc=1, prop={'size': 10}, frameon=False).get_title().set_color("white")
#plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89, , format='%.0E')#shrink = 0.89)
plt.subplot(2, 1, 2)
plt.plot(dof[2][:], np.average(caustic_v_min[caustic_x_cen - caustic_x_aver : caustic_x_cen + caustic_x_aver, :], axis=0))
plt.xlabel('$L_2$ distance, m', fontsize=12)
plt.ylabel('transverse profile, μm', fontsize=12)
plt.legend(title='vertical caustic', loc=1, prop={'size': 10}, frameon=False).get_title().set_color("white")
#plt.colorbar(pad=0.01).formatter.set_powerlimits((0, 0))#shrink = 0.89, , format='%.0E')#shrink = 0.89)
plt.tight_layout()
plt.savefig('{0}_caustic_min_sum_biq.lp_.png'.format(trash_path), dpi=500, bbox_inches = 'tight')
plt.show(block=False)
plt.pause(1)
plt.close()

