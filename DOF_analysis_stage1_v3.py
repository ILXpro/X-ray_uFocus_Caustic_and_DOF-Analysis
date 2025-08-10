### Stage_1 - for collect and fit data
### IL

### test line profile
#import sys 
#print("Python version") 
#print (sys.version)
#import numpy as np; print(np.__version__)


import os
import sys
import glob
import natsort

import numpy as np
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm

from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
#from matplotlib.colors import LogNorm

from scipy import ndimage
from skimage import data, img_as_float
from skimage.feature import peak_local_max
from skimage.measure import profile_line

from lmfit.models import GaussianModel, LinearModel, ConstantModel

from matplotlib_scalebar.scalebar import ScaleBar



def get_flist(path):

	def last_chars(x):						# return last N charts
		return(x[-15:])
	
	fextension = 'tiff','TIFF','tif','TIF','png','PNG', 'raw'

	flist = os.listdir(path)				# get unsorted file list
	flist = [i for i in flist if i.endswith(fextension)]

	#flist.sort()								# good initial sort but doesnt sort numerically very well 
	#flist = sorted(flist, key = last_chars)		# sort by last N charts
	#flist = sorted(flist)

	flist = natsort.natsorted(flist)

	return flist


def get_roi (image, image_path, ROI):	# open, crop and conver image to 2D array
	im = Image.open(image_path+image)					# open image
	#im.convert('L')									# 
	im = im.crop(ROI)									# crop image by ROI
	im = np.asarray(im)									# conver image to 2D array
	
	#plt.imshow(im, interpolation='none', cmap='jet')	# cmap='viridis'/'nipy_spectral
	#plt.tight_layout()
	#plt.colorbar()
	##plt.axis('off')
	#plt.show()

	return im

def get_polygone (center, lenth, width, direction): # use this function for generate 2D array of polygone points
													# center - (2,) array of center; lenth, width and direction - line profile 'lenth', 'width' and 'direction' == 'h' or 'v'
	if direction == 'h':
		lp_lenth = lenth
		lp_width = width
	elif direction == 'v':
		lp_width = lenth
		lp_lenth = width
	polygone = np.array([[center[0]-lp_width/2, center[0]+lp_width/2, center[0]+lp_width/2, center[0]-lp_width/2],
						[center[1]-lp_lenth/2, center[1]-lp_lenth/2, center[1]+lp_lenth/2, center[1]+lp_lenth/2]])
	return polygone

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

## new fun
def get_fp(im_roi, lp_lenth, lp_num, fine):
	coordinates = peak_local_max(im_roi, num_peaks = 1)[0, :].T
	coordinates = np.array([coordinates[1], coordinates[0]])
	#print (coordinates[0], coordinates[1], 'peak_local_max')

	for n in range(fine):
		### !new version of hlp, vlp, xlp
		# Make a line with "num" points...
		hx, hy = np.linspace(coordinates[0]-lp_lenth/2, coordinates[0]+lp_lenth/2, lp_num), np.linspace(coordinates[1], coordinates[1], lp_num)
		vx, vy = np.linspace(coordinates[0], coordinates[0], lp_num), np.linspace(coordinates[1]-lp_lenth/2, coordinates[1]+lp_lenth/2, lp_num)
		
		# Extract the values along the line, using cubic interpolation
		# profile intensity
		hlp = ndimage.map_coordinates(np.transpose(im_roi), np.vstack((hx, hy))) # THIS SEEMS TO WORK CORRECTLY
		vlp = ndimage.map_coordinates(np.transpose(im_roi), np.vstack((vx, vy))) # THIS SEEMS TO WORK CORRECTLY
		xlp = np.linspace(-lp_lenth/2, lp_lenth/2, len(hlp))
		#print ('xlp', len(xlp), xlp)

		# FWHM
		fwhm_h = get_fit(hlp, xlp)[1]
		fwhm_h_err = get_fit(hlp, xlp)[2]
		fwhm_v = get_fit(vlp, xlp)[1]
		fwhm_v_err = get_fit(vlp, xlp)[2]
		#print (fwhm_h, fwhm_v, 'FWHM')
		
		coordinates = np.array([coordinates[0] + get_fit(hlp, xlp)[0], coordinates[1] + get_fit(vlp, xlp)[0]])
		#print (coordinates[0], coordinates[1], 'fitted coord, loop num.=', n)

		# plt.title('Image fit iteration = %s'%n)
		# plt.plot(xlp*pixel_size, hlp, '.', label='hor. line prof., exp.data')
		# plt.plot(xlp*pixel_size, vlp, '.', label='vrt. line prof., exp.data')
		# plt.plot(xlp*pixel_size, get_fit(hlp, xlp)[3], label='hor. line prof., fitted')
		# plt.plot(xlp*pixel_size, get_fit(vlp, xlp)[3], label='vrt. line prof., fitted')
		# plt.xlabel('distance, um')
		# plt.ylabel('intensity, a.u.')
		# plt.legend(loc=1, prop={'size': 6})
		# plt.grid(True)
		# #plt.show()
		# plt.show(block=False)
		# plt.pause(.1)
		# plt.close()

	return coordinates, fwhm_h, fwhm_v, fwhm_h_err, fwhm_v_err, hlp, vlp, xlp 

# # old fun
# # def get_cfp(im_roi, lp_lenth, lp_width, coordinates=None):	# get coarse focus position
	
# 	###image_max = ndi.maximum_filter(im_roi, size=3, mode='constant')
# 	if coordinates == None:
# 		coordinates = peak_local_max(im_roi, num_peaks = 1)[0, :]		# num_peaks = 1 or min_distance=xxx /// or use "numpy.argmax"
# 	#else: pass

# 	##print ('{0}	-> {1} precentered focus position'.format(flist[i], coordinates))

# 	hlp = profile_line(im_roi, (coordinates[0], coordinates[1]-lp_lenth/2), (coordinates[0], coordinates[1]+lp_lenth/2), linewidth=lp_width, order=0)
# 	vlp = profile_line(im_roi, (coordinates[0]-lp_lenth/2, coordinates[1]), (coordinates[0]+lp_lenth/2, coordinates[1]), linewidth=lp_width, order=0)
# 	xlp = np.arange(len(hlp))-(len(hlp)-1)/2
# 	#print (hlp.shape, vlp.shape, xlp.shape)

# 	coord_new_x = get_fit(hlp)[0]
# 	coord_new_y = get_fit(vlp)[0]
# 	##print ('{0}	-> {1} precentered focus position'.format(flist[i], coordinates))

# 	fwhm_h = get_fit(hlp)[1]
# 	fwhm_h_err = get_fit(hlp)[2]
# 	fwhm_v = get_fit(vlp)[1]
# 	fwhm_v_err = get_fit(vlp)[2]


# 	#plt.imshow(im_roi)
# 	#plt.fill(get_polygone(coordinates, lp_lenth, lp_width, 'h')[1, :], get_polygone(coordinates, lp_lenth, lp_width, 'h')[0, :], color='#1f77b4ff', alpha=0.3, label='FWHM hor. line profile, pix - %.1f'%(fwhm_h)) # edgecolor='orange')
# 	#plt.fill(get_polygone(coordinates, lp_lenth, lp_width, 'v')[1, :], get_polygone(coordinates, lp_lenth, lp_width, 'v')[0, :], color='#ff7f0eff', alpha=0.3, label='FWHM ver. line profile, pix - %.1f'%(fwhm_v)) # edgecolor='orange')
# 	#plt.plot(coordinates[1], coordinates[0], 'r.', label='prefoc. posit., pix - (%.1f, %.1f)'%(coordinates[1], coordinates[0]))
# 	#plt.legend(loc=1, prop={'size': 8})
# 	#plt.show()

# 	#plt.plot(np.arange(len(hlp))-lp_lenth/2, hlp, 'x', color='#1f77b4ff', label='hor. line profile')
# 	#plt.plot(np.arange(len(hlp))-lp_lenth/2, get_fit(hlp)[3], color='#1f77b4ff', label='hor. line profile fit (FWHM, pix - %.1f)'%(fwhm_h))
# 	#plt.plot(np.arange(len(vlp))-lp_lenth/2, vlp, 'x', color='#ff7f0eff',label='ver. line profile')
# 	#plt.plot(np.arange(len(vlp))-lp_lenth/2, get_fit(vlp)[3], color='#ff7f0eff', label='ver. line profile fit (FWHM, pix - %.1f)'%(fwhm_v))
# 	#plt.xlabel('distance, pix')
# 	#plt.ylabel('intensity, a.u.')
# 	#plt.legend(loc=1, prop={'size': 6})
# 	#plt.show()

# 	coord_new_x = get_fit(hlp)[0]
# 	coord_new_y = get_fit(vlp)[0]
# 	##print ('{0}	-> {1} precentered focus position'.format(flist[i], coordinates))
# 	##coordinates = np.array([coordinates[0]+coord_new_y, coordinates[1]+coord_new_x]) # new coordinates 
# 	coordinates = [coordinates[0]+coord_new_y, coordinates[1]+coord_new_x] # new coordinates 
	
# 	return coordinates, fwhm_h, fwhm_v, fwhm_h_err, fwhm_v_err, hlp, vlp, xlp
# # old fun
# def get_ffp(coordinates, N):			# get tuned focus position and size
# 	coordinates = coordinates
	
# 	if N < 1:
# 		ffp = cfp
# 		pass
# 	else:
# 		for n in range(N):
# 			ffp = get_cfp(im_roi, lp_lenth, lp_width, coordinates)
# 			coordinates = ffp[0]
	
# 	while len(ffp[5]) != len(ffp[6]) or len(ffp[5]) != len(ffp[7]) or len(ffp[6]) != len(ffp[7]) or len(ffp[5]) != len(range(lp_lenth+1)):
# 		ffp = get_cfp(im_roi, lp_lenth, lp_width, coordinates)		# some problems whith not equal len(hlp/vlp/xlp)
# 		coordinates = ffp[0]
	
# 	return ffp

# def get_focus_analysis():	# get fine focus position
	
# 	return

### input par

#image_path = 'focus12/'
#image_path = 'Test/NPD R4 E15keV/'

#trash_path = '_'+image_path
#pixel_size = 0.1625				# in um!
#ROI = (0, 0, 1000, 1000)	# ROI
#fine = 5						# fit loops = 10 is best fit, but it crashs. 5 - is good!

# line profile settings
#lp_lenth = 90					# line profile, pix
#lp_lenth = 180					# line profile, pix
#lp_width = 1					# in pix, not realized
#lp_num = lp_lenth*1				# line profile points number 

#start	= 0.225					# start position in m!
#start	= 0.334					# start position in m!
#step	= 0.01					# step in m
#average = 1						# number of images in same position

#log = 'on'						# on/off save logs through "sys.stdout"

### start from here
if not os.path.exists(trash_path):       # check and ...
	os.makedirs(trash_path)              # create folder for trash


if log == 'on':
	orig_stdout = sys.stdout
	f = open('%s_output_results.txt'%(trash_path), 'w')
	sys.stdout = f
#else: pass

flist = get_flist(image_path)
print ('{0} files found in folder "{1}"\n'.format(len(flist), image_path))
#print (flist)

flist_plus = []
lp = np.zeros((len(flist), 3, lp_num), dtype='float32')
#print (lp.shape)

for i in range(len(flist)):
#for i in range(2):
	#print (i)

	im_roi = get_roi(flist[i], image_path, ROI)

	#cfp = get_cfp(im_roi, lp_lenth, lp_width)
	#ffp = get_ffp(cfp[0], 3)
	ffp = get_fp(im_roi, lp_lenth, lp_num, fine)

	coordinates = ffp[0]
	coordinates_x = coordinates[1]
	coordinates_y = coordinates[0]
	fwhm_h = ffp[1]*pixel_size
	fwhm_h_err = ffp[3]*pixel_size
	fwhm_v = ffp[2]*pixel_size
	fwhm_v_err = ffp[4]*pixel_size

	hlp_h = np.array(ffp[5])
	vlp_v = np.array(ffp[6])
	xlp_x = np.array(ffp[7]*pixel_size)
	#print (hlp_h.shape, vlp_v.shape, xlp_x.shape)
	#print (flist[i])


	if not os.path.exists(trash_path):       # check and ...
		os.makedirs(trash_path)              # create folder for trash

	plt.imshow(im_roi)
	plt.fill(get_polygone(coordinates, lp_lenth, lp_width, 'v')[0, :], get_polygone(coordinates, lp_lenth, lp_width, 'v')[1, :], color='#1f77b4ff', alpha=0.3, label='hor. line profile (FWHM) - %.3f +/- %.3f um'%(fwhm_h, fwhm_h_err), linewidth=0) # edgecolor='orange')
	plt.fill(get_polygone(coordinates, lp_lenth, lp_width, 'h')[0, :], get_polygone(coordinates, lp_lenth, lp_width, 'h')[1, :], color='#ff7f0eff', alpha=0.3, label='ver. line profile (FWHM) - %.3f +/- %.3f um'%(fwhm_v, fwhm_v_err), linewidth=0) # edgecolor='orange')
	#plt.plot(coordinates[1], coordinates[0], 'r.', ms=.5, label='prefoc. posit., pix - (%.2f, %.2f)'%(coordinates[1], coordinates[0]))
	plt.legend(loc=1, prop={'size': 8})
	plt.savefig('{0}{1}.png'.format(trash_path, flist[i]), dpi=1000, bbox_inches = 'tight')
	#plt.show()
	plt.close()	# use 'plt.close()' for close and not use 'plt.show()'

	print ('{0}	->	{1}/{2} analyzed:'.format(flist[i], i+1, len(flist)))
	print ('focus position, pix.:		(%.3f, %.3f)'%(coordinates[0], coordinates[1]))
	print ('focus size (FWHM), um:		(%.3f+/-%.3f, %.3f+/-%.3f)'%(fwhm_h, fwhm_h_err, fwhm_v, fwhm_v_err))
	print ('"{0}{1}.png"	- was saved as result!\n'.format(trash_path, flist[i]))

	flist_plus.append([flist[i], fwhm_h, fwhm_v, fwhm_h_err, fwhm_v_err, coordinates_x, coordinates_y])
	#flist_plus = np.append(flist_plus, np.array([flist[i], fwhm_h, fwhm_v]))
	

	lp_prep = np.array([hlp_h, vlp_v, xlp_x], dtype='float32')
	lp[i,:,:] = lp_prep
	#print (lp[i,:,:])
	#print (lp_prep.shape)
	
flist_plus = np.array(flist_plus)
#print (flist_plus[:,1])
#print (flist_plus[:,0].shape)


#print (lp.shape)
caustic_h = np.array(lp[:,0,:].T)
caustic_v = np.array(lp[:,1,:].T)
caustic_x = np.array(lp[:,2,:].T)

### save it!
np.save("%sdata_flist_plus"%trash_path, flist_plus)
np.save("%sdata_lp"%trash_path, lp)
#np.savetxt("%s_flist_plus.csv"%trash_path, flist_plus, delimiter=",")
#np.savetxt("%s_caustic_h.csv"%trash_path, caustic_h, delimiter=",")
#np.savetxt("%s_caustic_v.csv"%trash_path, caustic_v, delimiter=",")
#np.savetxt("%s_caustic_x.csv"%trash_path, caustic_x, delimiter=",")


yticks = lp[0,2,:]

x = np.arange(len(flist_plus[:,0]))
fwhm_h = np.array(flist_plus[:,1], dtype='f')
fwhm_v = np.array(flist_plus[:,2], dtype='f')
xticks = np.array(flist_plus[:,0])

flist_l2 = np.loadtxt("%sflist_l2.txt"%image_path, dtype=str)
l2 = []
for i in range(len(flist_plus[:,0])):
	l2_index = np.where(flist_l2[:,0] == flist_plus[i,0])[0]
	l2.append(float(flist_l2[l2_index[0],1]))
np.save("%sdata_l2"%trash_path, l2)
np.savetxt("%s_data_l2.csv"%trash_path, l2, delimiter=",")


plt.plot(x, fwhm_h, label='hor. focus size, um')
plt.plot(x, fwhm_v, label='ver. focus size, um')
plt.xticks(x, xticks, rotation=90, fontsize=2)
plt.legend(loc=1, prop={'size': 10})
#plt.gca().invert_yaxis()
plt.xlabel('File names')
plt.ylabel('Focus size, um')
plt.title('DOF results ("%s")'%(image_path))
plt.grid(True)
plt.tight_layout()
plt.savefig('%s_DOF_results_of_%sim_lp%sx%spix.png'%(trash_path, len(flist), lp_lenth, lp_width), dpi=500)
#plt.show()
plt.close()	# use 'plt.close()' for close and not use 'plt.show()'

plt.plot(l2, fwhm_h, label='hor. focus size, um')
plt.plot(l2, fwhm_v, label='ver. focus size, um')
#plt.xticks(x, xticks, rotation=90, fontsize=2)
plt.legend(loc=1, prop={'size': 10})
#plt.gca().invert_yaxis()
plt.xlabel('$L_2$ distance, m')
plt.ylabel('Focus size, um')
plt.title('DOF results ("%s")'%(image_path))
plt.grid(True)
plt.tight_layout()
plt.savefig('%s_DOF_results_L2_of_%sim_lp%sx%spix.png'%(trash_path, len(flist), lp_lenth, lp_width), dpi=500)
#plt.show()
plt.close()	# use 'plt.close()' for close and not use 'plt.show()'


plt.plot(x, fwhm_h, label='hor. focus size, um')
plt.plot(x, fwhm_v, label='ver. focus size, um')
plt.yscale('log')
plt.xticks(x, xticks, rotation=90, fontsize=2)
plt.legend(loc=1, prop={'size': 10})
#plt.gca().invert_yaxis()
plt.xlabel('File names')
plt.ylabel('Focus size, um')
plt.title('DOF results log ("%s")'%(image_path))
plt.grid(True)
plt.tight_layout()
plt.savefig('%s_DOF_results_log_of_%sim_lp%sx%spix.png'%(trash_path, len(flist), lp_lenth, lp_width), dpi=500)
#plt.show()
plt.close()	# use 'plt.close()' for close and not use 'plt.show()'


print ('"%s_DOF_results_of_%sim_lp%sx%spix.png"	->	DOF summary results was saved!\n'%(trash_path, len(flist), lp_lenth, lp_width))
####################


#plt.imshow(caustic_h)
plt.imshow(caustic_h)#, interpolation='bicubic')
###plt.xticks(range(len(caustic_h[0,:])), xticks, rotation=90, fontsize=2)
###plt.yticks(range(len(caustic_h[:,0])), yticks, fontsize=2)
plt.xticks(list(range(len(xticks))), xticks, rotation=90, fontsize=2)
plt.yticks(list(range(len(yticks))), yticks, fontsize=2)
plt.xlabel('File names')
plt.ylabel('caustic_h size, um')
plt.colorbar()
plt.tight_layout()
plt.savefig('%s_caustic_h_results_of%sim_lp%sx%spix.png'%(trash_path, len(flist), lp_lenth, lp_width), dpi=500)
#plt.show()
plt.close()

#plt.imshow(caustic_v)
plt.imshow(caustic_v)#, interpolation='bicubic')
###plt.xticks(range(len(caustic_v[0,:])), xticks, rotation=90, fontsize=2)
###plt.yticks(range(len(caustic_v[:,0])), yticks, fontsize=2)
plt.xticks(list(range(len(xticks))), xticks, rotation=90, fontsize=2)
plt.yticks(list(range(len(yticks))), yticks, fontsize=2)
plt.xlabel('File names')
plt.ylabel('caustic_v size, um')
plt.colorbar()
plt.tight_layout()
plt.savefig('%s_caustic_v_results_of%sim_lp%sx%spix.png'%(trash_path, len(flist), lp_lenth, lp_width), dpi=500)
#plt.show()
plt.close()

### statistics 
fwhm_h_min_i = np.argmin(flist_plus[:,1].astype(dtype = "float64"))
fwhm_v_min_i = np.argmin(flist_plus[:,2].astype(dtype = "float64"))
#print (fwhm_h_min_i, fwhm_v_min_i)

print ('Minimun focus size (FWHM)')
print ('%s:	%.3f+/-%.3f um	->	min. hor. size'%(flist_plus[fwhm_h_min_i, 0], flist_plus[fwhm_h_min_i, 1].astype(np.float64), flist_plus[fwhm_h_min_i, 3].astype(np.float64)))
L2h	= (fwhm_h_min_i//average)*step+start 
print ('L2 = %.3fm'%(L2h))
print ('%s:	%.3f+/-%.3f um	->	min. ver. size'%(flist_plus[fwhm_v_min_i, 0], flist_plus[fwhm_v_min_i, 2].astype(np.float64), flist_plus[fwhm_v_min_i, 4].astype(np.float64)))
L2v	= (fwhm_v_min_i//average)*step+start 
print ('L2 = %.3fm'%(L2v))


### out

## min in hor. direction
ROI_delta_y = round(float(flist_plus[fwhm_h_min_i, 6]) - (ROI[2]-ROI[0])/2)
ROI_delta_x = round(float(flist_plus[fwhm_h_min_i, 5]) - (ROI[3]-ROI[1])/2)
#print (ROI_delta_y, ROI_delta_x)
ROI_h = [ROI[0]+ROI_delta_y, ROI[1]+ROI_delta_x, ROI[2]+ROI_delta_y, ROI[3]+ROI_delta_x]		# new ROI with offset
#print (ROI_h)

im_roi = get_roi(flist[fwhm_h_min_i], image_path, ROI_h)

plt.imshow(im_roi, norm=PowerNorm(gamma=1. / 2.))#, norm=LogNorm())# cmap='gray', vmin=np.amin(im_roi), vmax=np.amax(im_roi))
#plt.figure(frameon=False)
plt.axis('off')
plt.colorbar(shrink = 0.89)
scalebar = ScaleBar(pixel_size, 'um', location='lower right', fixed_value=20, border_pad=.5)
plt.gca().add_artist(scalebar)
plt.savefig('{0}_{1}_hor_min.png'.format(trash_path, flist_plus[fwhm_h_min_i, 0]), dpi=1000, bbox_inches = 'tight', pad_inches = 0)
#plt.show()
plt.close()	# use 'plt.close()' for close and not use 'plt.show()'


plt.plot(caustic_x[:,fwhm_h_min_i], caustic_h[:,fwhm_h_min_i], 'x', color='#1f77b4ff', label='hor. line profile (exp.)')
plt.plot(caustic_x[:,fwhm_h_min_i], get_fit(caustic_h[:,fwhm_h_min_i], caustic_x[:,fwhm_h_min_i])[3], color='#1f77b4ff', label='hor. line profile (fit) - %.2f+/-%.2f um'%(flist_plus[fwhm_h_min_i, 1].astype(np.float64), flist_plus[fwhm_h_min_i, 3].astype(np.float64)))
plt.plot(caustic_x[:,fwhm_h_min_i], caustic_v[:,fwhm_h_min_i], 'x', color='#ff7f0eff', label='ver. line profile (exp.)')
plt.plot(caustic_x[:,fwhm_h_min_i], get_fit(caustic_v[:,fwhm_h_min_i], caustic_x[:,fwhm_h_min_i])[3], color='#ff7f0eff', label='ver. line profile (fit) - %.2f+/-%.2f um'%(flist_plus[fwhm_h_min_i, 2].astype(np.float64), flist_plus[fwhm_h_min_i, 4].astype(np.float64)))
plt.xlabel('distance, um')
plt.ylabel('intensity, a.u.')
plt.legend(loc=1, prop={'size': 6})
plt.grid(True)
plt.savefig('{0}_{1}_gauss_fit_h_hor_and_ver.lp.png'.format(trash_path, flist_plus[fwhm_h_min_i, 0]), dpi=1000, bbox_inches = 'tight')
#plt.show()
plt.close()	# use 'plt.close()' for close and not use 'plt.show()'


## min in ver. direction
ROI_delta_y = round(float(flist_plus[fwhm_v_min_i, 6]) - (ROI[2]-ROI[0])/2)
ROI_delta_x = round(float(flist_plus[fwhm_v_min_i, 5]) - (ROI[3]-ROI[1])/2)
#print (ROI_delta_y, ROI_delta_x)
ROI_v = [ROI[0]+ROI_delta_y, ROI[1]+ROI_delta_x, ROI[2]+ROI_delta_y, ROI[3]+ROI_delta_x]		# new ROI with offset
#print (ROI_h)

im_roi = get_roi(flist[fwhm_v_min_i], image_path, ROI_v)
plt.imshow(im_roi, norm=PowerNorm(gamma=1. / 2.))#, norm=LogNorm())# cmap='gray', vmin=np.amin(im_roi), vmax=np.amax(im_roi))
#plt.figure(frameon=False)
plt.axis('off')
plt.colorbar(shrink = 0.89)
scalebar = ScaleBar(pixel_size, 'um', location='lower right', fixed_value=20, border_pad=.5)
plt.gca().add_artist(scalebar)
plt.savefig('{0}_{1}_ver_min.png'.format(trash_path, flist_plus[fwhm_v_min_i, 0]), dpi=1000, bbox_inches = 'tight', pad_inches = 0)
#plt.show()
plt.close()	# use 'plt.close()' for close and not use 'plt.show()'


plt.plot(caustic_x[:,fwhm_v_min_i], caustic_h[:,fwhm_v_min_i], 'x', color='#1f77b4ff', label='hor. line profile (exp.)')
plt.plot(caustic_x[:,fwhm_v_min_i], get_fit(caustic_h[:,fwhm_v_min_i], caustic_x[:,fwhm_v_min_i])[3], color='#1f77b4ff', label='hor. line profile (fit) - %.2f+/-%.2f um'%(flist_plus[fwhm_v_min_i, 1].astype(np.float64), flist_plus[fwhm_v_min_i, 3].astype(np.float64)))
plt.plot(caustic_x[:,fwhm_v_min_i], caustic_v[:,fwhm_v_min_i], 'x', color='#ff7f0eff', label='ver. line profile (exp.)')
plt.plot(caustic_x[:,fwhm_v_min_i], get_fit(caustic_v[:,fwhm_v_min_i], caustic_x[:,fwhm_v_min_i])[3], color='#ff7f0eff', label='ver. line profile (fit) - %.2f+/-%.2f um'%(flist_plus[fwhm_v_min_i, 2].astype(np.float64), flist_plus[fwhm_v_min_i, 4].astype(np.float64)))
plt.xlabel('distance, um')
plt.ylabel('intensity, a.u.')
plt.legend(loc=1, prop={'size': 6})
plt.grid(True)
plt.savefig('{0}_{1}_gauss_fit_v_hor_and_ver.lp.png'.format(trash_path, flist_plus[fwhm_v_min_i, 0]), dpi=1000, bbox_inches = 'tight')
#plt.show()
plt.close()	# use 'plt.close()' for close and not use 'plt.show()'


if log == 'on':
	sys.stdout = orig_stdout
	f.close()
#else: pass
 

