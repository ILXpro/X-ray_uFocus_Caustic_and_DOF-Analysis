### 
### Stage_1 - for collect and fit data
### Stage_2 - for analyze and print results, on development stage 
### by IL


import matplotlib.pyplot as plt



### input par

#image_path = 'Test/SCD R3 E10.3keV/'
#image_path = 'Test/NPD R3 E10.3keV/'
#image_path = 'Test/SCD R3 E15keV/'
#image_path = 'Test/NPD R3 E15keV/'
#image_path = 'Test/SCD R4 E10.3keV/'
#image_path = 'Test/NPD R4 E10.3keV/'
#image_path = 'Test/SCD R4 E15keV/'
#image_path = 'Test/NPD R4 E15keV/'

#article putch
image_path = 'Test/NPD R4 E10.3keV_2/'
#image_path = 'Test/SCD R4 E10.3keV_2/'


trash_path = '_'+image_path


pixel_size = 0.1625				# 2D detector pixel size, in um!
ROI = (0, 0, 1000, 1000)		# ROI of analysis for "Stage 1"
ROI2 = [92, 92]				# ROI2 size for print image in "Stage 2"

fine = 5						# fit loops = 10 is best fit, but it crashs. 5 - is good!

# line profile settings
lp_lenth = 90					# line profile, pix
lp_lenth = 180					# line profile, pix
lp_width = 1					# in pix, not realized
lp_num = lp_lenth*1				# line profile points number
fit_error = 0.2					# in um

# ? for Stage 1
#start	= 0.225					# start position in m!
start	= 0.334					# start position in m!
step	= 0.01					# step in m
average = 1						# number of images in same position

# for Stage 2 ()
L2start	= 0.225					# start position in m!
L2step	= 0.01					# step in m
average = 1						# number of images in same position


log = 'on'						# on/off save logs through "sys.stdout"


# fix font size for article
plt.rcParams['font.size'] = '15' # font size tuning


#exec(open("DOF_analysis_stage1_v3.py").read())

exec(open("DOF_analysis_stage2_v3.py").read())

