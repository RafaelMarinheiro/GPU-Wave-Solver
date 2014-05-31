# from pylab import *
# from numpy import outer
# rc('text', usetex=False)
# a=outer(arange(0,1,0.01),ones(10))
# print a
# figure(figsize=(10,5))
# subplots_adjust(top=0.8,bottom=0.05,left=0.01,right=0.99)
# maps=[m for m in cm.datad if not m.endswith("_r")]
# maps.sort()
# l=len(maps)+1
# for i, m in enumerate(maps):
#     subplot(1,l,i+1)
#     axis("off")
#     imshow(a,aspect='auto',cmap=get_cmap(m),origin="lower")
#     title(m,rotation=90,fontsize=10)
# savefig("colormaps.png",dpi=100,facecolor='gray')

from pylab import *
import numpy

for i in range(0, 200):
	print "Frame " + str(i)
	data = numpy.loadtxt("frames/frame" + str(i))
	figure(figsize=(5,5))
	# imshow(data, aspect='auto', cmap=get_cmap('seismic'),origin="lower")
	imshow(data, aspect='auto', cmap=get_cmap('seismic'), vmin=-0.1, vmax=0.1,origin="lower")
	savefig("frames/frame" + str(i) + ".png",dpi=100,facecolor='gray')