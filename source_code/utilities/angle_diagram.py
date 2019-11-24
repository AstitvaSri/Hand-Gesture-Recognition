def angle_diagram(x1,x2,x3,x4,palm_point,input_image):
	row = input_image.shape[0]
	col = input_image.shape[1]

	copy = np.zeros((row,col))

	copy = input_image

	#box coordinates
	x1=[40,15]
	x2=[20,25]


	x4=[40,65]
	x3=[60,55]

	palm_point = [75,75]

	top_mid = [(x1[0]+x2[0])//2,(x1[1]+x2[1])//2]
	base_mid = [(x3[0]+x4[0])//2,(x3[1]+x4[1])//2]
	block_mid = [(top_mid[0]+base_mid[0])//2,(top_mid[1]+base_mid[1])//2]

	
	#BELOW COMMENTED CODE DRAWS A BOX AROUND IMAGE'S SEGMENT
	
	#drawing the block == top
	# m = (x2[1]-x1[1])/(x2[0]-x1[0])
	# c = x2[1] - m*x2[0]

	# for x in range(x2[0],x1[0]):
	# 	y=int(m*x+c)
	# 	blank[x,y] = 255

	# #drawing the block == bottom

	# m = (x4[1]-x3[1])/(x4[0]-x3[0])
	# c = x4[1] - m*x4[0]

	# for x in range(x4[0],x3[0]):
	# 	y=int(m*x+c)
	# 	blank[x,y] = 255

	# #drawing the block == right

	# m = (x4[1]-x2[1])/(x4[0]-x2[0])
	# c = x4[1] - m*x4[0]

	# for x in range(x2[0],x4[0]):
	# 	y=int(m*x+c)
	# 	blank[x,y] = 255

	# #drawing the block == left

	# m = (x3[1]-x1[1])/(x3[0]-x1[0])
	# c = x3[1] - m*x3[0]

	# for x in range(x1[0],x3[0]):
	# 	y=int(m*x+c)
	# 	blank[x,y] = 255


	m = (base_mid[1]-top_mid[1])/(base_mid[0]-top_mid[0])
	c = base_mid[1] - m*base_mid[0]

	for x in range(block_mid[0],base_mid[0]):
		y=int(m*x+c)
		copy[x,y] = 255
	print(top_mid,base_mid,block_mid)

	m_to_palm = (base_mid[1]-palm_point[1])/(base_mid[0]-palm_point[0])
	c_of_palm = palm_point[1] - m_to_palm*palm_point[0]

	print(m_to_palm,c_of_palm)
	for x in range(base_mid[0],palm_point[0]):
		# print(m,x,c_of_palm,y)
		y=int(m_to_palm*x+c_of_palm)
		copy[x,y] = 255

	theta = atan(m_to_palm)
	theta = int((theta/pi)*180)
	if(m_to_palm>0):
		theta += 90
	else:
		theta = -1*theta

	print(theta,"degrees")

	fig = plt.figure()
	ax0 = fig.add_subplot(2,1,1)
	ax0.imshow(input_image,cmap='gray')
	ax1 = fig.add_subplot(2,1,2)
	ax1.imshow(copy,cmap='gray')

	plt.show()
