from net_parts import *
class AE(nn.Module):
	def __init__(self):
		super(AE,self).__init__()
		
        self.inc = inconv(6, 64)
        self.down1 = down1(64, 128)
        self.down2 = down1(128, 256)
        self.down3 = down2(256, 512)
        self.down4 = down2(512, 512)
        self.down5 = down2(512, 1024)
        self.encode = encoder_out(1024, 1024)

        self.up1 = only_up(1024, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 64)
        self.outc = outconv(64, 1)

		self.poseRegress1 = nn.Sequential(
			fc(1024*4*7,2048),
			fc(2048,128),
			nn.Linear(128,2))


	def forward(self, x):

		x_inc = self.inc(x)
		x1 = self.down1(x_inc)
		x2 = self.down2(x2)
		x3 = self.down3(x2)
		x4 = self.down4(x3)
		x5 = self.down5(x4)
		x_en = self.encoder_out(x5)

		y = self.up1(x_en)
		y = self.up2(y, x4)
		y = self.up3(y, x2)
		y = self.up4(y, x1)
		depth = self.outc(y)

		featureVector = x_en.view((-1,1024*4*7))
		#print(featureVector.size())
		pose_xy = self.poseRegress1(featureVector)
		#pose_yaw = self.poseRegress2(featureVector)

		return pose_xy, depth