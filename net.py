from net_parts import *
class AE(nn.Module):
	def __init__(self):
		super(AE,self).__init__()
		self.encoder = nn.Sequential(
			inconv(6, 64),
			down1(64, 128),
			down1(128, 256),
			down2(256, 512),
			down2(512, 512),
			down2(512, 1024),
			encoder_out(1024, 1024))

		self.decoder = nn.Sequential(
			only_up(1024, 512),
			only_up(512, 256),
			only_up(256, 128),
			only_up(128, 64),
			only_up(64, 32),
			outconv(32, 1))

		self.poseRegress1 = nn.Sequential(
			fc(1024*4*7,2048),
			fc(2048,128),
			nn.Linear(128,2))

		
	def forward(self, x):

		featureVector = self.encoder(x)
		featureVector_pose = featureVector.view((-1,1024*4*7))
		#print(featureVector.size())
		pose_xy = self.poseRegress1(featureVector_pose)
		#pose_yaw = self.poseRegress2(featureVector)
		depth = self.decoder(featureVector)


		return pose_xy, depth