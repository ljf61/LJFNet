
import sys
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from net import AE
from data import MyDataset
import numpy




def test_net(net, epochs=5, batchSize=1, lr=0.01, val_percent=0.05,
              cp=True, gpu=True):
    directory = '/home/ljf/LJF/LJFNet/test_0.2/test1/'
    datasetImage = 'rgb_list.txt'
    datasetDepth = 'depth_list.txt'
    #dir_checkpoint = '/home/ljf/LJF/LJFNet/train_0.2/'
    datasetPose = 'vo_gt.txt'
    poses = []
    poses_pry = []
    images = []
    depths= []
    
    with open(directory+datasetImage) as f1:
        next(f1)  # skip the 3 header lines
        next(f1)
        next(f1)
        for line in f1:

            fname = line.strip('\n')
            images.append(directory+fname+'.png')
    f1.close()
    with open(directory+datasetDepth) as f2:
        next(f2)  # skip the 3 header lines
        next(f2)
        next(f2)
        for line in f2:
            fname = line.strip('\n')
            depths.append(directory+fname+'.png')
    f2.close()
    with open(directory + datasetPose) as f3:
        next(f3)  # skip the 3 header lines
        next(f3)
        next(f3)
        for line in f3:
            p0, p1, p2, p3, p4, p5, p6= line.split()
            #p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            p5 = float(p5)
            p6 = float(p6)
            poses.append((p1, p2, p3))
            #poses_pry.append((p4, p5, p6))
    f3.close()
    my_dataset = MyDataset(poses, images,depths)

    test_loader = torch.utils.data.DataLoader(dataset=my_dataset,
                                               batch_size=batchSize,
                                               shuffle=False,
                                               )
    #optimizer = optim.SGD(net.parameters(),
    #                    lr=lr, momentum=0.9, weight_decay=0.0005)
    #optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    #optimizer = optim.Adagrad(net.parameters(), lr=0.01, lr_decay=0, weight_decay=0)

    #optim.SGD([
    #            {'params': model.base.parameters()},
    #           {'params': model.classifier.parameters(), 'lr': 1e-3}
    #        ], lr=1e-2, momentum=0.9)

    xy = numpy.zeros((len(test_loader),3))
    criterion = nn.SmoothL1Loss()
    criterion2 = nn.MSELoss()
    # When iteration starts, queue and thread start to load dataset from files.
	#print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
	#data_iter = iter(train_loader)
	# Mini-batch images and labels.
	#poses, images, depths = data_iter.next()
	#print(images.shape)
	# Actual usage of data loader is as below.
    for i, (poses, images, depths) in enumerate(test_loader):

        # Your training code will be written here
        images = Variable(images).cuda()
        depths = Variable(depths).cuda()
        poses = Variable(poses).cuda()
        #optimizer.zero_grad()  # zero the gradient buffer
        pose, depth = net(images)
        p = pose.cpu().data.numpy()
        p = numpy.squeeze(p)
        xy[i,:] = p
        #print(pose)
        #filetest.write(pose.data)
 
        loss1 = criterion(depth, depths)
        loss2 = criterion2(pose, poses)
        #loss = loss1 + loss2
        #loss.backward()
        #optimizer.step()
        #if i % 10 == 0:
    	print ('Step [%d/%d], Loss1: %.4f,  Loss2: %.4f'
    	       %(i+1, len(test_loader)//batchSize, loss1.data[0], loss2.data[0]))

    numpy.savetxt('/home/ljf/LJF/LJFNet/test_0.2/test1/results(xyz)_0.txt',xy, delimiter=' ')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')

    (options, args) = parser.parse_args()

    net = AE()

    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

    if options.gpu:
        net.cuda()
        cudnn.benchmark = True

    try:
        test_net(net, options.epochs, options.batchsize, options.lr,
                  gpu=options.gpu)
    except KeyboardInterrupt:
        sys.exit(0)

