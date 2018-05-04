
import sys
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from net import AE
from data import MyDataset




def train_net(net, epochs=5, batchSize=1, lr=0.01, val_percent=0.05,
              cp=True, gpu=True):
    directory = '/home/ljf/LJF/XXXNet/train_0.5/data/'
    datasetImage = 'rgb_list.txt'
    datasetDepth = 'depth_list.txt'
    dir_checkpoint = '/home/ljf/LJF/XXXNet/train_0.5/'
    datasetPose = 'vo_gt.txt'
    poses_xy = []
    poses_yaw = []
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
            poses_xy.append((p1, p2))
            #poses_yaw.append((p4, p5, p6))
    f3.close()
    my_dataset = MyDataset(poses_xy,images,depths)

    train_loader = torch.utils.data.DataLoader(dataset=my_dataset,
                                               batch_size=batchSize,
                                               shuffle=True,
                                               )
    #optimizer = optim.SGD(net.parameters(),
    #                   lr=lr, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam([
               {'params': net.encoder.parameters(), 'lr': 0.001},
               {'params': net.decoder.parameters()},
               {'params': net.poseRegress1.parameters(), 'lr': 0.001}
               #{'params': net.poseRegress2.parameters(), 'lr': 0.001}
            ], lr=0.01, betas=(0.9, 0.999))

    #optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    #optimizer = optim.Adagrad(net.parameters(), lr=0.01, lr_decay=0, weight_decay=0)

    #optimizer = optim.SGD([
    #           {'params': net.encoder.parameters()},
    #           {'params': net.decoder.parameters()},
    #           {'params': net.poseRegress.parameters(), 'lr': 1e-3}
    #        ], lr=1e-2, momentum=0.9)


    criterion = nn.SmoothL1Loss()
    criterion2 = nn.MSELoss()
    # When iteration starts, queue and thread start to load dataset from files.
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        #data_iter = iter(train_loader)
        # Mini-batch images and labels.
        #poses, images, depths = data_iter.next()
        #print(images.shape)
        # Actual usage of data loader is as below.
        l1 = 0;
        l2 = 0;
        l3 = 0;
        for i, (poses_xy, images, depths) in enumerate(train_loader):
            # Your training code will be written here
            images = Variable(images).cuda()
            depths = Variable(depths).cuda()
            poses_xy = Variable(poses_xy).cuda()
            #poses_yaw = Variable(poses_yaw).cuda()
            optimizer.zero_grad()  # zero the gradient buffer
            pose_xy, depth = net(images)
            loss1 = criterion(depth, depths)
            l1 += loss1
            loss2 = criterion2(pose_xy, poses_xy)*100
            l2 += loss2
            #loss3 = criterion2(pose_yaw, poses_yaw)
            #l3 += loss3
            #loss = loss1 + loss2 + loss3
            loss = loss1 + loss2 
            l3 += loss
            loss.backward()
            optimizer.step()
            #print ('Loss1: %.4f, Loss2: %.4f, Loss: %.4f'
            #    %(loss1.data[0], loss2.data[0], loss.data[0]))

            if (i+1)%20 == 0:
                print ('Epoch [%d/%d], Step [%d], ave_Loss1: %.4f, ave_Loss2: %.4f, Loss: %.4f'
                       %(epoch+1, epochs, i+1,  l1/i, l2/i, l3/i))
        if cp and epoch%10 == 0:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}_0.pth'.format(epoch + 1))

            print('Checkpoint {} saved !'.format(epoch + 1))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=5,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
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
        train_net(net, options.epochs, options.batchsize, options.lr,
                  gpu=options.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        sys.exit(0)

