import datetime
import glob
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from model import EncBlock, DecBlock, Generator
from torch.autograd import Variable

date_split = ['20130101', '20141231', '20201231']

if __name__ == '__main__' :
    time_step   = 7
    x_image     = []
    y_image     = []
    z_image     = []
    ERA5        = sorted(glob.glob('../data/ERA5/global/*.npy'))
    OSTIA       = sorted(glob.glob('../data/OSTIA/peninsula/image1.0/*.npy'))

    ########################################################################################
    # ERA5 & OSTIA 데이터 불러오기

    for i in range(len(ERA5)):
        year = int(ERA5[i].split('\\')[-1].split('.')[0].split('_')[0])

        if year == 2015 or year == 2016:
            if len(y_image) == 0:
                y_image.append(np.load(ERA5[i]))

        print('ERA5', i + 1, '/', len(ERA5))
    
    ERA5_mean  = np.load('../statistics/ERA5_mean.npy')
    ERA5_std   = np.load('../statistics/ERA5_std.npy')
    OSTIA_mean = np.load('../statistics/OSTIA1.0_mean.npy')
    OSTIA_std  = np.load('../statistics/OSTIA1.0_std.npy')

    y_ERA5  = np.concatenate(y_image)
    y_ERA5  = (y_ERA5 - ERA5_mean) / ERA5_std
    y_OSTIA = OSTIA[730:730 + len(y_ERA5)]

    ########################################################################################
    # 모델 불러오기

    G = Generator(EncBlock, DecBlock)
    checkpoint = torch.load('checkpoint/20000/checkpoint.pt', map_location='cpu')
    G.load_state_dict(checkpoint['G_model'])
    G.eval()

    ########################################################################################
    # ERA5 → OSTIA 예측 데이터 생성

    error = []

    date = datetime.datetime(int(date_split[1][:4]), int(date_split[1][4:6]), int(date_split[1][6:8]))
    date = date + relativedelta(days=time_step)

    for i in range(len(y_ERA5) - time_step + 1):
        with torch.no_grad():
            input = np.array(y_ERA5[i:i + time_step])[:, 150:270, 450:570]
            input = Variable(torch.from_numpy(input)).type(torch.FloatTensor).unsqueeze(0).unsqueeze(1)
            input[torch.isnan(input)] = 0
            label = np.load(y_OSTIA[i + time_step - 1])
            infer = G(input)[0][0][0].cpu().data.numpy() * OSTIA_std[0, 750:1350, 2250:2850] + OSTIA_mean[0, 750:1350, 2250:2850]

        if not os.path.exists('result/{}-{:02d}-{:02d}'.format(date.year, date.month, date.day)):
            os.makedirs('result/{}-{:02d}-{:02d}'.format(date.year, date.month, date.day))

        np.save('result/{}-{:02d}-{:02d}/infer.npy'.format(date.year, date.month, date.day), infer)
        np.save('result/{}-{:02d}-{:02d}/label.npy'.format(date.year, date.month, date.day), label)

        error.append((infer - label) ** 2)

        plt.figure(figsize=(20, 10))
        plt.imshow(infer)
        plt.colorbar()
        plt.axis('off')
        plt.savefig('result/{}-{:02d}-{:02d}/infer.png'.format(date.year, date.month, date.day), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(20, 10))
        plt.imshow(label)
        plt.colorbar()
        plt.axis('off')
        plt.savefig('result/{}-{:02d}-{:02d}/label.png'.format(date.year, date.month, date.day), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(20, 10))
        plt.imshow(error[-1])
        plt.colorbar()
        plt.axis('off')
        plt.savefig('result/{}-{:02d}-{:02d}/error.png'.format(date.year, date.month, date.day), bbox_inches='tight')
        plt.close()

        date = date + relativedelta(days=1)

        print(i + 1, '/', len(y_ERA5))

    print(np.nanmean(error))

    map = np.sqrt(np.nanmean(error, axis=0))

    plt.figure(figsize=(20, 10))
    plt.imshow(map)
    plt.colorbar()
    plt.axis('off')
    plt.savefig('result/RMSE.png', bbox_inches='tight')
    plt.close()
