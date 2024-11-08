import datetime
import glob
import numpy as np
import os
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt
from model import DisBlock2d, DisBlock3d, Discriminator, EncBlock, DecBlock, Generator
from torch.autograd import Variable

date_split = ['20130101', '20141231', '20161231', '20201231']
# 훈련기간 : 2013~2014, 2017~2020
# 시험기간 : 2015~2016

name = [
    'YD', # 영덕
    'NH', # 남해
    'TY', # 통영
    'YS', # 여수
    'WD', # 완도
    'MP', # 목포
    'BH', # 보령
    'CB', # 칠발도
    'PH', # 포항
    'GM', # 거문도
    'GJ', # 거제도
    'DJ', # 덕적도
    'DH', # 동해
    'MR', # 마라도
    'YY', # 외연도
    'UL', # 울릉도
]

path = [
    '../data/SST_ts/YD.dat',
    '../data/SST_ts/NH.dat',
    '../data/SST_ts/TY.dat',
    '../data/SST_ts/YS.dat',
    '../data/SST_ts/WD.dat',
    '../data/SST_ts/MP.dat',
    '../data/SST_ts/BH.dat',
    '../data/SST_ts/CB.dat',
    '../data/SST_ts/PH.dat',
    '../data/SST_ts/GM.dat',
    '../data/SST_ts/GJ.dat',
    '../data/SST_ts/DJ.dat',
    '../data/SST_ts/DH.dat',
    '../data/SST_ts/MR.dat',
    '../data/SST_ts/YY.dat',
    '../data/SST_ts/UL.dat',
]

if __name__ == '__main__' :
    batch_size  = 1
    iterations  = 20000
    time_step   = 7
    x_array     = []
    y_array     = []
    z_array     = []
    x_image     = []
    y_image     = []
    z_image     = []
    ERA5        = sorted(glob.glob('../data/ERA5/global/*.npy'))
    OSTIA       = sorted(glob.glob('../data/OSTIA/peninsula/image1.0/*.npy'))

    ########################################################################################
    # insitu 데이터 불러오기

    for h in range(len(path)):
        data = pd.read_csv(path[h], encoding='cp949')
        data.index = data['date']

        x_data = data.loc[pd.IndexSlice[date_split[0]:date_split[1]], :]
        x_data = np.stack([x_data.loc[:, 'date'], x_data.loc[:, 'SST']], axis=1)

        if float(date_split[0]) != x_data[0][0]:
            new_data = [[float(date_split[0]), np.nan]]
        else:
            new_data = [x_data[0]]
 
        for i in range(1, len(x_data)):
            if i == 1:
                prev_date = datetime.datetime(int(date_split[0][:4]), int(date_split[0][4:6]), int(date_split[0][6:]))
            else:
                prev_date = datetime.datetime(int(x_data[i - 1][0]) // 10000, (int(x_data[i - 1][0]) // 100) % 100, int(x_data[i - 1][0]) % 100)

            next_date = datetime.datetime(int(x_data[i][0]) // 10000, (int(x_data[i][0]) // 100) % 100, int(x_data[i][0]) % 100)

            for j in range((next_date - prev_date).days - 1):
                new_data.append(np.ones_like(x_data[0]) * np.nan)

            new_data.append(x_data[i])

        x_array.append(np.array(new_data)[:, 1:])

        y_data = data.loc[pd.IndexSlice[date_split[1]:date_split[2]], :]
        y_data = np.stack([y_data.loc[:, 'date'], y_data.loc[:, 'SST']], axis=1)[1:]

        new_data = [y_data[0]]

        for i in range(1, len(y_data)):
            prev_date = datetime.datetime(int(y_data[i - 1][0]) // 10000, (int(y_data[i - 1][0]) // 100) % 100, int(y_data[i - 1][0]) % 100)
            next_date = datetime.datetime(int(y_data[i][0]) // 10000, (int(y_data[i][0]) // 100) % 100, int(y_data[i][0]) % 100)

            for j in range((next_date - prev_date).days - 1):
                new_data.append(np.ones_like(y_data[0]) * np.nan)

            new_data.append(y_data[i])

        y_array.append(np.array(new_data)[:, 1:])

        z_data = data.loc[pd.IndexSlice[date_split[2]:date_split[3]], :]
        z_data = np.stack([z_data.loc[:, 'date'], z_data.loc[:, 'SST']], axis=1)[1:]
        
        new_data = [z_data[0]]
        
        for i in range(1, len(z_data)):
            prev_date = datetime.datetime(int(z_data[i - 1][0]) // 10000, (int(z_data[i - 1][0]) // 100) % 100, int(z_data[i - 1][0]) % 100)
            next_date = datetime.datetime(int(z_data[i][0]) // 10000, (int(z_data[i][0]) // 100) % 100, int(z_data[i][0]) % 100)
            
            for j in range((next_date - prev_date).days - 1):
                new_data.append(np.ones_like(z_data[0]) * np.nan)
                
            new_data.append(z_data[i])
            
        z_array.append(np.array(new_data)[:, 1:])

    x_array = np.array(x_array, dtype=np.float32) # insitu 훈련 데이터1
    y_array = np.array(y_array, dtype=np.float32) # insitu 시험 데이터
    z_array = np.array(z_array, dtype=np.float32) # insitu 훈련 데이터2

    print('x_array:', x_array.shape, 'y_array:', y_array.shape, 'z_array:', z_array.shape)

    array_mean = np.expand_dims(np.load('../statistics/array_mean.npy'), axis=1)
    array_std  = np.expand_dims(np.load('../statistics/array_std.npy'), axis=1)

    x_array = (x_array - array_mean) / array_std
    y_array = (y_array - array_mean) / array_std
    z_array = (z_array - array_mean) / array_std

    ########################################################################################
    # ERA5 데이터 불러오기

    for i in range(len(ERA5)):
        year = int(ERA5[i].split('\\')[-1].split('.')[0].split('_')[0])

        if year == 2015 or year == 2016:
            if len(y_image) == 0:
                y_image.append(np.array(np.load(ERA5[i])))
        elif year < 2015:
            if len(x_image) == 0:
                x_image.append(np.array(np.load(ERA5[i])))
        else:
            if len(z_image) == 0:
                z_image.append(np.array(np.load(ERA5[i])))

        print('ERA5', i + 1, '/', len(ERA5))

    x_ERA5 = np.concatenate(x_image) # ERA5 훈련 데이터1
    y_ERA5 = np.concatenate(y_image) # ERA5 시험 데이터
    z_ERA5 = np.concatenate(z_image) # EAR5 훈련 데이터2

    print('x_ERA5:', x_ERA5.shape, 'y_ERA5:', y_ERA5.shape, 'z_ERA5:', z_ERA5.shape)
    plt.imshow(x_ERA5[0])
    plt.show()

    plt.imshow(x_ERA5[0][150:270, 450:570])
    plt.show()

    ERA5_mean  = np.load('../statistics/ERA5_mean.npy')
    ERA5_std   = np.load('../statistics/ERA5_std.npy')
    OSTIA_mean = np.load('../statistics/OSTIA1.0_mean.npy')
    OSTIA_std  = np.load('../statistics/OSTIA1.0_std.npy')

    x_ERA5 = (x_ERA5 - ERA5_mean) / ERA5_std
    y_ERA5 = (y_ERA5 - ERA5_mean) / ERA5_std
    z_ERA5 = (z_ERA5 - ERA5_mean) / ERA5_std

    ########################################################################################
    # OSTIA 데이터 불러오기

    x_OSTIA = OSTIA[:730]                   # OSTIA 훈련 데이터1
    y_OSTIA = OSTIA[730:730 + len(y_ERA5)]  # OSTIA 시험 데이터
    z_OSTIA = OSTIA[-len(z_ERA5):]          # OSTIA 훈련 데이터2

    OSTIA = np.load(x_OSTIA[0])
    plt.imshow(OSTIA)
    plt.show()

    ########################################################################################
    # 훈련에 필요한 함수 정의

    def calculate_gradient_penalty(discriminator, real_x, fake_x, condition=None):
        eta = torch.FloatTensor(real_x.size(0), 1, 1, 1).uniform_(0, 1)
        
        interpolated_x = eta * real_x + (1 - eta) * fake_x
        
        # define it to calculate gradient
        interpolated_x = Variable(interpolated_x, requires_grad=True)
        condition      = Variable(condition, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = discriminator(interpolated_x, condition)
        
        gradient = autograd.grad(outputs=prob_interpolated, inputs=[interpolated_x, condition], grad_outputs=torch.ones(prob_interpolated.size()), create_graph=True, retain_graph=True)
        gradient = torch.cat([gradient[0].view(real_x.size(0), -1), gradient[1].view(condition.size(0), -1)], dim=1)
            
        return ((gradient.norm(2, dim=1) - 1) ** 2).mean()

    def Generate_Batch(x_ERA5, x_OSTIA, x_array, OSTIA_statistics, batch_size, time_step):
        x_batch = []
        y_batch = []
        z_batch = []

        while len(x_batch) < batch_size:
            time_index = np.random.randint(len(x_ERA5) - time_step + 1)
            x_batch.append(x_ERA5[time_index:time_index + time_step])
            y_batch.append(np.load(x_OSTIA[time_index + time_step - 1]))
            z_batch.append(x_array[:, time_index:time_index + time_step])

        x_batch = np.array(x_batch)
        x_batch = Variable(torch.from_numpy(x_batch)).type(torch.FloatTensor)
        x_batch[torch.isnan(x_batch)] = 0

        y_batch = (np.array(y_batch) - OSTIA_statistics[0][:, 750:1350, 2250:2850]) / OSTIA_statistics[1][:, 750:1350, 2250:2850] # OSTIA 한반도 600x600
        y_batch = Variable(torch.from_numpy(y_batch)).type(torch.FloatTensor)

        z_batch = np.transpose(z_batch, (0, 2, 1, 3))
        z_batch = np.reshape(z_batch, (batch_size, time_step, -1))
        z_batch = Variable(torch.from_numpy(z_batch)).type(torch.FloatTensor)

        return x_batch[:, :, 150:270, 450:570].unsqueeze(1), y_batch.unsqueeze(1), z_batch # ERA5 한반도 120x120  

    ########################################################################################
    # 모델 및 옵티마이저 정의

    D = Discriminator(DisBlock2d, DisBlock3d)
    G = Generator(EncBlock, DecBlock)

    D_optimizer = optim.Adam(D.parameters(), lr=0.0001, betas=(0, 0.9), weight_decay=0.0001)
    G_optimizer = optim.Adam(G.parameters(), lr=0.0001, betas=(0, 0.9), weight_decay=0.0001)

    loss = [None] * 5
    i = 0

    while i < iterations:
        ####################################################################################
        # 훈련용 데이터 준비
        
        if np.random.randint(2) == 0:
            input, label, insitu = Generate_Batch(x_ERA5, x_OSTIA, x_array, (OSTIA_mean, OSTIA_std), batch_size, time_step)
        else:
            input, label, insitu = Generate_Batch(z_ERA5, z_OSTIA, z_array, (OSTIA_mean, OSTIA_std), batch_size, time_step)

        H = 750
        W = 2250

        # infer(예측 OSTIA)  : (600, 600),
        # array(예측 insitu) : (16,)
        infer, array = G(input)
        array = torch.stack([
            array[:, :, 1068 - H, 2588 - W], # YD
            array[:, :, 1102 - H, 2559 - W], # NH
            array[:, :, 1103 - H, 2564 - W], # TY
            array[:, :, 1107 - H, 2554 - W], # YS
            array[:, :, 1116 - H, 2537 - W], # WD
            array[:, :, 1104 - H, 2527 - W], # MP
            array[:, :, 1071 - H, 2528 - W], # BH            
            array[:, :, 1110 - H, 2509 - W], # CB
            array[:, :, 1075 - H, 2589 - W], # PH
            (array[:, :, 1119 - H, 2546 - W] + array[:, :, 1120 - H, 2546 - W]) / 2, # GM 
            array[:, :, 1110 - H, 2570 - W], # GJ
            array[:, :, 1057 - H, 2520 - W], # DJ
            array[:, :, 1054 - H, 2591 - W], # DH
            (array[:, :, 1139 - H, 2520 - W] + array[:, :, 1140 - H, 2520 - W]) / 2, # MR
            (array[:, :, 1077 - H, 2508 - W] + array[:, :, 1078 - H, 2508 - W]) / 2, # YY
            array[:, :, 1054 - H, 2621 - W]], axis=2) # UL

        print('input:', input.shape, 'predicted OSTIA:', infer.shape, 'predicted insitu:', array.shape)

        # OSTIA ground-truth에 nan값이 존재하는 경우 무시하도록 mask 생성
        mask = torch.ones_like(label[:, :1])
        mask[torch.isnan(label[:, :1])] = 0
        label[torch.isnan(label)] = 0
        infer = infer * mask

        # insitu ground-truth에 nan값이 존재하는 경우 무시하도록 mask 생성
        mask = torch.ones_like(insitu)
        mask[torch.isnan(insitu)] = 0
        insitu[torch.isnan(insitu)] = 0
        array = array * mask

        ####################################################################################
        # 모델 훈련

        D_optimizer.zero_grad()
        D_loss = [-D(label, input).mean(), D(infer.detach(), input).mean(), calculate_gradient_penalty(D, label, infer, input)]
        (D_loss[0] + D_loss[1] + 10 * D_loss[2]).backward()
        D_optimizer.step()

        loss[0] = D_loss[0].item()
        loss[1] = D_loss[1].item()
        loss[2] = D_loss[2].item()

        if i % 5 == 0:
            G_optimizer.zero_grad()
            G_loss = [-D(infer, input).mean(), nn.MSELoss()(array, insitu)]
            #         SR Loss,                 MSE Loss
            (G_loss[0] + 10 * G_loss[1]).backward()
            G_optimizer.step()

            loss[3] = G_loss[0].item()
            loss[4] = G_loss[1].item()

        print(i + 1, '/', iterations, '\tD:', loss[0], loss[1], loss[2], '\tG:', loss[3], loss[4])
        i += 1

        ####################################################################################
        # 1000번 반복마다 모델 저장

        if i % 1000 == 0 or i == 1:
            D.eval()
            G.eval()

            with torch.no_grad():
                image = [G(input)[0].cpu().data.numpy(), label.cpu().data.numpy()]

            if not os.path.exists('checkpoint/{}'.format(i)):
                os.makedirs('checkpoint/{}'.format(i))

            for h in range(batch_size):
                a = image[0][h] * OSTIA_std[:, 750:1350, 2250:2850] + OSTIA_mean[:, 750:1350, 2250:2850]
                b = image[1][h] * OSTIA_std[:, 750:1350, 2250:2850] + OSTIA_mean[:, 750:1350, 2250:2850]
                c = np.concatenate([a, b], axis=-2)
                c = np.transpose(c, [1, 2, 0])

                plt.figure(figsize=(20, 10))
                plt.imshow(c)
                plt.colorbar()
                plt.axis('off')
                plt.savefig('checkpoint/{}/{}.png'.format(i, h), bbox_inches='tight')
                plt.close()

                diff = np.abs(a - b)[0]
                diff[np.isnan(diff)] = 0

                plt.figure(figsize=(20, 10))
                plt.imshow(diff, cmap='gray')
                plt.axis('off')
                plt.colorbar()
                plt.savefig('checkpoint/{}/{}_difference_map.png'.format(i, h), bbox_inches='tight')
                plt.close()

            torch.save({
                        'D_model': D.state_dict(),
                        'G_model': G.state_dict(),
                        'D_optimizer': D_optimizer.state_dict(),
                        'G_optimizer': G_optimizer.state_dict(),
                        }, 'checkpoint/{}/checkpoint.pt'.format(i))

            D.train()
            G.train()
