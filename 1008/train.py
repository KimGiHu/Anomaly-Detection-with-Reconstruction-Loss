import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
from model import *
from tqdm import tqdm

# 시드 값 설정
seed = 42

# 기본 시드 고정
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# CUDA 사용 시 추가 설정
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티-GPU 사용 시
    # CuDNN 결정론적 및 비결정론적 동작 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 데이터 준비
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
# npy file path setting
npy_file_path = os.getcwd()

M04 = np.load(os.path.join(npy_file_path, 'data/M04.npy'))

print(f"npy array shape: {M04.shape}")

file_index = 0

time = np.arange(M04.shape[1]) * (0.4)  # 5000 points = 2000 us
print(f"time array shape : {time.shape}")

channels = ['C1', 'C2', 'C3', 'C4']
num_channels = len(channels)
print(f"the number of unique channel : {num_channels}")

# ########################################### 4가지 고유 파형의 형태를 plot함. ###########################################
# plt.figure(figsize=(10, 6))
# channels = ['C1', 'C2', 'C3', 'C4']
# for channel_index in range(M04.shape[2]):
#     data_to_plot = M04[file_index, :, channel_index] # 14763개 데이터 중에서 첫번째 데이터를 plot하기 위해 받아옴.
#     plt.plot(time, data_to_plot, label=f"Channel {channels[channel_index]}") # 고유파형 채널별로 plot함.

# plt.title(f"Data from File {file_index} (Channels 1-4)") # plot하는 그래프의 이름 정하기
# plt.xlabel("Time [us]") # plot하는 그래프 x축 라벨
# plt.ylabel("Amplitude") # plot하는 그래프 y축 레벨
# plt.legend() # plot된 고유파형들 라벨링
# plt.grid(True) # 격자 활성화
# plt.savefig('./test.png', dpi=600)

# 데이터 전처리 파트
data_input = M04[:int((M04.shape[0])*0.8),:,:]
print(f'the shape of input data : {data_input.shape}')

scaler = MinMaxScaler() # Min-Max 스케일링
data_input = scaler.fit_transform(data_input.reshape(-1,1)).reshape(data_input.shape)
print(f'the shape of Min-Max scaled input data : {data_input.shape}')

data_input = data_input.transpose(0,2,1) # 입력 형태 (N, 5002, 4)를 (N, 4, 5002)로 변경함.
print(f'the shape of transposed input data : {data_input.shape}')

data_tensor = torch.tensor(data_input, dtype=torch.float32) # numpy 데이터를 tensor 위에 올림

# 원-핫 벡터 생성
one_hot_vector = np.zeros(4)

case = 4
one_hot_vector[case-1] = 1
labels_one_hot = np.tile(one_hot_vector, (data_input.shape[0], 1))
print(f"the shape of one-hot labels : {labels_one_hot.shape}")
labels_one_hot = torch.tensor(labels_one_hot, dtype=torch.float32)

dataset1 = CustomDataset(data_tensor, labels_one_hot)
dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=True)

# 손실 함수 정의 - my proposal
def loss_function_sum(x_recon, x, mu, logvar):    

    MSE_lib = nn.functional.mse_loss(x_recon, x,  reduction='sum') # divided into batch size, time steps
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)# see Appendix B from VAE paper:
                                                                                           # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    
    return MSE_lib + 1.0*KLD + 1e-12

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 학습 파라미터 설정
input_dim = num_channels # 고유 채널 숫자
hidden_dim = data_input.shape[2] # 샘플링 타임당 수집되는 샘플 수 
latent_dim = 512 # 잠재공간의 특징점(feature) 갯수
condition_dim = 4 # 조건 : 모듈 네임을 원-핫 벡터로 변환하여 잠재공간의 입출력으로 사용함.
dropout = 0.2 # 드랍아웃 확률

num_epochs = 100

if __name__ == '__main__':
    # 조건부 변이 오토인코더 정의하기
    model_CVAE = CVAE_rev(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        condition_dim=condition_dim,
        dropout_prob=dropout
    ).to(device)

    # 모델의 학습 가중치들 학습모드로 활성화, 옵티마이저 설정(학습률 및 학습 파라미터 정의) 및 학습 스케쥴러 설정
    model_CVAE.train()
    optimizer = optim.Adam(model_CVAE.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True) # 3회 이상 개선점이 없으면 스케쥴러 업데이트, 그 폭은 10%씩 바뀜, 업데이트하는 것 시각화활성화

    for epoch in range(num_epochs):
        total_loss = 0 # 에포크당 평균 손실값을 계산하기 위해 사용.
        middle_loss = 0 # 배치사이즈마다 계산된 손실값의 총합을 저장하기 위해 사용.

        # tqdm을 이용하여 학습의 진행상황을 시각적으로 표시함.
        pbar = tqdm(dataloader1, desc=f'Epoch {epoch+1}/{num_epochs}', total=len(dataloader1), ncols=100)

        for x, c in pbar :
            # 입력 데이터 및 원-핫 조건벡터 tensor들을 device 위에 얹힘
            x = x.to(device)
            c = c.to(device)
            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 모델의 출력 값과 입력 값을 이용하여 손실값 계산.
            x_recon, mu, logvar = model_CVAE(x, c)
            loss = loss_function_sum(x_recon, x, mu, logvar)
            loss.backward() # back-propagation을 이용하여 손실함수 값 계산
            middle_loss += loss.item() # 현재 배치에 대한 손실함수 값을 middel_loss에 저장
            optimizer.step() # 가중치 파라미터 업데이트

            total_loss += middle_loss / 4 # 고유파형 4개의 정보에 대해서 분할함
            pbar.set_postfix(loss=total_loss)
        scheduler.step(total_loss) # total_loss값을 비교하여 학습 스케쥴러 업데이트
        avg_loss = total_loss / (len(dataloader1)+1) # 데이터로더의 길이는 전체 샘플수 / 배치사이즈 한 값이다.
        print(f' Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    # 학습 완료 후 모델 저장
    torch.save(model_CVAE.state_dict(), "./model/1007/M04_normal_80percent.pth")
