import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import seaborn as sns
import argparse
from model import *
from tqdm import tqdm
import os 
from sklearn.manifold import TSNE
import pandas as pd

print(torch.cuda.is_available())
# cuda 캐시 정리
torch.cuda.empty_cache()

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

# 데이터 전처리 파트
# data_input = M04[:,:,:] # 전체 데이터 테스트
data_input = M04[:int((M04.shape[0])*0.8),:,:] # 학습한 데이터
data_test = M04[int((M04.shape[0])*0.8):,:,:] # 테스트 데이터
print(f'the shape of input data : {data_input.shape}')

scaler_train = MinMaxScaler() # Min-Max 스케일링 학습용
sclaer_test = MinMaxScaler() # Min-Max 스케일링 테스트용
data_input = scaler_train.fit_transform(data_input.reshape(-1,1)).reshape(data_input.shape)
data_test = sclaer_test.fit_transform(data_test.reshape(-1,1)).reshape(data_test.shape)
print(f'the shape of Min-Max scaled trained data : {data_input.shape}')
print(f'the shape of Min-Max scaled test data : {data_test.shape}')

data_input = data_input.transpose(0,2,1) # 입력 형태 (N, 5002, 4)를 (N, 4, 5002)로 변경함.
data_test = data_test.transpose(0,2,1)
print(f'the shape of transposed trained data : {data_input.shape}')
print(f'the shape of transposed test data : {data_test.shape}')

data_tensor = torch.tensor(data_input, dtype=torch.float32) # numpy 데이터를 tensor 위에 올림 
test_tensor = torch.tensor(data_test, dtype=torch.float32)

# 원-핫 벡터 생성
one_hot_vector = np.zeros(4)

case = 4 #  unique 파형 4개
one_hot_vector[case-1] = 1
labels_one_hot = np.tile(one_hot_vector, (data_input.shape[0], 1))
test_labels_one_hot = np.tile(one_hot_vector, (data_test.shape[0], 1))

print(f"the shape of one-hot trained labels : {labels_one_hot.shape}")
print(f"the shape of one-hot test labels : {labels_one_hot.shape}")

labels_one_hot = torch.tensor(labels_one_hot, dtype=torch.float32)
test_labels_one_hot = torch.tensor(test_labels_one_hot, dtype=torch.float32)

# 학습 데이터세트 및 데이터로드 정의
dataset_train = CustomDataset(data_tensor, labels_one_hot)
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=False)

# 테스트 데이터세트 및 데이터로드 정의
dataset_test = CustomDataset(test_tensor, labels_one_hot)
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False)

# 손실 함수 정의 - 라이브러리
def loss_function(x_recon, x, mu, logvar):    
    MSE_lib = nn.functional.mse_loss(x_recon, x, reduction='sum') # divided into batch size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# see Appendix B from VAE paper:
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

if __name__ == '__main__':
    # 조건부 변이 오토인코더 정의하기
    model_CVAE = CVAE_rev(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        condition_dim=condition_dim,
        dropout_prob=dropout
    ).to(device)

    model_CVAE.load_state_dict(torch.load("./model/1007/M04_normal_80percent.pth"))
    
    model_CVAE.eval()

    with torch.no_grad():
        mse = []
        latent_mu = []
        latent_logvar = []
        latent_z = []

        pbar = tqdm(dataloader_train, total=len(dataloader_train), ncols=100)

        for data_input, label in pbar :
            data_input, label = data_input.to(device), label.to(device)

            # 모델 출력값 받아오기
            x_recon, mu, logvar = model_CVAE(data_input, label)
            z = model_CVAE.reparameterize(mu, logvar)

            mse.extend(((data_input - x_recon)**2).mean(dim=(1,2)).cpu().numpy())

            latent_z.append(z.cpu().numpy())
    
    with torch.no_grad():
        mse_test = []
        latent_mu_test = []
        latent_logvar_test = []
        latent_z_test = []

        pbar_test = tqdm(dataloader_test, total=len(dataloader_test), ncols=100)

        for data, label in pbar_test :
            data, label = data.to(device), label.to(device)

            # 모델 출력값 받아오기
            x_recon, mu, logvar = model_CVAE(data, label)
            z = model_CVAE.reparameterize(mu, logvar)

            mse_test.extend(((data - x_recon)**2).mean(dim=(1,2)).cpu().numpy())

            latent_z_test.append(z.cpu().numpy())
    
    ################################ t-SNE 기법을 이용한 잠재공간 시각화 ################################
    
    # # 잠재공간 z를 numpy 배열로 변환
    # latent_z = np.concatenate(latent_z, axis=0)
    # latent_z_test = np.concatenate(latent_z_test, axis=0)
    
    # #  잠재공간 Z의 차원을 2d로 변환
    # latent_z_2d = latent_z.reshape(latent_z.shape[0], -1)
    # latent_z_test_2d = latent_z_test.reshape(latent_z_test.shape[0], -1)
    # # t-SNE 시각화를 위한 잠재공간 정의
    # perplexity_train = min(30, len(latent_z_2d) - 1)
    # perplexity_test = min(30, len(latent_z_test_2d) - 1)

    # # t-SNE 시각화를 위한 잠재공간 정의
    # tsne_z = TSNE(n_components=2, perplexity=perplexity_train, random_state=42)
    # tsne_z_test = TSNE(n_components=2, perplexity=perplexity_test, random_state=42)

    # latent_tsne_z = tsne_z.fit_transform(latent_z_2d)
    # latent_tsne_z_test = tsne_z_test.fit_transform(latent_z_test_2d)

    # plt.figure(figsize=(12,6))
    # plt.scatter(latent_tsne_z[:len(latent_z), 0], latent_tsne_z[:len(latent_z), 1], label='Train Data Inference', alpha=0.6)
    # plt.scatter(latent_tsne_z_test[:len(latent_z_test), 0], latent_tsne_z_test[:len(latent_z_test), 1], label='Test Data Inference', alpha=0.6)
    # plt.title('Train vs. test t-SNE of Latent Space')
    # plt.ylabel('Y of latent Space')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('./figure/1008/tsne_latent_space.png', dpi=600)

    # print("추론한 결과 : 잠재공간 t-SNE 시각화")
    # plt.clf() # figure 초기화
    # plt.cla() # figure 축 초기화
    # plt.close() # 현재 figure 닫기

    ################################ 히스토그램을 이용한 손실값 시각화 ################################
    # LaTeX 스타일 활성화
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.figure(figsize=(12,6))
    plt.hist(mse, bins=1000, alpha=0.5, label='Train Data')
    plt.hist(mse_test, bins=200, alpha=0.5, label='Test Data')
    plt.legend(fontsize=12)

    # 축 라벨 및 파라미터 설정
    plt.title("Train vs. Test Recosntrucion Loss Histogram")
    plt.xlabel('MSE', fontsize=14, fontweight='bold')
    plt.ylabel('Loss Density', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.xlim(1e-6, 1)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))

    plt.savefig('./figure/1008/Histogram.png', dpi=600)

    print("추론한 결과 : 손실밀도 히스토그램 시각화")

    plt.clf() # figure 초기화
    plt.cla() # figure 축 초기화
    plt.close() # 현재 figure 닫기
    exit()
    ################################ 커널 밀도 추정(Kernel Density Estimation)을 이용한 손실값 시각화 ################################
    # LaTeX 스타일 활성화
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(12,6))
    # 커널 밀도 추정 그리기
    sns.kdeplot((mse), fill=True, label='Normal Loss', cut=0) # 학습한 데이터
    sns.kdeplot((mse_test), fill=True, label='Normal Loss', cut=0) # 테스트 데이터

    plt.legend(fontsize=12)
    plt.title("Train vs. Test Recosntrucion Loss Kernel Density Estimation")
    plt.xlabel('MSE', fontsize=14, fontweight='bold')
    plt.ylabel('Loss Density', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.xlim(1e-6, 1)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))

    plt.savefig('./figure/1008/KDEplot.png',dpi=600)

    print("추론 결과 : 손실값의 커널 밀도추정 시각화")

    plt.clf() # figure 초기화
    plt.cla() # figure 축 초기화
    plt.close() # 현재 figure 닫기

    ################################ 상자그림(Box Plot)을 이용한 손실값 시각화 ################################
    # LaTeX 스타일 활성화
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(12,6))
    data_train = pd.DataFrame({'MSE_train':mse, 'Type': 'Train Data'})
    data_test = pd.DataFrame({'MSE_test':mse_test, 'Type': 'Test Data'})
    
    # 박스 그림 그리기
    sns.boxplot(x='Type', y='MSE_train',data=data_train, label='Train Loss')
    sns.boxplot(x='Type', y='MSE_test',data=data_test, label='Test Loss')

    plt.legend(fontsize=12)

    # 박스 그림 축 및 라벨 설정
    plt.title("Train vs. Test Recosntrucion Loss Box plot")
    plt.xlabel('Type', fontsize=14, fontweight='bold')
    plt.ylabel('MSE', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.ylim(1e-6, 1)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))
    plt.savefig('./figure/1008/BOXplot.png',dpi=600)

    print("추론 결과 : 손실값의 박스 그림 시각화")
