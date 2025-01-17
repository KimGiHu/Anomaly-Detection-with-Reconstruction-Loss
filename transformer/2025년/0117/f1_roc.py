import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, confusion_matrix
import random
import seaborn as sns
import argparse
import os
from tqdm import tqdm
from utils import *
import math 
import pandas as pd

print(torch.cuda.is_available())
# cuda 캐시 정리
torch.cuda.empty_cache()

# argsparse를 이용한 다양한 데이터셋을 편하게 설정
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='./data/M02.npy', help='Enter the dataset path')
parser.add_argument('--model', type=str,
                    default='./model/1019/M01_single.pth', help='Enter the trained model path')
parser.add_argument('--case_num', type=int,
                    default='2', help='Enter the case number')
parser.add_argument('--batch_size', type=int,
                    default='64', help='Enter the batch size of dataloader')
parser.add_argument('--model_path', type=str,
                    default='./transformer/model/', help='Enter the dataset path')
parser.add_argument('--figure_path', type=str,
                    default='./figure/JKPS/1217', help='Enter the dataset path')
args = parser.parse_args()

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

################################ Figure 저장할 디렉토리 생성하기 ################################
createDirectory(args.figure_path)

# 커스텀 Transformer Encoder Layer  
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)
        self.attn_weights = None  # Attention 가중치를 저장할 변수

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask, need_weights=True)
        self.attn_weights = attn_weights  # 가중치 저장
        src = src + self.dropout1(src2) # residual term 더해서 계산
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5003):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        # 입력 x의 크기: (batch_size, seq_len, feature_dim)
        batch_size, seq_len, feature_dim = x.size()

        # Positional Encoding 크기 확인 및 조정
        if feature_dim != self.d_model:
            raise ValueError(f"Input feature_dim ({feature_dim}) does not match PositionalEncoding d_model ({self.d_model})")

        if seq_len > self.encoding.size(1):
            raise ValueError(f"Input seq_len ({seq_len}) exceeds PositionalEncoding max_len ({self.encoding.size(1)})")

        positional_encoding = self.encoding[:, :seq_len, :].to(x.device)
        return x + positional_encoding

# Variational Autoencoder (VAE) 정의
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim, num_heads=4, num_layers=3, dropout_prob=0.1):
        super(VAE, self).__init__()
        
        # 커스텀 인코더 레이어 정의
              
        # Fully-Connected Layer : FC to Encoder
        self.fc1_input_to_latent = nn.Linear(hidden_dim, 4*latent_dim)
        self.fc2_input_to_latent = nn.Linear(4*latent_dim,2*latent_dim)
        self.fc3_input_to_latent = nn.Linear(2*latent_dim,latent_dim)

        # Positional Encoding 
        self.encoder_positional_encoding = PositionalEncoding(d_model=latent_dim)
        self.decoder_positional_encoding = PositionalEncoding(latent_dim)

        # Transformer's Encoder
        encoder_layer = CustomTransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dropout=dropout_prob)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(latent_dim+condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim+condition_dim, latent_dim)
        
        # Transformer's Decoder
        self.fc_latent_to_input = nn.Linear(latent_dim + condition_dim, latent_dim)
        self.fc_decoder_input = nn.Linear(latent_dim, latent_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dropout=dropout_prob)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Fully-Connected Layer : Decoder to FC
        self.fc1_output = nn.Linear(latent_dim, 2*latent_dim)
        self.fc2_output = nn.Linear(2*latent_dim, 4*latent_dim)
        self.fc3_output = nn.Linear(4*latent_dim, hidden_dim)
    
    def encode(self, x, c):
        x = self.fc1_input_to_latent(x)
        x = self.fc2_input_to_latent(x)
        x = self.fc3_input_to_latent(x)
        
        x = self.encoder_positional_encoding(x)

        x = x.transpose(0, 1)  # (seq_len, batch_size, feature_dim) 형태로 맞춤
        h = self.encoder(x).transpose(0, 1)  # Transformer Encoder 적용 후 차원 변환
        h = torch.cat([h, c.unsqueeze(1).expand(-1, h.size(1), -1)], dim=-1)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        # Latent space와 조건을 결합 (c를 시퀀스 차원에 맞게 반복하여 결합)
        c_expanded = c.unsqueeze(1).expand(-1, z.size(1), -1)  # c를 (batch_size, seq_len, condition_dim) 형태로 확장
        z = torch.cat([z, c_expanded], dim=-1)
        
        z = self.fc_latent_to_input(z)  # Latent space에서 input_dim으로 변환
        z = self.fc_decoder_input(z)

        z = self.decoder_positional_encoding(z)

        z = z.transpose(0, 1)  # (seq_len, batch_size, feature_dim) 형태로 맞춤
        x_recon = self.decoder(z).transpose(0, 1).squeeze(1)  # Transformer Decoder 적용 후 차원 축소
        x_recon = self.fc1_output(x_recon)
        x_recon = self.fc2_output(x_recon)
        return self.fc3_output(x_recon)
    
    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, c), mu, log_var

    def get_attention_weights(self):
        # Transformer Encoder 레이어에서 Attention 가중치 추출
        attn_weights = [layer.attn_weights for layer in self.encoder.layers]
        return attn_weights

# 학습 중 Attention 가중치 시각화 예제
def visualize_attention_weights(attn_weights, layer_idx=0, head_idx=0):
    if attn_weights is not None:
        attn_weights = attn_weights[layer_idx]  # 특정 레이어의 Attention 가중치 선택
        attn_weights = attn_weights[head_idx].cpu().detach().numpy()  # 특정 헤드의 가중치 선택
        plt.imshow(attn_weights, cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.savefig('./transformer.png', dpi=600)

# 손실 함수 정의 - my proposal
def loss_function_sum(x_recon, x, mu, logvar):
    MSE_lib = nn.functional.mse_loss(x_recon, x, reduction='sum') # divided into batch size, time steps
    KLD = -0.5 * torch.sum(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)# see Appendix B from VAE paper:
                                                                                           # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    return MSE_lib + KLD + 1e-12

# 모델의 체크포인트 저장
def save_checkpoint(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    # print(f"Checkpoint saved at epoch {epoch}")

# 모델의 체크포인트 불러오기
def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    # print(f"Checkpoint loaded. Resuming from epoch {epoch} with loss {loss:.4f}")
    return epoch, loss

# npy file path setting
npy_file_path = os.getcwd()
M01_old = np.load(os.path.join(npy_file_path, 'data/M01_old.npy'))
Module_data = np.load(os.path.join(npy_file_path, args.data_path))

# time steps : 5002 points
time = np.arange(Module_data.shape[1]) * (0.4)  # 5000 points = 2000 us

# number of channels
channels = ['C1', 'C2', 'C3', 'C4']
num_channels = len(channels)

## Modulator 1번(신버전), 2번, 4번 로드
if args.data_path == 'data/M01.npy':
    if ((((int((M01_old.shape[0])*0.8))+(int((Module_data.shape[0])*0.8))) % 8) == 1) :
        data_input = np.concatenate((M01_old[:int((M01_old.shape[0])*0.8)-1,:,:],Module_data[:int((Module_data.shape[0])*0.8)-1,:,:]), axis=0)
    else :
        data_input = np.concatenate((M01_old[:int((M01_old.shape[0])*0.8),:,:],Module_data[:int((Module_data.shape[0])*0.8),:,:]), axis=0)

else :
    if (((int((Module_data.shape[0])*0.8)) % 8) == 1) :
        data_input = Module_data[:int((Module_data.shape[0])*0.8)-1,:,:]
    else :
        data_input = Module_data[:int((Module_data.shape[0])*0.8),:,:]

# Min-Max 스케일링
scaler_P1 = MinMaxScaler() 
scaler_P2 = MinMaxScaler() 
scaler_P3 = MinMaxScaler() 
scaler_P4 = MinMaxScaler()

import joblib # 파일이름과 fitting된 스케일러 변수 저장
scaler_filename = ['channel1','channel2','channel3','channel4']
for i in range(0, num_channels):
    if i==0:
        data_input[:,:,i] = scaler_P1.fit_transform(data_input[:,:,i].reshape(-1,1)).reshape(data_input[:,:,i].shape)
        joblib.dump(scaler_P1, scaler_filename[i])
    if i==1:
        data_input[:,:,i] = scaler_P2.fit_transform(data_input[:,:,i].reshape(-1,1)).reshape(data_input[:,:,i].shape)
        joblib.dump(scaler_P2, scaler_filename[i])
    if i==2:
        data_input[:,:,i] = scaler_P3.fit_transform(data_input[:,:,i].reshape(-1,1)).reshape(data_input[:,:,i].shape)
        joblib.dump(scaler_P3, scaler_filename[i])
    if i==3:
        data_input[:,:,i] = scaler_P4.fit_transform(data_input[:,:,i].reshape(-1,1)).reshape(data_input[:,:,i].shape)
        joblib.dump(scaler_P4, scaler_filename[i])

# 입력 형태 (N, 5002, 4)를 (N, 4, 5002)로 변경함.
data_input = data_input.transpose(0,2,1) 

# numpy 데이터를 tensor 위에 올림
data_tensor = torch.tensor(data_input, dtype=torch.float32) 

# 원-핫 벡터 생성
one_hot_vector = np.zeros(4)

case = args.case_num
one_hot_vector[case-1] = 1
labels_one_hot = np.tile(one_hot_vector, (data_input.shape[0], 1))
labels_one_hot = torch.tensor(labels_one_hot, dtype=torch.float32)

# DataLoader의 생성자 시드 정의
g = torch.Generator()
g.manual_seed(seed)

# 학습 데이터세트 및 데이터로더 정의
dataset1 = CustomDataset(data_tensor, labels_one_hot)
dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, generator=g)

# 하이퍼 파라미터 설정
input_dim = 4
hidden_dim = 5002
latent_dim = 512
condition_dim = 4
num_heads = 8
num_layers = 4

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_VAE = VAE(input_dim, hidden_dim, latent_dim, condition_dim, num_heads, num_layers).to(device)
optimizer = optim.Adam(model_VAE.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, verbose=True) # 3회 이상 개선점이 없으면 스케쥴러 업데이트, 그 폭은 20%씩 바뀜, 업데이트하는 것 시각화활성화

# 라이브러리 손실함수
criterion = nn.MSELoss(reduction='mean')  # 복원손실 값으로 MSE 사용

# 테스트 데이터
if (((int((Module_data.shape[0])*0.8)) % 8) == 1) :
    data_test = Module_data[int((Module_data.shape[0])*0.8)-1:,:,:]
else :
    data_test = Module_data[int((Module_data.shape[0])*0.8):,:,:]

# ########### 1. Original Anomaly Signal ###########
# # npy file path setting
# npy_file_path = os.getcwd() 
# M02_anomaly = np.load(os.path.join(npy_file_path, 'data/M02_anomaly.npy'), allow_pickle=True)

# anomaly_input = M02_anomaly[:,1:]
# print(f'the shape of anomaly data at June.02.2023 : {anomaly_input.shape}')

# # pulse on/off 되는 타이밍을 고려 : 신호가 들어오는 1000번째부터 나가는 4000번째까지 선택
# # 신호가 들어오는 구간 전체를 선택
# processed_input = anomaly_input[:, :]
# print(f'The shape of anomaly data after selecting pulse signal region: {processed_input.shape}')

# # 데이터가 부족한 경우 외삽을 통해 데이터를 확장
# required_length = 100040
# current_length = processed_input.shape[0]

# if current_length < required_length:
#     num_missing_rows = required_length - current_length
#     num_missing_rows_each_side = num_missing_rows // 2

#     # 앞쪽과 뒤쪽에 데이터를 반복하여 확장
#     front_extension = processed_input[:num_missing_rows_each_side, :]
#     front_extension_less = processed_input[:int((num_missing_rows - num_missing_rows_each_side)*13/10), :]
#     back_extension = processed_input[-(num_missing_rows - num_missing_rows_each_side):, :]
#     back_extension_less = processed_input[-int((num_missing_rows - num_missing_rows_each_side)*1/10):, :]
#     # 데이터를 앞과 뒤에 반복된 데이터를 추가하여 확장
#     processed_input = np.vstack([ front_extension_less,
#                                   front_extension,
#                                   processed_input,
#                                   back_extension,
#                                   back_extension_less])
    
# # DataFrame으로 변환 (각 열이 채널이 되도록 설정)
# df = pd.DataFrame(processed_input[:,:])

# # 이동평균 적용( 윈도우 크기 설정 : 96062개의 샘플을 5002개로 줄이기 위한 window 크기 설정)
# window_size = max(1, int(len(df) / 5002))  # 최소 윈도우 크기를 1로 설정하여 오류 방지
# rolloing_mean_df = df.rolling(window=window_size, min_periods=1).mean()

# # 윈도우 사이즈 간격으로 다운 샘플링 ( 5002개 샘플만을 남김 )
# downsampled_df = rolloing_mean_df.iloc[::window_size].reset_index(drop=True)
# downsampled_df = downsampled_df[:5002]

# # 결과 넘파이 배열로 변환
# data_anomaly_origin = downsampled_df.values
# print(data_anomaly_origin.shape)
# data_anomaly_origin = data_anomaly_origin.reshape(1, data_anomaly_origin.shape[0], data_anomaly_origin.shape[1])

# ########## 2. Generated Anomaly Signal ###########
# data_nosiy = data_test[2:3,:,:]

# # # 일부 구간 에러 비율조정.
# changed_ratio = 0.50 

# for i in range (0,1000):
#     data_nosiy[:,3000+i:3000+(i+1),0] = data_nosiy[:,3000+i:3000+(i+1),0] - changed_ratio*data_nosiy[:,3000+i:3000+(i+1),0]
#     data_nosiy[:,3000+i:3000+(i+1),1] = data_nosiy[:,3000+i:3000+(i+1),1] - changed_ratio*data_nosiy[:,3000+i:3000+(i+1),1]
#     data_nosiy[:,3000+i:3000+(i+1),2] = data_nosiy[:,3000+i:3000+(i+1),2] - changed_ratio*data_nosiy[:,3000+i:3000+(i+1),2]
#     data_nosiy[:,3000+i:3000+(i+1),3] = data_nosiy[:,3000+i:3000+(i+1),3] - changed_ratio*data_nosiy[:,3000+i:3000+(i+1),3] 

# data_anomaly = data_nosiy

# error 비율 0 퍼센트
# data_anomaly = data_test[100:101,:,:]

# numpy 파일로 저장할 경로 설정
anomaly_file_path = './data/anomaly_gausssian_sampel1500_multi_variance.npy'

data_anomaly = np.load(anomaly_file_path)
augment_size = 1500

data_anomaly_0 = data_anomaly[:augment_size*1,:,:]
data_anomaly_1st = data_anomaly[augment_size*1:augment_size*2,:,:]
data_anomaly_2nd = data_anomaly[augment_size*2:augment_size*3,:,:]
data_anomaly_3rd = data_anomaly[augment_size*3:augment_size*4,:,:]
data_anomaly_4th = data_anomaly[augment_size*4:augment_size*5,:,:]
data_anomaly_5th = data_anomaly[augment_size*5:augment_size*6,:,:]
data_anomaly_6th = data_anomaly[augment_size*6:augment_size*7,:,:]
data_anomaly_7th = data_anomaly[augment_size*7:augment_size*8,:,:]
data_anomaly_8th = data_anomaly[augment_size*8:augment_size*9,:,:]
data_anomaly_9th = data_anomaly[augment_size*9:augment_size*10,:,:]
data_anomaly_10th = data_anomaly[augment_size*10:augment_size*11,:,:]

# 스케일러 불러오기

load_channel1 = joblib.load('channel1')
load_channel2 = joblib.load('channel2')
load_channel3 = joblib.load('channel3')
load_channel4 = joblib.load('channel4')

#### M02 테스트 정상신호와 이상신호 스케일링 ####
for i in range(0, num_channels):
    if i==0:
        data_test[:,:,i] = load_channel1.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)
        # data_anomaly_origin[:,:,i] = load_channel1.transform(data_anomaly_origin[:,:,i].reshape(-1,1)).reshape(data_anomaly_origin[:,:,i].shape)
        data_anomaly_0[:,:,i] = load_channel1.transform(data_anomaly_0[:,:,i].reshape(-1,1)).reshape(data_anomaly_0[:,:,i].shape)
        data_anomaly_1st[:,:,i] = load_channel1.transform(data_anomaly_1st[:,:,i].reshape(-1,1)).reshape(data_anomaly_1st[:,:,i].shape)
        data_anomaly_2nd[:,:,i] = load_channel1.transform(data_anomaly_2nd[:,:,i].reshape(-1,1)).reshape(data_anomaly_2nd[:,:,i].shape)
        data_anomaly_3rd[:,:,i] = load_channel1.transform(data_anomaly_3rd[:,:,i].reshape(-1,1)).reshape(data_anomaly_3rd[:,:,i].shape)
        data_anomaly_4th[:,:,i] = load_channel1.transform(data_anomaly_4th[:,:,i].reshape(-1,1)).reshape(data_anomaly_4th[:,:,i].shape)
        data_anomaly_5th[:,:,i] = load_channel1.transform(data_anomaly_5th[:,:,i].reshape(-1,1)).reshape(data_anomaly_5th[:,:,i].shape)
        data_anomaly_6th[:,:,i] = load_channel1.transform(data_anomaly_6th[:,:,i].reshape(-1,1)).reshape(data_anomaly_6th[:,:,i].shape)
        data_anomaly_7th[:,:,i] = load_channel1.transform(data_anomaly_7th[:,:,i].reshape(-1,1)).reshape(data_anomaly_7th[:,:,i].shape)
        data_anomaly_8th[:,:,i] = load_channel1.transform(data_anomaly_8th[:,:,i].reshape(-1,1)).reshape(data_anomaly_8th[:,:,i].shape)
        data_anomaly_9th[:,:,i] = load_channel1.transform(data_anomaly_9th[:,:,i].reshape(-1,1)).reshape(data_anomaly_9th[:,:,i].shape)
        data_anomaly_10th[:,:,i] = load_channel1.transform(data_anomaly_10th[:,:,i].reshape(-1,1)).reshape(data_anomaly_10th[:,:,i].shape)

    if i==1:
        data_test[:,:,i] = load_channel2.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)
        # data_anomaly_origin[:,:,i] = load_channel2.transform(data_anomaly_origin[:,:,i].reshape(-1,1)).reshape(data_anomaly_origin[:,:,i].shape)
        data_anomaly_0[:,:,i] = load_channel2.transform(data_anomaly_0[:,:,i].reshape(-1,1)).reshape(data_anomaly_0[:,:,i].shape)
        data_anomaly_1st[:,:,i] = load_channel2.transform(data_anomaly_1st[:,:,i].reshape(-1,1)).reshape(data_anomaly_1st[:,:,i].shape)
        data_anomaly_2nd[:,:,i] = load_channel2.transform(data_anomaly_2nd[:,:,i].reshape(-1,1)).reshape(data_anomaly_2nd[:,:,i].shape)
        data_anomaly_3rd[:,:,i] = load_channel2.transform(data_anomaly_3rd[:,:,i].reshape(-1,1)).reshape(data_anomaly_3rd[:,:,i].shape)
        data_anomaly_4th[:,:,i] = load_channel2.transform(data_anomaly_4th[:,:,i].reshape(-1,1)).reshape(data_anomaly_4th[:,:,i].shape)
        data_anomaly_5th[:,:,i] = load_channel2.transform(data_anomaly_5th[:,:,i].reshape(-1,1)).reshape(data_anomaly_5th[:,:,i].shape)
        data_anomaly_6th[:,:,i] = load_channel2.transform(data_anomaly_6th[:,:,i].reshape(-1,1)).reshape(data_anomaly_6th[:,:,i].shape)
        data_anomaly_7th[:,:,i] = load_channel2.transform(data_anomaly_7th[:,:,i].reshape(-1,1)).reshape(data_anomaly_7th[:,:,i].shape)
        data_anomaly_8th[:,:,i] = load_channel2.transform(data_anomaly_8th[:,:,i].reshape(-1,1)).reshape(data_anomaly_8th[:,:,i].shape)
        data_anomaly_9th[:,:,i] = load_channel2.transform(data_anomaly_9th[:,:,i].reshape(-1,1)).reshape(data_anomaly_9th[:,:,i].shape)
        data_anomaly_10th[:,:,i] = load_channel2.transform(data_anomaly_10th[:,:,i].reshape(-1,1)).reshape(data_anomaly_10th[:,:,i].shape)

    if i==2:
        data_test[:,:,i] = load_channel3.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)
        # data_anomaly_origin[:,:,i] = load_channel3.transform(data_anomaly_origin[:,:,i].reshape(-1,1)).reshape(data_anomaly_origin[:,:,i].shape)
        data_anomaly_0[:,:,i] = load_channel3.transform(data_anomaly_0[:,:,i].reshape(-1,1)).reshape(data_anomaly_0[:,:,i].shape)
        data_anomaly_1st[:,:,i] = load_channel3.transform(data_anomaly_1st[:,:,i].reshape(-1,1)).reshape(data_anomaly_1st[:,:,i].shape)
        data_anomaly_2nd[:,:,i] = load_channel3.transform(data_anomaly_2nd[:,:,i].reshape(-1,1)).reshape(data_anomaly_2nd[:,:,i].shape)
        data_anomaly_3rd[:,:,i] = load_channel3.transform(data_anomaly_3rd[:,:,i].reshape(-1,1)).reshape(data_anomaly_3rd[:,:,i].shape)
        data_anomaly_4th[:,:,i] = load_channel3.transform(data_anomaly_4th[:,:,i].reshape(-1,1)).reshape(data_anomaly_4th[:,:,i].shape)
        data_anomaly_5th[:,:,i] = load_channel3.transform(data_anomaly_5th[:,:,i].reshape(-1,1)).reshape(data_anomaly_5th[:,:,i].shape)
        data_anomaly_6th[:,:,i] = load_channel3.transform(data_anomaly_6th[:,:,i].reshape(-1,1)).reshape(data_anomaly_6th[:,:,i].shape)
        data_anomaly_7th[:,:,i] = load_channel3.transform(data_anomaly_7th[:,:,i].reshape(-1,1)).reshape(data_anomaly_7th[:,:,i].shape)
        data_anomaly_8th[:,:,i] = load_channel3.transform(data_anomaly_8th[:,:,i].reshape(-1,1)).reshape(data_anomaly_8th[:,:,i].shape)
        data_anomaly_9th[:,:,i] = load_channel3.transform(data_anomaly_9th[:,:,i].reshape(-1,1)).reshape(data_anomaly_9th[:,:,i].shape)
        data_anomaly_10th[:,:,i] = load_channel3.transform(data_anomaly_10th[:,:,i].reshape(-1,1)).reshape(data_anomaly_10th[:,:,i].shape)

    if i==3:
        data_test[:,:,i] = load_channel4.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)
        # data_anomaly_origin[:,:,i] = load_channel4.transform(data_anomaly_origin[:,:,i].reshape(-1,1)).reshape(data_anomaly_origin[:,:,i].shape)
        data_anomaly_0[:,:,i] = load_channel4.transform(data_anomaly_0[:,:,i].reshape(-1,1)).reshape(data_anomaly_0[:,:,i].shape)
        data_anomaly_1st[:,:,i] = load_channel4.transform(data_anomaly_1st[:,:,i].reshape(-1,1)).reshape(data_anomaly_1st[:,:,i].shape)
        data_anomaly_2nd[:,:,i] = load_channel4.transform(data_anomaly_2nd[:,:,i].reshape(-1,1)).reshape(data_anomaly_2nd[:,:,i].shape)
        data_anomaly_3rd[:,:,i] = load_channel4.transform(data_anomaly_3rd[:,:,i].reshape(-1,1)).reshape(data_anomaly_3rd[:,:,i].shape)
        data_anomaly_4th[:,:,i] = load_channel4.transform(data_anomaly_4th[:,:,i].reshape(-1,1)).reshape(data_anomaly_4th[:,:,i].shape)
        data_anomaly_5th[:,:,i] = load_channel4.transform(data_anomaly_5th[:,:,i].reshape(-1,1)).reshape(data_anomaly_5th[:,:,i].shape)
        data_anomaly_6th[:,:,i] = load_channel4.transform(data_anomaly_6th[:,:,i].reshape(-1,1)).reshape(data_anomaly_6th[:,:,i].shape)
        data_anomaly_7th[:,:,i] = load_channel4.transform(data_anomaly_7th[:,:,i].reshape(-1,1)).reshape(data_anomaly_7th[:,:,i].shape)
        data_anomaly_8th[:,:,i] = load_channel4.transform(data_anomaly_8th[:,:,i].reshape(-1,1)).reshape(data_anomaly_8th[:,:,i].shape)
        data_anomaly_9th[:,:,i] = load_channel4.transform(data_anomaly_9th[:,:,i].reshape(-1,1)).reshape(data_anomaly_9th[:,:,i].shape)
        data_anomaly_10th[:,:,i] = load_channel4.transform(data_anomaly_10th[:,:,i].reshape(-1,1)).reshape(data_anomaly_10th[:,:,i].shape)


# 입력 형태 (N, 5002, 4)를 (N, 4, 5002)로 변경함.
data_test = data_test.transpose(0,2,1)
# data_anomaly_origin = data_anomaly_origin.transpose(0,2,1)
data_anomaly_0 = data_anomaly_0.transpose(0,2,1)
data_anomaly_1st = data_anomaly_1st.transpose(0,2,1)
data_anomaly_2nd = data_anomaly_2nd.transpose(0,2,1)
data_anomaly_3rd = data_anomaly_3rd.transpose(0,2,1)
data_anomaly_4th = data_anomaly_4th.transpose(0,2,1)
data_anomaly_5th = data_anomaly_5th.transpose(0,2,1)
data_anomaly_6th = data_anomaly_6th.transpose(0,2,1)
data_anomaly_7th = data_anomaly_7th.transpose(0,2,1)
data_anomaly_8th = data_anomaly_8th.transpose(0,2,1)
data_anomaly_9th = data_anomaly_9th.transpose(0,2,1)
data_anomaly_10th = data_anomaly_10th.transpose(0,2,1)

# numpy 데이터를 tensor 위에 올림
test_tensor = torch.tensor(data_test, dtype=torch.float32)
# anomaly_origin_tensor = torch.tensor(data_anomaly_origin, dtype=torch.float32)
anomaly_0_tensor = torch.tensor(data_anomaly_0, dtype=torch.float32)
anomaly_1st_tensor = torch.tensor(data_anomaly_1st, dtype=torch.float32)
anomaly_1st_tensor = torch.tensor(data_anomaly_1st, dtype=torch.float32)
anomaly_2nd_tensor = torch.tensor(data_anomaly_2nd, dtype=torch.float32)
anomaly_3rd_tensor = torch.tensor(data_anomaly_3rd, dtype=torch.float32)
anomaly_4th_tensor = torch.tensor(data_anomaly_4th, dtype=torch.float32)
anomaly_5th_tensor = torch.tensor(data_anomaly_5th, dtype=torch.float32)
anomaly_6th_tensor = torch.tensor(data_anomaly_6th, dtype=torch.float32)
anomaly_7th_tensor = torch.tensor(data_anomaly_7th, dtype=torch.float32)
anomaly_8th_tensor = torch.tensor(data_anomaly_8th, dtype=torch.float32)
anomaly_9th_tensor = torch.tensor(data_anomaly_9th, dtype=torch.float32)
anomaly_10th_tensor = torch.tensor(data_anomaly_10th, dtype=torch.float32)

# 원-핫 벡터 레이블 생성
one_hot_vector = np.zeros(4)
case = args.case_num #  unique 파형 4개
one_hot_vector[case-1] = 1

# 테스트, 원본 이상신호, 생성된 랜덤 가우시안 이상신호
test_labels_one_hot = np.tile(one_hot_vector, (data_test.shape[0], 1))
anomaly_labels_one_not = np.tile(one_hot_vector, (data_anomaly_1st.shape[0], 1))
# origin_anomaly_label = np.tile(one_hot_vector, (data_anomaly_origin.shape[0], 1))

# 레이블 리스트를 tensor위에 두기
test_labels_one_hot = torch.tensor(test_labels_one_hot, dtype=torch.float32)
anomaly_labels_one_not = torch.tensor(anomaly_labels_one_not, dtype=torch.float32)
# origin_anomaly_label = torch.tensor(origin_anomaly_label, dtype=torch.float32)

# 테스트 데이터세트 및 데이터로드 정의
dataset_test = CustomDataset(test_tensor, test_labels_one_hot)
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, generator=g)

# 이상신호 데이터세트 및 데이터로드 정의

# # 원본 이상신호 
# dataset_origin_anomaly = CustomDataset(anomaly_origin_tensor,origin_anomaly_label)
# dataloader_anomaly_origin = DataLoader(dataset_origin_anomaly, batch_size=args.batch_size, shuffle=False, generator=g)

# 가우시안 노이즈 그룹 1 : 평균 0, 분산 0.001
dataset_anomaly_0 = CustomDataset(anomaly_0_tensor, anomaly_labels_one_not)
dataloader_anomaly_0 = DataLoader(dataset_anomaly_0, batch_size=args.batch_size, shuffle=False, generator=g)

# 가우시안 노이즈 그룹 1 : 평균 0, 분산 0.01
dataset_anomaly_1st = CustomDataset(anomaly_1st_tensor, anomaly_labels_one_not)
dataloader_anomaly_1st = DataLoader(dataset_anomaly_1st, batch_size=args.batch_size, shuffle=False, generator=g)

# 가우시안 노이즈 그룹 2 : 평균 0, 분산 0.02
dataset_anomaly_2nd = CustomDataset(anomaly_2nd_tensor, anomaly_labels_one_not)
dataloader_anomaly_2nd = DataLoader(dataset_anomaly_2nd, batch_size=args.batch_size, shuffle=False, generator=g)

# 가우시안 노이즈 그룹 3 : 평균 0, 분산 0.03
dataset_anomaly_3rd = CustomDataset(anomaly_3rd_tensor, anomaly_labels_one_not)
dataloader_anomaly_3rd = DataLoader(dataset_anomaly_3rd, batch_size=args.batch_size, shuffle=False, generator=g)

# 가우시안 노이즈 그룹 4 : 평균 0, 분산 0.04
dataset_anomaly_4th = CustomDataset(anomaly_4th_tensor, anomaly_labels_one_not)
dataloader_anomaly_4th = DataLoader(dataset_anomaly_4th, batch_size=args.batch_size, shuffle=False, generator=g)

# 가우시안 노이즈 그룹 5 : 평균 0, 분산 0.05
dataset_anomaly_5th = CustomDataset(anomaly_5th_tensor, anomaly_labels_one_not)
dataloader_anomaly_5th = DataLoader(dataset_anomaly_5th, batch_size=args.batch_size, shuffle=False, generator=g)

# 가우시안 노이즈 그룹 6 : 평균 0, 분산 0.06
dataset_anomaly_6th = CustomDataset(anomaly_6th_tensor, anomaly_labels_one_not)
dataloader_anomaly_6th = DataLoader(dataset_anomaly_6th, batch_size=args.batch_size, shuffle=False, generator=g)

# 가우시안 노이즈 그룹 7 : 평균 0, 분산 0.07
dataset_anomaly_7th = CustomDataset(anomaly_7th_tensor, anomaly_labels_one_not)
dataloader_anomaly_7th = DataLoader(dataset_anomaly_7th, batch_size=args.batch_size, shuffle=False, generator=g)

# 가우시안 노이즈 그룹 8 : 평균 0, 분산 0.08
dataset_anomaly_8th = CustomDataset(anomaly_8th_tensor, anomaly_labels_one_not)
dataloader_anomaly_8th = DataLoader(dataset_anomaly_8th, batch_size=args.batch_size, shuffle=False, generator=g)

# 가우시안 노이즈 그룹 9 : 평균 0, 분산 0.09
dataset_anomaly_9th = CustomDataset(anomaly_9th_tensor, anomaly_labels_one_not)
dataloader_anomaly_9th = DataLoader(dataset_anomaly_9th, batch_size=args.batch_size, shuffle=False, generator=g)

# 가우시안 노이즈 그룹 10 : 평균 0, 분산 0.10
dataset_anomaly_10th = CustomDataset(anomaly_10th_tensor, anomaly_labels_one_not)
dataloader_anomaly_10th = DataLoader(dataset_anomaly_10th, batch_size=args.batch_size, shuffle=False, generator=g)

# 베스트 모델 불러오기 
model_VAE.load_state_dict(torch.load('./transformer/best_model.pth'))

# 모델 평가 플래그 세팅
model_VAE.eval()

# 모델 추론 및 MSE 결과저장하기
with torch.no_grad():
    mse = []
    latent_mu = []
    latent_logvar = []
    latent_z = []

    pbar = tqdm(dataloader1, total=len(dataloader1), ncols=100)

    for x_input, label in pbar :
        x_input, label = x_input.to(device), label.to(device)
        
        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)
        
        mse.extend(((origin - regenerate)**2).mean(axis=(1,2)))

        latent_z.append(z.cpu().numpy())
    
with torch.no_grad():
    mse_test = []
    latent_mu_test = []
    latent_logvar_test = []
    latent_z_test = []

    pbar_test = tqdm(dataloader_test, total=len(dataloader_test), ncols=100)

    for x_input, label in pbar_test :
        x_input, label = x_input.to(device), label.to(device)

        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

        mse_test.extend(((origin - regenerate)**2).mean(axis=(1,2)))
        latent_z_test.append(z.cpu().numpy())

with torch.no_grad():
    mse_anomaly_0 = []
    mse_anomaly_1st = []
    mse_anomaly_2nd = []
    mse_anomaly_3rd = []
    mse_anomaly_4th = []
    mse_anomaly_5th = []
    mse_anomaly_6th = []
    mse_anomaly_7th = []
    mse_anomaly_8th = []
    mse_anomaly_9th = []
    mse_anomaly_10th = []
    # latent_mu_anomaly = []
    # latent_logvar_anomaly = []
    latent_z_anomaly_0 = []
    latent_z_anomaly_1st = []
    latent_z_anomaly_2nd = []
    latent_z_anomaly_3rd = []
    latent_z_anomaly_4th = []
    latent_z_anomaly_5th = []
    latent_z_anomaly_6th = []
    latent_z_anomaly_7th = []
    latent_z_anomaly_8th = []
    latent_z_anomaly_9th = []
    latent_z_anomaly_10th = []

    pbar_anomaly_0 = tqdm(dataloader_anomaly_0, total=len(dataloader_anomaly_0), ncols=100)

    for x_input, label in pbar_anomaly_0 :
        x_input, label = x_input.to(device), label.to(device)

        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

        mse_anomaly_0.extend(((origin - regenerate)**2).mean(axis=(1,2)))
        latent_z_anomaly_0.append(z.cpu().numpy())

    pbar_anomaly_1st = tqdm(dataloader_anomaly_1st, total=len(dataloader_anomaly_1st), ncols=100)

    for x_input, label in pbar_anomaly_1st :
        x_input, label = x_input.to(device), label.to(device)

        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

        mse_anomaly_1st.extend(((origin - regenerate)**2).mean(axis=(1,2)))
        latent_z_anomaly_1st.append(z.cpu().numpy())

    pbar_anomaly_2nd = tqdm(dataloader_anomaly_2nd, total=len(dataloader_anomaly_2nd), ncols=100)

    for x_input, label in pbar_anomaly_2nd :
        x_input, label = x_input.to(device), label.to(device)

        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

        mse_anomaly_2nd.extend(((origin - regenerate)**2).mean(axis=(1,2)))
        latent_z_anomaly_2nd.append(z.cpu().numpy())

    pbar_anomaly_3rd = tqdm(dataloader_anomaly_3rd, total=len(dataloader_anomaly_3rd), ncols=100)

    for x_input, label in pbar_anomaly_3rd :
        x_input, label = x_input.to(device), label.to(device)

        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

        mse_anomaly_3rd.extend(((origin - regenerate)**2).mean(axis=(1,2)))
        latent_z_anomaly_3rd.append(z.cpu().numpy())
    
    pbar_anomaly_4th = tqdm(dataloader_anomaly_4th, total=len(dataloader_anomaly_4th), ncols=100)

    for x_input, label in pbar_anomaly_4th :
        x_input, label = x_input.to(device), label.to(device)

        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

        mse_anomaly_4th.extend(((origin - regenerate)**2).mean(axis=(1,2)))
        latent_z_anomaly_4th.append(z.cpu().numpy())
    
    pbar_anomaly_5th = tqdm(dataloader_anomaly_5th, total=len(dataloader_anomaly_5th), ncols=100)

    for x_input, label in pbar_anomaly_5th :
        x_input, label = x_input.to(device), label.to(device)

        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

        mse_anomaly_5th.extend(((origin - regenerate)**2).mean(axis=(1,2)))
        latent_z_anomaly_5th.append(z.cpu().numpy())

    pbar_anomaly_6th = tqdm(dataloader_anomaly_6th, total=len(dataloader_anomaly_6th), ncols=100)

    for x_input, label in pbar_anomaly_6th :
        x_input, label = x_input.to(device), label.to(device)

        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

        mse_anomaly_6th.extend(((origin - regenerate)**2).mean(axis=(1,2)))
        latent_z_anomaly_6th.append(z.cpu().numpy())
    
    pbar_anomaly_7th = tqdm(dataloader_anomaly_7th, total=len(dataloader_anomaly_7th), ncols=100)

    for x_input, label in pbar_anomaly_7th :
        x_input, label = x_input.to(device), label.to(device)

        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

        mse_anomaly_7th.extend(((origin - regenerate)**2).mean(axis=(1,2)))
        latent_z_anomaly_7th.append(z.cpu().numpy())
    
    pbar_anomaly_8th = tqdm(dataloader_anomaly_8th, total=len(dataloader_anomaly_8th), ncols=100)

    for x_input, label in pbar_anomaly_8th :
        x_input, label = x_input.to(device), label.to(device)

        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

        mse_anomaly_8th.extend(((origin - regenerate)**2).mean(axis=(1,2)))
        latent_z_anomaly_8th.append(z.cpu().numpy())
    
    pbar_anomaly_9th = tqdm(dataloader_anomaly_9th, total=len(dataloader_anomaly_9th), ncols=100)

    for x_input, label in pbar_anomaly_9th :
        x_input, label = x_input.to(device), label.to(device)

        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

        mse_anomaly_9th.extend(((origin - regenerate)**2).mean(axis=(1,2)))
        latent_z_anomaly_9th.append(z.cpu().numpy())
    
    pbar_anomaly_10th = tqdm(dataloader_anomaly_10th, total=len(dataloader_anomaly_10th), ncols=100)

    for x_input, label in pbar_anomaly_10th :
        x_input, label = x_input.to(device), label.to(device)

        # 모델 출력값 받아오기
        x_recon, mu, logvar = model_VAE(x_input, label)
        z = model_VAE.reparameterize(mu, logvar)

        origin = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()

        origin = origin.transpose(0,2,1)
        regenerate = regenerate.transpose(0,2,1)

        for i in range(0, num_channels):
            if i==0:
                origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==1:
                origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==2:
                origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

            if i==3:
                origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
                regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

        mse_anomaly_10th.extend(((origin - regenerate)**2).mean(axis=(1,2)))
        latent_z_anomaly_10th.append(z.cpu().numpy())

mse_anomaly =[]

mse_anomaly.extend(mse_anomaly_0)
mse_anomaly.extend(mse_anomaly_1st)
mse_anomaly.extend(mse_anomaly_2nd) 
mse_anomaly.extend(mse_anomaly_3rd)
mse_anomaly.extend(mse_anomaly_4th)
mse_anomaly.extend(mse_anomaly_5th)
mse_anomaly.extend(mse_anomaly_6th)
mse_anomaly.extend(mse_anomaly_7th)
mse_anomaly.extend(mse_anomaly_8th)
mse_anomaly.extend(mse_anomaly_9th)
mse_anomaly.extend(mse_anomaly_10th)

mse_normal = np.array(mse_test)
mse_anomaly = np.array(mse_anomaly)

print(mse_normal.shape)
print(mse_anomaly.shape)

##################################################################################################################################################################################################
######################################################################## Anomaly Detection Visualization adn Quantization ########################################################################
##################################################################################################################################################################################################

######################################################################## 1. Mean and Standard ########################################################################

normal_mean = np.mean(mse_normal)
normal_std = np.std(mse_normal)

fault_mean = np.mean(mse_anomaly)
fault_std = np.std(mse_anomaly)

print("Normal Loss - Mean: ", normal_mean, " Std: ", normal_std)
print("Fault Loss - Mean: ", fault_mean, " Std: ", fault_std)

######################################################################## 2. ROC and F1-score ########################################################################

# 정상 데이터는 라벨 0, 비정상 데이터는 라벨 1로 가정
true_labels = np.concatenate([np.zeros_like(mse_normal), np.ones_like(mse_anomaly)])   # proposed model
scores = np.concatenate([mse_normal, mse_anomaly])   # proposed model

# ROC 곡선 계산
fpr, tpr, thresholds = roc_curve(true_labels, scores)

######################################################################## 2-2. Plot the F1-score and Confusion Matrix ########################################################################

# Youden's J 통계량 (J-Statistic) 사용
# Youden's J 통계량은 (TPR - FPR)의 최대값을 기준으로 최적의 threshold를 선택한다.
# 이 방법은 tpr과 fpr의 차이가 가장 큰 지점을 찾는다.

j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

print( f'ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : {optimal_threshold}')

# predicted_labels = (scores >= optimal_threshold).astype(int)

# FPR 조건 (10% 이하)
fpr_threshold = 0.1

# FPR 조건을 만족하는 인덱스 필터링
valid_indices = np.where(fpr <= fpr_threshold)[0]

if len(valid_indices) > 0:
    # FPR 조건을 만족하는 범위에서 Youden's J 통계량 계산
    j_scores_valid = tpr[valid_indices] - fpr[valid_indices]
    optimal_idx_relative = np.argmax(j_scores_valid)  # 조건을 만족하는 범위 내에서 최적 인덱스
    optimal_idx = valid_indices[optimal_idx_relative]  # 전체 인덱스에 매핑
    optimal_threshold = thresholds[optimal_idx]

    print(f"최적의 threshold (FPR <= {fpr_threshold * 100}%): {optimal_threshold}")
else:
    print(f"FPR <= {fpr_threshold * 100}%를 만족하는 threshold가 없습니다.")
    optimal_threshold = None

# 최적의 threshold를 기반으로 예측 레이블 생성
if optimal_threshold is not None:
    predicted_labels = (scores >= optimal_threshold).astype(int)
else:
    predicted_labels = None

# F1-score, precision, recall 계산
f1 = f1_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)

# 출력
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# 혼동 행렬 출력
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print('Confusion Matrix:')
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',  annot_kws={"size": 16})
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(args.figure_path + 'Normal vs. Fault Confusion Matrix.png', dpi=600)

plt.clf() # figure 초기화
plt.cla() # figure 축 초기화
plt.close() # 현재 figure 닫기

######################################################################## 2-2. Plot the ROC Curve ########################################################################
roc_auc = auc(fpr, tpr)
# LaTeX 스타일 활성화
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ROC 곡선 그리기
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(args.figure_path+' Normal vs. Fault ROC Curve.png', dpi=600)
