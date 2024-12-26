import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import os
from utils import EarlyStopping, CustomDataset, createDirectory
import argparse
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import math 

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

# for i in range(0, num_channels):
#     if i==0:
#         data_input[:,:,i] = scaler_P1.fit(data_input[:,:,i].reshape(-1,1))
#         joblib.dump(scaler_P1, scaler_filename[i])
#     if i==1:
#         data_input[:,:,i] = scaler_P2.fit(data_input[:,:,i].reshape(-1,1))
#         joblib.dump(scaler_P2, scaler_filename[i])
#     if i==2:
#         data_input[:,:,i] = scaler_P3.fit(data_input[:,:,i].reshape(-1,1))
#         joblib.dump(scaler_P3, scaler_filename[i])
#     if i==3:
#         data_input[:,:,i] = scaler_P4.fit(data_input[:,:,i].reshape(-1,1))
#         joblib.dump(scaler_P4, scaler_filename[i])

# # 스케일러 불러오기
# load_channel1 = joblib.load('channel1')
# load_channel2 = joblib.load('channel2')
# load_channel3 = joblib.load('channel3')
# load_channel4 = joblib.load('channel4')

# for i in range(0, num_channels):
#     if i==0:
#         data_input[:,:,i] = scaler_P1.transform(data_input[:,:,i])
#     if i==1:
#         data_input[:,:,i] = scaler_P2.transform(data_input[:,:,i])
#     if i==2:
#         data_input[:,:,i] = scaler_P3.transform(data_input[:,:,i])
#     if i==3:
#         data_input[:,:,i] = scaler_P4.transform(data_input[:,:,i])

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

# #######################################################################
# ############################ STEP 1. TRAIN ############################
# #######################################################################

# # TensorBoard Writer 초기화
# writer = SummaryWriter()
# model_VAE.train()
# num_epochs = 20

# for epoch in range(num_epochs):
#     total_loss = 0
#     middle_loss = 0
#     # tqdm을 이용하여 학습의 진행상황을 시각적으로 표시함.
#     pbar = tqdm(dataloader1, desc=f'Epoch {epoch+1}/{num_epochs}', total=len(dataloader1), ncols=100)
#     for x, c in pbar :
#         x, c = x.to(device), c.to(device)
#         optimizer.zero_grad()
#         x_recon, mu, log_var = model_VAE(x, c)

#         # VAE 손실 계산
#         recon_loss = criterion(x_recon, x)
#         kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#         loss = recon_loss + kl_divergence
#         # loss = loss_function_sum(x_recon, x, mu, log_var)

#         loss.backward()
#         middle_loss += loss.item()
#         optimizer.step()

#         total_loss += middle_loss/4
#         pbar.set_postfix(loss=middle_loss)

#     scheduler.step(middle_loss) # total_loss값을 비교하여 학습 스케쥴러 업데이트
        
#     # TensorBoard에 손실 기록
#     writer.add_scalar('Loss/Epoch', middle_loss, epoch)
#     print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss:{middle_loss:.4f}")

#     # 첫번째 epoch일 때, middle loss 값을 best loss로 저장
#     if epoch == 0 :
#         best_loss = middle_loss

#     # 손실값이 급격히 증가하면 이전 체크포인트로 복구
#     if middle_loss > best_loss * 2:  # 손실값이 두 배 이상 증가할 때
#         _, _ = load_checkpoint('./transformer/checkpoint.pth', model_VAE, optimizer)
#         continue

#     # 손실값이 개선되면 체크포인트 저장
#     if middle_loss < best_loss:
#         best_loss = middle_loss
#         save_checkpoint(epoch, model_VAE, optimizer, loss.item(), './transformer/checkpoint.pth')
        
# torch.save(model_VAE.state_dict(),'./transformer/best_model.pth')

# writer.close()

######################################################################
############################ STEP 2. TEST ############################
######################################################################

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
# data_anomaly = downsampled_df.values
# print(data_anomaly.shape)
# data_anomaly = data_anomaly.reshape(1, data_anomaly.shape[0], data_anomaly.shape[1])

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

########## 3. Anomaly Signal with Random Gaussian Noise ###########

# # 가우시안 노이즈 생성
# mean = 0  # 평균값
# augment_size = 200
# # std_dev = 0.05  # 표준편차
# noise_0 = np.random.normal(mean, 0.001, size=(data_test[100:100+augment_size,:,:]).shape)
# noise_1 = np.random.normal(mean, 0.01, size=(data_test[100:100+augment_size,:,:]).shape)
# noise_2 = np.random.normal(mean, 0.02, size=(data_test[100:100+augment_size,:,:]).shape)
# noise_3 = np.random.normal(mean, 0.03, size=(data_test[100:100+augment_size,:,:]).shape)
# noise_4 = np.random.normal(mean, 0.04, size=(data_test[100:100+augment_size,:,:]).shape)
# noise_5 = np.random.normal(mean, 0.05, size=(data_test[100:100+augment_size,:,:]).shape)
# noise_6 = np.random.normal(mean, 0.06, size=(data_test[100:100+augment_size,:,:]).shape)
# noise_7 = np.random.normal(mean, 0.07, size=(data_test[100:100+augment_size,:,:]).shape)
# noise_8 = np.random.normal(mean, 0.08, size=(data_test[100:100+augment_size,:,:]).shape)
# noise_9 = np.random.normal(mean, 0.09, size=(data_test[100:100+augment_size,:,:]).shape)
# noise_10 = np.random.normal(mean, 0.10, size=(data_test[100:100+augment_size,:,:]).shape)

# # 원본 신호에 가우시안 노이즈 추가하기
# augment_anomaly0 = data_test[100:100+augment_size,:,:]
# augment_anomaly1 = data_test[100:100+augment_size,:,:]
# augment_anomaly2 = data_test[100:100+augment_size,:,:]
# augment_anomaly3 = data_test[100:100+augment_size,:,:]
# augment_anomaly4 = data_test[100:100+augment_size,:,:]
# augment_anomaly5 = data_test[100:100+augment_size,:,:]
# augment_anomaly6 = data_test[100:100+augment_size,:,:]
# augment_anomaly7 = data_test[100:100+augment_size,:,:]
# augment_anomaly8 = data_test[100:100+augment_size,:,:]
# augment_anomaly9 = data_test[100:100+augment_size,:,:]
# augment_anomaly10 = data_test[100:100+augment_size,:,:]

# for i in range (0,5002):
#     if i>=1500 and i<=4000 :
#         # data_anomaly[:,i:i+1,:] = data_anomaly[:,i:i+1,:]*(1 + noise)
#         augment_anomaly0 = augment_anomaly0*(1 + noise_0[:,i:i+1,:]) 
#         augment_anomaly1 = augment_anomaly1*(1 + noise_1[:,i:i+1,:]) 
#         augment_anomaly2 = augment_anomaly2*(1 + noise_2[:,i:i+1,:]) 
#         augment_anomaly3 = augment_anomaly3*(1 + noise_3[:,i:i+1,:]) 
#         augment_anomaly4 = augment_anomaly4*(1 + noise_4[:,i:i+1,:]) 
#         augment_anomaly5 = augment_anomaly5*(1 + noise_5[:,i:i+1,:]) 
#         augment_anomaly6 = augment_anomaly6*(1 + noise_6[:,i:i+1,:]) 
#         augment_anomaly7 = augment_anomaly7*(1 + noise_7[:,i:i+1,:]) 
#         augment_anomaly8 = augment_anomaly8*(1 + noise_8[:,i:i+1,:]) 
#         augment_anomaly9 = augment_anomaly9*(1 + noise_9[:,i:i+1,:]) 
#         augment_anomaly10 = augment_anomaly10*(1 + noise_10[:,i:i+1,:]) 

# data_anomaly = np.concatenate((augment_anomaly0,
#                                augment_anomaly1,
#                                augment_anomaly2,
#                                augment_anomaly3,
#                                augment_anomaly4,
#                                augment_anomaly5,
#                                augment_anomaly6,
#                                augment_anomaly7,
#                                augment_anomaly8,
#                                augment_anomaly9,
#                                augment_anomaly10), axis=0)

# # numpy 파일로 저장할 경로 설정
# anomaly_file_path = './data/anomaly_gaussian_sample_200_ver5.npy'

# # numpy 배열을 .npy 파일로 저장
# np.save(anomaly_file_path, data_anomaly)

# exit()

# numpy 파일로 저장할 경로 설정
anomaly_file_path = './data/anomaly_gaussian_sample_2000_ver5.npy'

data_anomaly = np.load(anomaly_file_path)
augment_size = 2000

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

load_channel1 = joblib.load('channel1_normal')
load_channel2 = joblib.load('channel2_normal')
load_channel3 = joblib.load('channel3_normal')
load_channel4 = joblib.load('channel4_normal')

load_abnormal_channel1 = joblib.load('channel1_abnormal')
load_abnormal_channel2 = joblib.load('channel2_abnormal')
load_abnormal_channel3 = joblib.load('channel3_abnormal')
load_abnormal_channel4 = joblib.load('channel4_abnormal')

# # 스케일링시 주의사항 : transform은 한번 만 사용 가능함을 반드시 인지하기 #
# # 반복해서 transform을 할 경우가 있으면 반드시 저장과 불러오기를 이용하기 #
# for i in range(0, num_channels):
#     if i==0:
#         data_test[:,:,i] = scaler_P1.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)
#     if i==1:
#         data_test[:,:,i] = scaler_P2.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)
#     if i==2:
#         data_test[:,:,i] = scaler_P3.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)
#     if i==3:
#         data_test[:,:,i] = scaler_P4.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)

#### M02 테스트 정상신호와 이상신호 스케일링 ####
for i in range(0, num_channels):
    if i==0:
        data_test[:,:,i] = load_channel1.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)
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
test_labels_one_hot = np.tile(one_hot_vector, (data_test.shape[0], 1))
anomaly_labels_one_not = np.tile(one_hot_vector, (data_anomaly_1st.shape[0], 1))

# 레이블 리스트를 tensor위에 두기
test_labels_one_hot = torch.tensor(test_labels_one_hot, dtype=torch.float32)
anomaly_labels_one_not = torch.tensor(anomaly_labels_one_not, dtype=torch.float32)

# 테스트 데이터세트 및 데이터로드 정의
dataset_test = CustomDataset(test_tensor, test_labels_one_hot)
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, generator=g)

# 이상신호 데이터세트 및 데이터로드 정의

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

# 가우시안 노이즈 그룹 5 : 평균 0, 분산 0.09
dataset_anomaly_9th = CustomDataset(anomaly_9th_tensor, anomaly_labels_one_not)
dataloader_anomaly_9th = DataLoader(dataset_anomaly_9th, batch_size=args.batch_size, shuffle=False, generator=g)

# 가우시안 노이즈 그룹 5 : 평균 0, 분산 0.05
dataset_anomaly_10th = CustomDataset(anomaly_10th_tensor, anomaly_labels_one_not)
dataloader_anomaly_10th = DataLoader(dataset_anomaly_10th, batch_size=args.batch_size, shuffle=False, generator=g)

# 베스트 모델 불러오기 
model_VAE.load_state_dict(torch.load('./transformer/best_model.pth'))

# 모델 평가 플래그 세팅
model_VAE.eval()

# 정상신호 테스트 결과 저장하기
# with torch.no_grad():
#     pbar = tqdm(dataloader_test, total=len(dataloader_test), ncols=100)
#     for x_input, label in pbar :
#         x_input, label = x_input.to(device), label.to(device)
#         x_recon, _, _ = model_VAE(x_input, label)

#         origin = (x_input).cpu().numpy()
#         regenerate = (x_recon).cpu().numpy()

#     origin = origin.transpose(0,2,1)
#     regenerate = regenerate.transpose(0,2,1)
#     for i in range(0, num_channels):
#         if i==0:
#             origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#             regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#         if i==1:
#             origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#             regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#         if i==2:
#             origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#             regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#         if i==3:
#             origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#             regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#     mse = ((origin - regenerate)**2).mean(axis=(1,2))
#     print(f'the Mean Square Error(MSE) of Reconstruction in Normal Signal : {mse}')

# # LaTeX 스타일 활성화
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# # plt.figure(figsize=(12,6))
# plt.figure(figsize=(12,6))

# # plot할 파형 정하기
# for channel_index in range(Module_data.shape[2]):    
#     # 배치사이즈마다 plot 저장하기
#     plt.plot(time, origin[0,:,channel_index], label = f'original {channels[channel_index]}')
#     # plt.plot(time, regenerate[0,:,channel_index], label = f'reconstruciton {channels[channel_index]}')

# # plt.legend()
# plt.xlabel("Time (us)")
# # plt.title('Original Signal vs. Reconstruction Signal')
# # plt.grid(True)
# plt.savefig('./original.png', dpi=600)
# # plt.savefig(args.figure_path+'inference.png', dpi=600)

#     # plt.plot(time, regenerate[0,:,channel_index])
#     # plt.title('Reconstruction Signal')
#     # plt.xlabel("Time (us)")
# plt.clf() # figure 초기화
# plt.cla() # figure 축 초기화
# plt.close() # 현재 figure 닫기

# exit()

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

    pbar_anomaly_6th = tqdm(dataloader_anomaly_6th, total=len(dataloader_anomaly_5th), ncols=100)

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
    
    pbar_anomaly_7th = tqdm(dataloader_anomaly_7th, total=len(dataloader_anomaly_5th), ncols=100)

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
    
    pbar_anomaly_8th = tqdm(dataloader_anomaly_8th, total=len(dataloader_anomaly_5th), ncols=100)

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
    
    pbar_anomaly_9th = tqdm(dataloader_anomaly_9th, total=len(dataloader_anomaly_5th), ncols=100)

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
    
    pbar_anomaly_10th = tqdm(dataloader_anomaly_6th, total=len(dataloader_anomaly_10th), ncols=100)

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
    

################################ Figure 저장할 디렉토리 생성하기 ################################
createDirectory(args.figure_path)

# ################################ t-SNE 기법을 이용한 잠재공간 시각화 ################################

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

# # LaTeX 스타일 활성화
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.figure(figsize=(10,6))

# plt.scatter(latent_tsne_z[:len(latent_z), 0], latent_tsne_z[:len(latent_z), 1], label='Train Data Inference', alpha=0.6)
# plt.scatter(latent_tsne_z_test[:len(latent_z_test), 0], latent_tsne_z_test[:len(latent_z_test), 1], label='Test Data Inference', alpha=0.6)
# plt.title('Train vs. test t-SNE of Latent Space')
# # plt.xlabel('X of latent Space')
# # plt.ylabel('Y of latent Space')
# plt.legend(fontsize=15)
# # plt.grid(True)
# plt.savefig('./tsne_latent_space.png', dpi=600)

# print("추론한 결과 : 잠재공간 t-SNE 시각화")
# plt.clf() # figure 초기화
# plt.cla() # figure 축 초기화
# plt.close() # 현재 figure 닫기

# exit() # 정지 명령어

################################ 플롯에 사용할 데이터 샘플 및 bin 갯수 설정하기 ################################

# 사용될 샘플 갯수 정하기
sample_normal = 12000
sample_anomaly = 2000
bins_normal = 500
bins = 500

# 가우시안 분포 만들기 : train, test, std 0.001, std 0.01, std 0.1
train_gaussian = (mse[:sample_normal]-np.mean(mse[:sample_normal]))/np.std(mse[:sample_normal])
test_gaussian = (mse_test[:sample_normal]-np.mean(mse_test[:sample_normal]))/np.std(mse_test[:sample_normal])

var_0_gaussian =(mse_anomaly_0[:sample_anomaly]-np.mean(mse_anomaly_0[:sample_anomaly]))/np.std(mse_anomaly_0[:sample_anomaly])
var_1_gaussian =(mse_anomaly_1st[:sample_anomaly]-np.mean(mse_anomaly_1st[:sample_anomaly]))/np.std(mse_anomaly_1st[:sample_anomaly])
var_10_gaussian =(mse_anomaly_10th[:sample_anomaly]-np.mean(mse_anomaly_10th[:sample_anomaly]))/np.std(mse_anomaly_10th[:sample_anomaly])


# ################################ t-SNE 기법을 이용한 출력결과 시각화 ################################
# from sklearn.manifold import TSNE

# # 출력결과 mse를 numpy 배열로 변환
# latent_mse = np.array(mse[:sample_normal])
# latent_mse_test = np.array(mse_test[:sample_normal])
# latent_anomaly = np.array(mse_anomaly_1st)
# #  출력결과 mse의 차원을 2d로 확장

# # latent_mse의 차원을 1차원으로 그대로 받아옴
# latent_mse_2d = np.log1p(latent_mse.reshape(latent_mse.shape[0], -1))
# latent_mse_test_2d = np.log1p(latent_mse_test.reshape(latent_mse_test.shape[0], -1))
# latent_mse_anomaly_2d = np.log1p(latent_anomaly.reshape(latent_anomaly.shape[0], -1))

# # 2차원으로 확장
# if latent_mse_2d.shape[1] == 1:  # 특성 수가 1인 경우
#     latent_mse_2d = np.hstack([latent_mse_2d, latent_mse_2d])
    
# if latent_mse_test_2d.shape[1] == 1:  # 특성 수가 1인 경우
#     latent_mse_test_2d = np.hstack([latent_mse_test_2d, latent_mse_test_2d])

# if latent_mse_anomaly_2d.shape[1] == 1:  # 특성 수가 1인 경우
#     latent_mse_anomaly_2d = np.hstack([latent_mse_anomaly_2d, latent_mse_anomaly_2d])

# # t-SNE 시각화를 위한 잠재공간 정의
# perplexity_train = min(30, len(latent_mse_2d) - 1)
# perplexity_test = min(30, len(latent_mse_test_2d) - 1)
# perplexity_anomaly = min(30, len(latent_mse_anomaly_2d) - 1)

# # t-SNE 시각화를 위한 잠재공간 정의
# tsne_z = TSNE(n_components=2, perplexity=perplexity_train, random_state=42)
# tsne_z_test = TSNE(n_components=2, perplexity=perplexity_test, random_state=42)
# tsne_z_anomaly = TSNE(n_components=2, perplexity=perplexity_anomaly, random_state=42)

# latent_tsne_z = tsne_z.fit_transform(latent_mse_2d)
# latent_tsne_z_test = tsne_z_test.fit_transform(latent_mse_test_2d)
# latent_tsne_z_anomaly = tsne_z_anomaly.fit_transform(latent_mse_anomaly_2d)

# # LaTeX 스타일 활성화
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.figure(figsize=(10,6))

# plt.scatter(latent_tsne_z[:len(latent_mse), 0], latent_tsne_z[:len(latent_mse), 1], label='Train Data Inference', alpha=0.6)
# plt.scatter(latent_tsne_z_test[:len(latent_mse_test), 0], latent_tsne_z_test[:len(latent_mse_test), 1], label='Test Data Inference', alpha=0.6)
# plt.scatter(latent_tsne_z_anomaly[:len(latent_anomaly), 0], latent_tsne_z_anomaly[:len(latent_anomaly), 1], label='Anomaly Data Inference', alpha=0.6, color='red')
# plt.title('Train vs. test t-SNE of Latent Space')
# # plt.xlabel('X of latent Space')
# # plt.ylabel('Y of latent Space')
# plt.legend(fontsize=15)
# # plt.grid(True)
# plt.savefig(args.figure_path+'./tsne_latent_space.png', dpi=600)

# print("추론한 결과 : 잠재공간 t-SNE 시각화")
# plt.clf() # figure 초기화
# plt.cla() # figure 축 초기화
# plt.close() # 현재 figure 닫기

# exit()

# ################################ K-means 클러스터링을 이용한 손실값 시각화 ################################
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

# # 손실값 병합
# z_mse_test = (mse_test-np.mean(mse_test))/np.std(mse_test)
# sigma1_mse_test = np.clip(z_mse_test, -1, 1)
# resotred_z_mse_test = (sigma1_mse_test*np.std(mse_test)) + np.mean(mse_test)

# z_anomaly = (mse_anomaly_1st-np.mean(mse_anomaly_1st))/np.std(mse_anomaly_1st)
# sigma1_anomaly = np.clip(z_anomaly, -1, 1)
# resotred_z_anomaly = sigma1_anomaly*np.std(mse_anomaly_1st) + np.mean(mse_anomaly_1st)

# z_anomaly_10th = (mse_anomaly_10th-np.mean(mse_anomaly_10th))/np.std(mse_anomaly_10th)
# sigma1_anomaly_10th = np.clip(z_anomaly_10th, -1, 1)
# resotred_z_anomaly_10th = sigma1_anomaly_10th*np.std(mse_anomaly_10th) + np.mean(mse_anomaly_10th)

# loss_values = np.concatenate([(resotred_z_mse_test), (resotred_z_anomaly), (resotred_z_anomaly_10th)]).reshape(-1, 1)

# # K-means 클러스터링 (클러스터 수: 2 또는 3)
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(loss_values)

# # PCA로 차원 축소 (시각화용)
# pca = PCA(n_components=1)
# reduced_loss = pca.fit_transform(loss_values)

# # LaTeX 스타일 활성화 및 시각화 플롯 크기 설정하기
# plt.rc('text', usetex=True) 
# plt.rc('font', family='serif')
# plt.figure(figsize=(10, 6))

# # 클러스터별로 색상 구분
# plt.scatter(loss_values[clusters == 0], reduced_loss[clusters == 0], c='green', label='Cluster 1 (Normal-Test)')
# plt.scatter(loss_values[clusters == 1], reduced_loss[clusters == 1], c='blue', label='Cluster 2 (Anomaly  r$\sigma$ 0.01)')
# plt.scatter(loss_values[clusters == 2], reduced_loss[clusters == 2], c='red', label='Cluster 3 (Anomaly  r$\sigma$ 0.10)')

# # 클러스터 중심
# centroids = kmeans.cluster_centers_
# # plt.scatter(pca.transform(centroids), centroids, c='black', s=200, marker='X', label='Cluster Centers')

# # 라벨 추가
# plt.title("Clustering of Reconstruction Loss")
# plt.ylabel("PCA Reduced Loss")
# plt.xlabel("Reconstruction Loss")
# plt.xscale("log")  # 로그 스케일로 보기
# plt.savefig(args.figure_path+'/KMeans-Clustering.png', dpi=600)

# print("추론한 결과 : 손실값 KMeans-클러스터링 시각화")

# plt.clf() # figure 초기화
# plt.cla() # figure 축 초기화
# plt.close() # 현재 figure 닫기

# exit()

################################ 히스토그램을 이용한 손실값 시각화 ################################
# LaTeX 스타일 활성화
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

mse_bins = int(np.array(mse).shape[0])
mse_test_bins = int(np.array(mse_test).shape[0])
plt.figure(figsize=(10,6))

# 히스토그램 그리기
# plt.hist(mse[:sample_normal], bins=bins_normal, stacked=True, alpha=0.5, label='Train',color='g')
# plt.hist(mse_test[:sample_normal], bins=bins_normal, stacked=True, alpha=0.5, label='Test',color='b')
# plt.hist(mse_anomaly_0, bins=bins, alpha=0.5, stacked=True, label='0.001')
# plt.hist(mse_anomaly_1st, bins=bins, alpha=0.5, stacked=True, label='0.01',color='r')
# # plt.hist(mse_anomaly_2nd[:sample], bins=bins, alpha=0.5, label='0.02')
# # plt.hist(mse_anomaly_3rd[:sample], bins=bins, alpha=0.5, label='0.03')
# # plt.hist(mse_anomaly_4th[:sample], bins=bins, alpha=0.5, label='0.04')
# # plt.hist(mse_anomaly_5th[:sample], bins=bins, alpha=0.5, label='0.05')
# # plt.hist(mse_anomaly_6th[:sample], bins=bins, alpha=0.5, label='0.06')
# # plt.hist(mse_anomaly_7th[:sample], bins=bins, alpha=0.5, label='0.07')
# # plt.hist(mse_anomaly_8th[:sample], bins=bins, alpha=0.5, label='0.08')
# # plt.hist(mse_anomaly_9th[:sample], bins=bins, alpha=0.5, label='0.09')
# plt.hist(mse_anomaly_10th, bins=bins, alpha=0.5, stacked=True, label='0.10')


# 손실값 z 변환 후, 1시그마 클립핑 뒤 z 역변환
z_mse_test = (mse_test-np.mean(mse_test))/np.std(mse_test)
sigma1_mse_test = np.clip(z_mse_test, -1, 1)
resotred_z_mse_test = (sigma1_mse_test*np.std(mse_test)) + np.mean(mse_test)

z_anomaly = (mse_anomaly_1st-np.mean(mse_anomaly_1st))/np.std(mse_anomaly_1st)
sigma1_anomaly = np.clip(z_anomaly, -1, 1)
resotred_z_anomaly = sigma1_anomaly*np.std(mse_anomaly_1st) + np.mean(mse_anomaly_1st)

z_anomaly_10th = (mse_anomaly_10th-np.mean(mse_anomaly_10th))/np.std(mse_anomaly_10th)
sigma1_anomaly_10th = np.clip(z_anomaly_10th, -1, 1)
resotred_z_anomaly_10th = sigma1_anomaly_10th*np.std(mse_anomaly_10th) + np.mean(mse_anomaly_10th)

# plt.hist(mse[:sample_normal], bins=bins_normal, stacked=True, alpha=0.5, label='Train',color='g')
# plt.hist(np.log1p(resotred_z_mse_test), bins=bins_normal, stacked=True, alpha=0.5, label='Test',color='b',density=True)
# plt.hist(mse_anomaly_0, bins=bins, alpha=0.5, stacked=True, label='0.001')
# plt.hist(np.log1p(resotred_z_anomaly), bins=bins, alpha=0.5, stacked=True, label='0.01',color='r',density=True)
# plt.hist(np.log1p(resotred_z_anomaly_10th), bins=bins, alpha=0.5, stacked=True, label='0.10',density=True)

# 커널 밀도 추정 그래프 그리기
sns.kdeplot(np.log1p(resotred_z_mse_test), fill=True, label='Normal', cut=0, color='green', common_norm=False,bw_adjust=0.25) # 테스트 데이터
sns.kdeplot(np.log1p(resotred_z_anomaly), fill=True, label='Pre-fault(1-$\sigma$ = 0.01)', cut=0, color='blue', common_norm=False, bw_adjust=0.25) # 테스트 데이터
sns.kdeplot(np.log1p(resotred_z_anomaly_10th), fill=True, label='Pre-fault(1-$\sigma$ = 0.10)', cut=0, color='red', common_norm=False, bw_adjust=0.25) # 테스트 데이터

plt.legend(fontsize=10)

# 축 라벨 및 파라미터 설정
plt.title("Train vs. Test Recosntrucion Loss Histogram")
plt.xlabel('MSE', fontsize=15, fontweight='bold')
plt.ylabel('Loss Density', fontsize=15, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-4, 1e+1)
plt.ylim(1e-3, 1e+5)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))

plt.savefig(args.figure_path+'/Histogram.png', dpi=600)

print("추론한 결과 : 손실값 히스토그램 시각화")

plt.clf() # figure 초기화
plt.cla() # figure 축 초기화
plt.close() # 현재 figure 닫기

exit()

################################ 커널 밀도 추정(Kernel Density Estimation)을 이용한 손실값 시각화 ################################
# LaTeX 스타일 활성화
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(10,6))

# # 히스토그램 그리기
# plt.hist(mse[:2], bins=1, alpha=0.5, label='Train Dataset Histogram',color='g')
# plt.hist(mse_test[:2], bins=1, alpha=0.5, label='Test Dataset Histogram',color='b')
# plt.hist(mse_anomaly_1st[:2], bins=1, alpha=0.5, label='Sigma 0.01 Histogram',color='r')
# plt.hist(mse_anomaly_2nd[:2], bins=1, alpha=0.5, label='Sigma 0.02 Histogram')
# plt.hist(mse_anomaly_3rd[:2], bins=1, alpha=0.5, label='Sigma 0.03 Histogram')
# plt.hist(mse_anomaly_4th[:2], bins=1, alpha=0.5, label='Sigma 0.04 Histogram')
# plt.hist(mse_anomaly_5th[:2], bins=1, alpha=0.5, label='Sigma 0.05 Histogram')
# plt.hist(mse_anomaly_6th[:2], bins=1, alpha=0.5, label='Sigma 0.06 Histogram')
# plt.hist(mse_anomaly_7th[:2], bins=1, alpha=0.5, label='Sigma 0.07 Histogram')
# plt.hist(mse_anomaly_8th[:2], bins=1, alpha=0.5, label='Sigma 0.08 Histogram')
# plt.hist(mse_anomaly_9th[:2], bins=1, alpha=0.5, label='Sigma 0.09 Histogram')
# plt.hist(mse_anomaly_10th[:2], bins=1, alpha=0.5, label='Sigma 0.10 Histogram')

# 커널 밀도 추정 그래프 그리기
sns.kdeplot((mse[:sample]), fill=True, label='Train Dataset KDE', cut=0, color='g', common_norm=False, bw_adjust=0.25) # 학습한 데이터
sns.kdeplot((mse_test[:sample]), fill=True, label='Test Dataset KDE', cut=0, color='b', common_norm=False,bw_adjust=0.25) # 테스트 데이터
sns.kdeplot((mse_anomaly_0[:sample]), fill=True, label='sigma 0.001 KDE', cut=0, color='gray', common_norm=False, bw_adjust=0.25) # 테스트 데이터
sns.kdeplot((mse_anomaly_1st[:sample]), fill=True, label='sigma 0.01 KDE', cut=0, color='r', common_norm=False, bw_adjust=0.25) # 테스트 데이터
# sns.kdeplot((mse_anomaly_2nd[:sample]), fill=True, label='sigma 0.02 KDE', cut=0) # 테스트 데이터
# sns.kdeplot((mse_anomaly_3rd[:sample]), fill=True, label='sigma 0.03 KDE', cut=0) # 테스트 데이터
# sns.kdeplot((mse_anomaly_4th[:sample]), fill=True, label='sigma 0.04 KDE', cut=0) # 테스트 데이터
# sns.kdeplot((mse_anomaly_5th[:sample]), fill=True, label='sigma 0.05 KDE', cut=0) # 테스트 데이터
# sns.kdeplot((mse_anomaly_6th[:sample]), fill=True, label='sigma 0.06 KDE', cut=0) # 테스트 데이터
# sns.kdeplot((mse_anomaly_7th[:sample]), fill=True, label='sigma 0.07 KDE', cut=0) # 테스트 데이터
# sns.kdeplot((mse_anomaly_8th[:sample]), fill=True, label='sigma 0.08 KDE', cut=0) # 테스트 데이터
# sns.kdeplot((mse_anomaly_9th[:sample]), fill=True, label='sigma 0.09 KDE', cut=0) # 테스트 데이터
sns.kdeplot((mse_anomaly_10th[:sample]), fill=True, label='sigma 0.10 KDE', cut=0, common_norm=False, bw_adjust=0.25) # 테스트 데이터

plt.legend(fontsize=10)
plt.title("Histogram and Kernel Density Estimation")
plt.xlabel('Mean Square Error', fontsize=15, fontweight='bold')
plt.ylabel('Loss Density', fontsize=15, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-5, 1e+3)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))

plt.savefig(args.figure_path+'/KDEplot.png',dpi=600)

print("추론 결과 : 손실값의 커널 밀도추정 시각화")

plt.clf() # figure 초기화
plt.cla() # figure 축 초기화
plt.close() # 현재 figure 닫기

################################ 상자그림(Box Plot)을 이용한 손실값 시각화 ################################
# LaTeX 스타일 활성화
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(10,6))
data_train = pd.DataFrame({'MSE_train':mse, 'Type': 'Train'})
data_test = pd.DataFrame({'MSE_test':mse_test, 'Type': 'Test'})
data_anomaly_0 = pd.DataFrame({'MSE_anomaly_0':mse_anomaly_0, 'Type': '0.001'})
data_anomaly_1st = pd.DataFrame({'MSE_anomaly_1st':mse_anomaly_1st, 'Type': '0.01'})
data_anomaly_2nd = pd.DataFrame({'MSE_anomaly_2nd':mse_anomaly_2nd, 'Type': '0.02'})
data_anomaly_3rd = pd.DataFrame({'MSE_anomaly_3rd':mse_anomaly_3rd, 'Type': '0.03'})
data_anomaly_4th = pd.DataFrame({'MSE_anomaly_4th':mse_anomaly_4th, 'Type': '0.04'})
data_anomaly_5th = pd.DataFrame({'MSE_anomaly_5th':mse_anomaly_5th, 'Type': '0.05'})
data_anomaly_6th = pd.DataFrame({'MSE_anomaly_6th':mse_anomaly_6th, 'Type': '0.06'})
data_anomaly_7th = pd.DataFrame({'MSE_anomaly_7th':mse_anomaly_7th, 'Type': '0.07'})
data_anomaly_8th = pd.DataFrame({'MSE_anomaly_8th':mse_anomaly_8th, 'Type': '0.08'})
data_anomaly_9th = pd.DataFrame({'MSE_anomaly_9th':mse_anomaly_9th, 'Type': '0.09'})
data_anomaly_10th = pd.DataFrame({'MSE_anomaly_10th':mse_anomaly_10th, 'Type': '0.10'})

# 박스 그림 그리기
sns.boxplot(x='Type', y='MSE_train',data=data_train, label='Train Dataset',color='g',flierprops=dict(marker='None'))
sns.boxplot(x='Type', y='MSE_test',data=data_test, label='Test Dataset',color='b',flierprops=dict(marker='None'))
sns.boxplot(x='Type', y='MSE_anomaly_0',data=data_anomaly_0, label='0.001',color='r',flierprops=dict(marker='None'))
sns.boxplot(x='Type', y='MSE_anomaly_1st',data=data_anomaly_1st, label='0.01',color='r',flierprops=dict(marker='None'))
sns.boxplot(x='Type', y='MSE_anomaly_2nd',data=data_anomaly_2nd, label='0.02',flierprops=dict(marker='None'))
sns.boxplot(x='Type', y='MSE_anomaly_3rd',data=data_anomaly_3rd, label='0.03',flierprops=dict(marker='None'))
sns.boxplot(x='Type', y='MSE_anomaly_4th',data=data_anomaly_4th, label='0.04',flierprops=dict(marker='None'))
sns.boxplot(x='Type', y='MSE_anomaly_5th',data=data_anomaly_5th, label='0.05',flierprops=dict(marker='None'))
sns.boxplot(x='Type', y='MSE_anomaly_6th',data=data_anomaly_6th, label='0.06',flierprops=dict(marker='None'))
sns.boxplot(x='Type', y='MSE_anomaly_7th',data=data_anomaly_7th, label='0.07',flierprops=dict(marker='None'))
sns.boxplot(x='Type', y='MSE_anomaly_8th',data=data_anomaly_8th, label='0.08',flierprops=dict(marker='None'))
sns.boxplot(x='Type', y='MSE_anomaly_9th',data=data_anomaly_9th, label='0.09',flierprops=dict(marker='None'))
sns.boxplot(x='Type', y='MSE_anomaly_10th',data=data_anomaly_10th, label='0.10',flierprops=dict(marker='None'))

# plt.legend(fontsize=25)

# 박스 그림 축 및 라벨 설정
plt.title("Train vs. Test vs. Anomaly Recosntrucion Loss Box plot")
plt.xlabel('$\sigma$', fontsize=10, fontweight='bold')
plt.ylabel('MSE', fontsize=10, fontweight='bold')
plt.yscale('log')
plt.ylim(1e-5, 1e+2)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))
# plt.legend(fontsize=10)
plt.savefig(args.figure_path+'/BOXplot.png',dpi=600)

print("추론 결과 : 손실값의 박스 그림 시각화")

# ### M02 정상 신호 테스트 plot ####
# with torch.no_grad():
#     pbar = tqdm(dataloader_test, total=len(dataloader_test), ncols=100)
#     for x_input, label in pbar :
#         x_input, label = x_input.to(device), label.to(device)
#         x_recon, _, _ = model_VAE(x_input, label)

#         origin = (x_input).cpu().numpy()
#         regenerate = (x_recon).cpu().numpy()

#     origin = origin.transpose(0,2,1)
#     regenerate = regenerate.transpose(0,2,1)
    
#     # 이상치 신호 Mean Square Error 구함, plot할 값들 저장
#     pbar = tqdm(dataloader_anomaly, total=len(dataloader_anomaly), ncols=100)
#     for anomaly_input, label in pbar :
#         anomaly_input, label = anomaly_input.to(device), label.to(device)
#         anomaly_recon, _, _ = model_VAE(anomaly_input, label)

#         origin_anomaly = (anomaly_input).cpu().numpy()
#         regenerate_anomaly = (anomaly_recon).cpu().numpy()
    
#     origin_anomaly = origin_anomaly.transpose(0,2,1)
#     regenerate_anomaly = regenerate_anomaly.transpose(0,2,1)
#     origin_length = int(origin_anomaly.shape[0])
#     regenerate_length = int(regenerate_anomaly.shape[0])

#     original = np.concatenate((origin,origin_anomaly), axis=0)
#     reconstruction = np.concatenate((regenerate,regenerate_anomaly), axis=0)
#     for i in range(0, num_channels):
#         if i==0:
#             original[:,:,i] = load_channel1.inverse_transform(original[:,:,i].reshape(-1,1)).reshape(original[:,:,i].shape)
#             reconstruction[:,:,i] = load_channel1.inverse_transform(reconstruction[:,:,i].reshape(-1,1)).reshape(reconstruction[:,:,i].shape)
#         if i==1:
#             original[:,:,i] = load_channel2.inverse_transform(original[:,:,i].reshape(-1,1)).reshape(original[:,:,i].shape)
#             reconstruction[:,:,i] = load_channel2.inverse_transform(reconstruction[:,:,i].reshape(-1,1)).reshape(reconstruction[:,:,i].shape)
#         if i==2:
#             original[:,:,i] = load_channel3.inverse_transform(original[:,:,i].reshape(-1,1)).reshape(original[:,:,i].shape)
#             reconstruction[:,:,i] = load_channel3.inverse_transform(reconstruction[:,:,i].reshape(-1,1)).reshape(reconstruction[:,:,i].shape)
#         if i==3:
#             original[:,:,i] = load_channel4.inverse_transform(original[:,:,i ].reshape(-1,1)).reshape(original[:,:,i].shape)
#             reconstruction[:,:,i] = load_channel4.inverse_transform(reconstruction[:,:,i].reshape(-1,1)).reshape(reconstruction[:,:,i].shape)
#     # 정상신호
#     origin = original[:len(original),:,:]
#     regenerate = reconstruction[:len(reconstruction),:,:]
#     # 이상신호
#     origin_anomaly = original[len(original)-(origin_length):,:,:]
#     regenerate_anomaly = reconstruction[len(reconstruction)-(regenerate_length):,:,:]

#     # for i in range(0, num_channels):
#     #     if i==0:
#     #         origin[:,:,i] = load_channel1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#     #         regenerate[:,:,i] = load_channel1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#     #     if i==1:
#     #         origin[:,:,i] = load_channel2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#     #         regenerate[:,:,i] = load_channel2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#     #     if i==2:
#     #         origin[:,:,i] = load_channel3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#     #         regenerate[:,:,i] = load_channel3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#     #     if i==3:
#     #         origin[:,:,i] = load_channel4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#     #         regenerate[:,:,i] = load_channel4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#     mse = ((origin - regenerate)**2).mean(axis=(1,2))
#     print(f'the Mean Square Error(MSE) of Reconstruction in Normal Signal : {mse}')

#     # LaTeX 스타일 활성화
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#     # plt.figure(figsize=(12,6))
#     plt.figure(figsize=(12,6))

#     # plot할 파형 정하기
#     for channel_index in range(Module_data.shape[2]):    
#         # 배치사이즈마다 plot 저장하기
#         plt.plot(time, origin[0,:,channel_index], label = f'original {channels[channel_index]}')
#         plt.plot(time, regenerate[0,:,channel_index], label = f'reconstruciton {channels[channel_index]}')

#     plt.legend()
#     plt.xlabel("Time (us)")
#     plt.title('Original Signal vs. Reconstruction Signal')
#     plt.grid(True)
#     plt.savefig('./inference_reconstruction.png', dpi=600)

#     plt.clf() # figure 초기화
#     plt.cla() # figure 축 초기화
#     plt.close() # 현재 figure 닫기

#     ### M02 이상치 신호 테스트 plot ####

#     mse = ((origin_anomaly - regenerate_anomaly)**2).mean(axis=(1,2))
#     print(f'the Mean Square Error(MSE) of Reconstruction in Anomaly Signal : {mse}')

#     for channel_index in range(Module_data.shape[2]):
#         channel_error = ((origin_anomaly[:,:,channel_index:channel_index+1] - regenerate_anomaly[:,:,channel_index:channel_index+1])).sum() # numpy 배열이기에 dim이 아닌, axis를 사용함.
#         print(f'the Error of Reconstruction in Anomaly Channel {channels[channel_index]} Signal : {channel_error}')

#     # LaTeX 스타일 활성화
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#     plt.figure(figsize=(12,6))

#     # 채널별 원본 이상신호와 복원신호 plot
#     for channel_index in range(Module_data.shape[2]):
#         plt.plot(time, origin_anomaly[0,:,channel_index], label = f'Anomaly Signal {channels[channel_index]}')
#         plt.plot(time, regenerate_anomaly[0,:,channel_index], label = f'Reconstruciton Signal {channels[channel_index]}')

#     plt.legend(fontsize=10)
#     plt.xlabel("Time (us)",fontsize=20)
#     plt.title('Anomaly Signal Vs. Reconstruction Signal',fontsize=20)
#     plt.grid(True)
#         # plt.savefig(args.figure_path+'inference_anomaly %s.png'%channels[channel_index], dpi=600)
#     plt.savefig('./inference of anomaly signal.png', dpi=600)

#     plt.clf() # figure 초기화
#     plt.cla() # figure 축 초기화
#     plt.close() # 현재 figure 닫기

# exit()

# ### 가중치 시각화 ###
# # # 모델 적용
# # x_recon, mu, log_var = model_VAE(x, c)

# # # Attention 가중치 시각화
# # attn_weights = model_VAE.get_attention_weights()
# # visualize_attention_weights(attn_weights, layer_idx=0, head_idx=0)
