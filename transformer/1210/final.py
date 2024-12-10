import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import os
from utils import EarlyStopping, CustomDataset
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
        # mu, log_var = torch.chunk(h, 2, dim=-1)
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
    print(f"Checkpoint saved at epoch {epoch}")

# 모델의 체크포인트 불러오기
def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded. Resuming from epoch {epoch} with loss {loss:.4f}")
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

for i in range(0, num_channels):
    if i==0:
        data_input[:,:,i] = scaler_P1.fit_transform(data_input[:,:,i].reshape(-1,1)).reshape(data_input[:,:,i].shape)
    if i==1:
        data_input[:,:,i] = scaler_P2.fit_transform(data_input[:,:,i].reshape(-1,1)).reshape(data_input[:,:,i].shape)
    if i==2:
        data_input[:,:,i] = scaler_P3.fit_transform(data_input[:,:,i].reshape(-1,1)).reshape(data_input[:,:,i].shape)
    if i==3:
        data_input[:,:,i] = scaler_P4.fit_transform(data_input[:,:,i].reshape(-1,1)).reshape(data_input[:,:,i].shape)

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
#         print("Loss exploded! Reloading from last checkpoint...")
#         _, _ = load_checkpoint('./checkpoint.pth', model_VAE, optimizer)
#         continue  # 다음 반복으로 이동

#     # 손실값이 개선되면 체크포인트 저장
#     if middle_loss < best_loss:
#         best_loss = middle_loss
#         save_checkpoint(epoch, model_VAE, optimizer, loss.item(), './checkpoint.pth')
        
# torch.save(model_VAE.state_dict(),'./mlp_transformer_CVAE.pth')

# writer.close()

######################################################################
############################ STEP 2. TEST ############################
######################################################################

# 테스트 데이터
if (((int((Module_data.shape[0])*0.8)) % 8) == 1) :
    data_test = Module_data[int((Module_data.shape[0])*0.8)-1:,:,:]
else :
    data_test = Module_data[int((Module_data.shape[0])*0.8):,:,:]

########### 1. Original Anomaly Signal ###########
# npy file path setting
npy_file_path = os.getcwd() 
M02_anomaly = np.load(os.path.join(npy_file_path, 'data/sample.npy'), allow_pickle=True)

anomaly_input = M02_anomaly[:,1:]
print(f'the shape of anomaly data at June.02.2023 : {anomaly_input.shape}')

# pulse on/off 되는 타이밍을 고려 : 신호가 들어오는 1000번째부터 나가는 4000번째까지 선택
# 신호가 들어오는 구간 전체를 선택
processed_input = anomaly_input[:, :]
print(f'The shape of anomaly data after selecting pulse signal region: {processed_input.shape}')

# 데이터가 부족한 경우 외삽을 통해 데이터를 확장
required_length = 100040
current_length = processed_input.shape[0]

if current_length < required_length:
    num_missing_rows = required_length - current_length
    num_missing_rows_each_side = num_missing_rows // 2

    # 앞쪽과 뒤쪽에 데이터를 반복하여 확장
    front_extension = processed_input[:num_missing_rows_each_side, :]
    front_extension_less = processed_input[:int((num_missing_rows - num_missing_rows_each_side)*13/10), :]
    back_extension = processed_input[-(num_missing_rows - num_missing_rows_each_side):, :]
    back_extension_less = processed_input[-int((num_missing_rows - num_missing_rows_each_side)*1/10):, :]
    # 데이터를 앞과 뒤에 반복된 데이터를 추가하여 확장
    processed_input = np.vstack([ front_extension_less,
                                  front_extension,
                                  processed_input,
                                  back_extension,
                                  back_extension_less])
    
# DataFrame으로 변환 (각 열이 채널이 되도록 설정)
df = pd.DataFrame(processed_input[:,:])

# 이동평균 적용( 윈도우 크기 설정 : 96062개의 샘플을 5002개로 줄이기 위한 window 크기 설정)
window_size = max(1, int(len(df) / 5002))  # 최소 윈도우 크기를 1로 설정하여 오류 방지
rolloing_mean_df = df.rolling(window=window_size, min_periods=1).mean()

# 윈도우 사이즈 간격으로 다운 샘플링 ( 5002개 샘플만을 남김 )
downsampled_df = rolloing_mean_df.iloc[::window_size].reset_index(drop=True)
downsampled_df = downsampled_df[:5002]

# 결과 넘파이 배열로 변환
data_anomaly = downsampled_df.values
print(data_anomaly.shape)
data_anomaly = data_anomaly.reshape(1, data_anomaly.shape[0], data_anomaly.shape[1])

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

#################### 스케일링시 주의사항 : transform은 한번 만 사용 가능함을 반드시 인지하기!!!!! ####################
#################### 반복해서 transform을 할 경우가 있으면 반드시 저장과 불러오기를 이용하기!!!!! ####################
# for i in range(0, num_channels):
#     if i==0:
#         data_test[:,:,i] = scaler_P1.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)
#     if i==1:
#         data_test[:,:,i] = scaler_P2.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)
#     if i==2:
#         data_test[:,:,i] = scaler_P3.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)
#     if i==3:
#         data_test[:,:,i] = scaler_P4.transform(data_test[:,:,i].reshape(-1,1)).reshape(data_test[:,:,i].shape)

for i in range(0, num_channels):
    if i==0:
        data_anomaly[:,:,i] = scaler_P1.transform(data_anomaly[:,:,i].reshape(-1,1)).reshape(data_anomaly[:,:,i].shape)
    if i==1:
        data_anomaly[:,:,i] = scaler_P2.transform(data_anomaly[:,:,i].reshape(-1,1)).reshape(data_anomaly[:,:,i].shape)
    if i==2:
        data_anomaly[:,:,i] = scaler_P3.transform(data_anomaly[:,:,i].reshape(-1,1)).reshape(data_anomaly[:,:,i].shape)
    if i==3:
        data_anomaly[:,:,i] = scaler_P4.transform(data_anomaly[:,:,i].reshape(-1,1)).reshape(data_anomaly[:,:,i].shape)

# 입력 형태 (N, 5002, 4)를 (N, 4, 5002)로 변경함.
data_test = data_test.transpose(0,2,1)
data_anomaly = data_anomaly.transpose(0,2,1)

# numpy 데이터를 tensor 위에 올림
test_tensor = torch.tensor(data_test, dtype=torch.float32)
anomaly_tensor = torch.tensor(data_anomaly, dtype=torch.float32)

# 원-핫 벡터 레이블 생성
one_hot_vector = np.zeros(4)
case = args.case_num #  unique 파형 4개
one_hot_vector[case-1] = 1
test_labels_one_hot = np.tile(one_hot_vector, (data_test.shape[0], 1))
anomaly_labels_one_not = np.tile(one_hot_vector, (data_anomaly.shape[0], 1))

# 레이블 리스트를 tensor위에 두기
test_labels_one_hot = torch.tensor(test_labels_one_hot, dtype=torch.float32)
anomaly_labels_one_not = torch.tensor(anomaly_labels_one_not, dtype=torch.float32)

# 테스트 데이터세트 및 데이터로드 정의
dataset_test = CustomDataset(test_tensor, test_labels_one_hot)
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, generator=g)

# 이상신호 데이터세트 및 데이터로드 정의
dataset_anomaly = CustomDataset(anomaly_tensor, anomaly_labels_one_not)
dataloader_anomaly = DataLoader(dataset_anomaly, batch_size=args.batch_size, shuffle=False, generator=g)

# model_VAE.load_state_dict(torch.load('./transformer_CVAE.pth'))

model_VAE.load_state_dict(torch.load('./mlp_transformer_CVAE.pth'))

model_VAE.eval()

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
#             origin[:,:,i] = scaler_P1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#             regenerate[:,:,i] = scaler_P1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#         if i==1:
#             origin[:,:,i] = scaler_P2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#             regenerate[:,:,i] = scaler_P2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#         if i==2:
#             origin[:,:,i] = scaler_P3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#             regenerate[:,:,i] = scaler_P3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#         if i==3:
#             origin[:,:,i] = scaler_P4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#             regenerate[:,:,i] = scaler_P4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

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
#     plt.plot(time, regenerate[0,:,channel_index], label = f'reconstruciton {channels[channel_index]}')

# plt.legend()
# plt.xlabel("Time (us)")
# plt.title('Original Signal vs. Reconstruction Signal')
# plt.grid(True)
# plt.savefig('./inference_reconstruction.png', dpi=600)
# # plt.savefig(args.figure_path+'inference.png', dpi=600)

#     # plt.plot(time, regenerate[0,:,channel_index])
#     # plt.title('Reconstruction Signal')
#     # plt.xlabel("Time (us)")
# plt.clf() # figure 초기화
# plt.cla() # figure 축 초기화
# plt.close() # 현재 figure 닫기

### M02 이상치 신호 테스트 plot ####
with torch.no_grad():
    # 이상치 신호 Mean Square Error 구함, plot할 값들 저장
    pbar = tqdm(dataloader_anomaly, total=len(dataloader_anomaly), ncols=100)
    for x_input, label in pbar :
        x_input, label = x_input.to(device), label.to(device)
        x_recon, _, _ = model_VAE(x_input, label)

        origin_anomaly = (x_input).cpu().numpy()
        regenerate = (x_recon).cpu().numpy()
    
    origin_anomaly = origin_anomaly.transpose(0,2,1)
    regenerate = regenerate.transpose(0,2,1)

    for i in range(0, num_channels):
        if i==0:
            origin_anomaly[:,:,i] = scaler_P1.inverse_transform(origin_anomaly[:,:,i].reshape(-1,1)).reshape(origin_anomaly[:,:,i].shape)
            regenerate[:,:,i] = scaler_P1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)
        if i==1:
            origin_anomaly[:,:,i] = scaler_P2.inverse_transform(origin_anomaly[:,:,i].reshape(-1,1)).reshape(origin_anomaly[:,:,i].shape)
            regenerate[:,:,i] = scaler_P2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)
        if i==2:
            origin_anomaly[:,:,i] = scaler_P3.inverse_transform(origin_anomaly[:,:,i].reshape(-1,1)).reshape(origin_anomaly[:,:,i].shape)
            regenerate[:,:,i] = scaler_P3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)
        if i==3:
            origin_anomaly[:,:,i] = scaler_P4.inverse_transform(origin_anomaly[:,:,i].reshape(-1,1)).reshape(origin_anomaly[:,:,i].shape)
            regenerate[:,:,i] = scaler_P4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

    mse = ((origin_anomaly - regenerate)**2).mean(axis=(1,2))
    print(f'the Mean Square Error(MSE) of Reconstruction in Anomaly Signal : {mse}')

for channel_index in range(Module_data.shape[2]):
    channel_error = ((origin_anomaly[:,:,channel_index:channel_index+1] - regenerate[:,:,channel_index:channel_index+1])).sum() # numpy 배열이기에 dim이 아닌, axis를 사용함.
    print(f'the Error of Reconstruction in Anomaly Channel {channels[channel_index]} Signal : {channel_error}')

# LaTeX 스타일 활성화
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(12,6))

# 채널별 원본 이상신호와 복원신호 plot
for channel_index in range(Module_data.shape[2]):
    plt.plot(time, origin_anomaly[0,:,channel_index], label = f'Anomaly Signal {channels[channel_index]}')
    plt.plot(time, regenerate[0,:,channel_index], label = f'Reconstruciton Signal {channels[channel_index]}')

plt.legend(fontsize=10)
plt.xlabel("Time (us)",fontsize=20)
plt.title('Anomaly Signal Vs. Reconstruction Signal',fontsize=20)
plt.grid(True)
    # plt.savefig(args.figure_path+'inference_anomaly %s.png'%channels[channel_index], dpi=600)
plt.savefig('./inference_generated anomaly C1~C4.png', dpi=600)

plt.clf() # figure 초기화
plt.cla() # figure 축 초기화
plt.close() # 현재 figure 닫기

exit()

# with torch.no_grad():
#     mse = []
#     latent_mu = []
#     latent_logvar = []
#     latent_z = []

#     pbar = tqdm(dataloader1, total=len(dataloader1), ncols=100)

#     for x_input, label in pbar :
#         x_input, label = x_input.to(device), label.to(device)
        
#         # 모델 출력값 받아오기
#         x_recon, mu, logvar = model_VAE(x_input, label)
#         z = model_VAE.reparameterize(mu, logvar)

#         origin = (x_input).cpu().numpy()
#         regenerate = (x_recon).cpu().numpy()

#         origin = origin.transpose(0,2,1)
#         regenerate = regenerate.transpose(0,2,1)

#         for i in range(0, num_channels):
#             if i==0:
#                 origin[:,:,i] = scaler_P1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#                 regenerate[:,:,i] = scaler_P1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#             if i==1:
#                 origin[:,:,i] = scaler_P2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#                 regenerate[:,:,i] = scaler_P2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#             if i==2:
#                 origin[:,:,i] = scaler_P3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#                 regenerate[:,:,i] = scaler_P3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#             if i==3:
#                 origin[:,:,i] = scaler_P4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#                 regenerate[:,:,i] = scaler_P4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)
        
#         mse.extend(((origin - regenerate)**2).mean(axis=(1,2)))

#         latent_z.append(z.cpu().numpy())
    
# with torch.no_grad():
#     mse_test = []
#     latent_mu_test = []
#     latent_logvar_test = []
#     latent_z_test = []

#     pbar_test = tqdm(dataloader_test, total=len(dataloader_test), ncols=100)

#     for x_input, label in pbar_test :
#         x_input, label = x_input.to(device), label.to(device)

#         # 모델 출력값 받아오기
#         x_recon, mu, logvar = model_VAE(x_input, label)
#         z = model_VAE.reparameterize(mu, logvar)

#         origin = (x_input).cpu().numpy()
#         regenerate = (x_recon).cpu().numpy()

#         origin = origin.transpose(0,2,1)
#         regenerate = regenerate.transpose(0,2,1)

#         for i in range(0, num_channels):
#             if i==0:
#                 origin[:,:,i] = scaler_P1.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#                 regenerate[:,:,i] = scaler_P1.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#             if i==1:
#                 origin[:,:,i] = scaler_P2.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#                 regenerate[:,:,i] = scaler_P2.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#             if i==2:
#                 origin[:,:,i] = scaler_P3.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#                 regenerate[:,:,i] = scaler_P3.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#             if i==3:
#                 origin[:,:,i] = scaler_P4.inverse_transform(origin[:,:,i].reshape(-1,1)).reshape(origin[:,:,i].shape)
#                 regenerate[:,:,i] = scaler_P4.inverse_transform(regenerate[:,:,i].reshape(-1,1)).reshape(regenerate[:,:,i].shape)

#         mse_test.extend(((origin - regenerate)**2).mean(axis=(1,2)))

#         latent_z_test.append(z.cpu().numpy())

# # ################################ t-SNE 기법을 이용한 잠재공간 시각화 ################################

# # # 잠재공간 z를 numpy 배열로 변환
# # latent_z = np.concatenate(latent_z, axis=0)
# # latent_z_test = np.concatenate(latent_z_test, axis=0)

# # #  잠재공간 Z의 차원을 2d로 변환
# # latent_z_2d = latent_z.reshape(latent_z.shape[0], -1)
# # latent_z_test_2d = latent_z_test.reshape(latent_z_test.shape[0], -1)
# # # t-SNE 시각화를 위한 잠재공간 정의
# # perplexity_train = min(30, len(latent_z_2d) - 1)
# # perplexity_test = min(30, len(latent_z_test_2d) - 1)

# # # t-SNE 시각화를 위한 잠재공간 정의
# # tsne_z = TSNE(n_components=2, perplexity=perplexity_train, random_state=42)
# # tsne_z_test = TSNE(n_components=2, perplexity=perplexity_test, random_state=42)

# # latent_tsne_z = tsne_z.fit_transform(latent_z_2d)
# # latent_tsne_z_test = tsne_z_test.fit_transform(latent_z_test_2d)

# # # LaTeX 스타일 활성화
# # plt.rc('text', usetex=True)
# # plt.rc('font', family='serif')
# # plt.figure(figsize=(10,6))

# # plt.scatter(latent_tsne_z[:len(latent_z), 0], latent_tsne_z[:len(latent_z), 1], label='Train Data Inference', alpha=0.6)
# # plt.scatter(latent_tsne_z_test[:len(latent_z_test), 0], latent_tsne_z_test[:len(latent_z_test), 1], label='Test Data Inference', alpha=0.6)
# # plt.title('Train vs. test t-SNE of Latent Space')
# # # plt.xlabel('X of latent Space')
# # # plt.ylabel('Y of latent Space')
# # plt.legend(fontsize=15)
# # # plt.grid(True)
# # plt.savefig('./tsne_latent_space.png', dpi=600)

# # print("추론한 결과 : 잠재공간 t-SNE 시각화")
# # plt.clf() # figure 초기화
# # plt.cla() # figure 축 초기화
# # plt.close() # 현재 figure 닫기

# # exit() # 정지 명령어

# ################################ 히스토그램을 이용한 손실값 시각화 ################################
# # LaTeX 스타일 활성화
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# mse_bins = int(np.array(mse).shape[0])
# mse_test_bins = int(np.array(mse_test).shape[0])
# plt.figure(figsize=(10,6))
# plt.hist(mse, bins=mse_bins, alpha=0.5, label='Train Data Inference')
# plt.hist(mse_test, bins=mse_test_bins, alpha=0.5, label='Test Data Inference')
# plt.legend(fontsize=15)

# # 축 라벨 및 파라미터 설정
# plt.title("Train vs. Test Recosntrucion Loss Histogram")
# plt.xlabel('MSE', fontsize=15, fontweight='bold')
# plt.ylabel('Loss Density', fontsize=15, fontweight='bold')
# plt.xscale('log')
# plt.xlim(1e-5, (1.5)*1e-1)
# plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))

# plt.savefig('./Histogram.png', dpi=600)

# print("추론한 결과 : 손실밀도 히스토그램 시각화")

# plt.clf() # figure 초기화
# plt.cla() # figure 축 초기화
# plt.close() # 현재 figure 닫기

# ################################ 커널 밀도 추정(Kernel Density Estimation)을 이용한 손실값 시각화 ################################
# # LaTeX 스타일 활성화
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.figure(figsize=(10,6))
# # 커널 밀도 추정 그리기
# plt.hist(mse, bins=mse_bins, alpha=0.5, label='Train Dataset Histogram')
# plt.hist(mse_test, bins=mse_test_bins, alpha=0.5, label='Test Dataset Histogram')
# sns.kdeplot((mse), fill=True, label='Train Dataset KDE', cut=0) # 학습한 데이터
# sns.kdeplot((mse_test), fill=True, label='Test Dataset KDE', cut=0) # 테스트 데이터

# plt.legend(fontsize=15)
# plt.title("Histogram and Kernel Density Estimation")
# plt.xlabel('Mean Square Error', fontsize=15, fontweight='bold')
# plt.ylabel('Loss Density', fontsize=15, fontweight='bold')
# plt.xscale('log')
# plt.xlim(1e-5, (1.5)*1e-1)
# plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))

# plt.savefig('./Histogram+KDEplot.png',dpi=600)

# print("추론 결과 : 손실값의 커널 밀도추정 시각화")

# plt.clf() # figure 초기화
# plt.cla() # figure 축 초기화
# plt.close() # 현재 figure 닫기

# ################################ 상자그림(Box Plot)을 이용한 손실값 시각화 ################################
# # LaTeX 스타일 활성화
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.figure(figsize=(10,6))
# data_train = pd.DataFrame({'MSE_train':mse, 'Type': 'Train Dataset'})
# data_test = pd.DataFrame({'MSE_test':mse_test, 'Type': 'Test Dataset'})

# # 박스 그림 그리기
# sns.boxplot(x='Type', y='MSE_train',data=data_train, label='Train Dataset')
# sns.boxplot(x='Type', y='MSE_test',data=data_test, label='Test Dataset')

# # plt.legend(fontsize=25)

# # 박스 그림 축 및 라벨 설정
# plt.title("Train vs. Test Recosntrucion Loss Box plot")
# plt.xlabel('Type', fontsize=15, fontweight='bold')
# plt.ylabel('MSE', fontsize=15, fontweight='bold')
# plt.yscale('log')
# plt.ylim(1e-5, (1.5)*1e-1)
# plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))
# plt.legend(fontsize=15)
# plt.savefig('./BOXplot.png',dpi=600)

# print("추론 결과 : 손실값의 박스 그림 시각화")


# ### 가중치 시각화 ###
# # # 모델 적용
# # x_recon, mu, log_var = model_VAE(x, c)

# # # Attention 가중치 시각화
# # attn_weights = model_VAE.get_attention_weights()
# # visualize_attention_weights(attn_weights, layer_idx=0, head_idx=0)
