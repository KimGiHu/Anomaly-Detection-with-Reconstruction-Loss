import torch
import torch.nn as nn

# CVAE 모델 정의
class Encoder_rev(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim, dropout_prob=0.2):
        super(Encoder_rev, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=12, padding=6)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.pool1 = nn.MaxPool1d(kernel_size=12, stride=2)
        self.conv1_dropout = nn.Dropout(p=dropout_prob)

        self.conv2 = nn.Conv1d(128, 128, kernel_size=12, padding=6)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.pool2 = nn.MaxPool1d(kernel_size=12, stride=2)
        self.conv2_dropout = nn.Dropout(p=dropout_prob)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=12, padding=6)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.pool3 = nn.MaxPool1d(kernel_size=12, stride=2)
        self.conv3_dropout = nn.Dropout(p=dropout_prob)

        def calc_seq_len(hidden_dim):
            tmp = (hidden_dim + 2 * 6 - 12) // 1 + 1  # conv1
            tmp = (tmp - 12) // 2 + 1           # pool1
            tmp = (tmp + 2 * 6 - 12) // 1 + 1  # conv2
            tmp = (tmp - 12) // 2 + 1           # pool2
            tmp = (tmp + 2 * 6 - 12) // 1 + 1  # conv3
            tmp = (tmp - 12) // 2 + 1           # pool3
            return tmp

        self.seq_len = calc_seq_len(hidden_dim)
        self.fc = nn.Linear(128 * (self.seq_len), latent_dim)

        self.bn0 = nn.BatchNorm1d(num_features=(latent_dim+condition_dim))
        self.fc_mu = nn.Linear(latent_dim + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim + condition_dim, latent_dim)


    def forward(self, x, c):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.gelu(x)
        x = self.pool1(x)
        x = self.conv1_dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.gelu(x)
        x = self.pool2(x)
        x = self.conv2_dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.gelu(x)
        x = self.pool3(x)
        x = self.conv3_dropout(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.dropout(x)

        x = torch.cat([x, c], dim=-1)
        x = self.bn0(x)
        x = torch.nn.functional.gelu(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder_rev(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, condition_dim, dropout_prob=0.2):
        super(Decoder_rev, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, 512 )
        self.fc2 = nn.Linear(512, 128 * hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.bn0 = nn.BatchNorm1d(num_features=128 * hidden_dim)

        self.up1 = nn.Upsample(scale_factor=2)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.trans_conv1 = nn.ConvTranspose1d(128, 128, kernel_size=12, padding=6)
        self.conv1_dropout = nn.Dropout(p=dropout_prob)

        self.up2 = nn.Upsample(scale_factor=2)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.trans_conv2 = nn.ConvTranspose1d(128, 128, kernel_size=12, padding=6)
        self.conv2_dropout = nn.Dropout(p=dropout_prob)

        self.up3 = nn.Upsample(size=5002-9)  # 원하는 출력 seq_len 설정
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.trans_conv3 = nn.ConvTranspose1d(128, output_dim, kernel_size=12, padding=1)
        self.conv3_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, z, c):
        z = torch.cat([z, c], dim=-1)
        h = self.fc1(z)
        h = self.fc2(h)
        h = self.bn0(h)
        h = h.view(h.size(0), 128, -1)
        h = torch.nn.functional.gelu(h)
        # h = self.dropout(h)
        

        h = self.up1(h)
        h = self.bn1(h)
        h = torch.nn.functional.gelu(self.trans_conv1(h))
        h = self.conv1_dropout(h)

        h = self.up2(h)
        h = self.bn2(h)
        h = torch.nn.functional.gelu(self.trans_conv2(h))
        h = self.conv2_dropout(h)

        h = self.up3(h)
        h = self.bn3(h)
        x_recon = torch.sigmoid(self.trans_conv3(h))
        
        return x_recon

class CVAE_rev(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, condition_dim, dropout_prob=0.2):
        super(CVAE_rev, self).__init__()
        self.encoder = Encoder_rev(input_dim, hidden_dim, latent_dim, condition_dim, dropout_prob)
        self.decoder = Decoder_rev(latent_dim, self.encoder.seq_len, input_dim, condition_dim, dropout_prob)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar + 1e-12)
        eps = torch.randn_like(std)
        return mu + eps * std + 1e-12

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar
