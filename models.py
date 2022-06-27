import torch
import torch.nn as nn
import torch.nn.functional as F
#early fusion模型
class Early_fusion(nn.Module):
    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.all_embed = nn.Sequential(
            nn.Linear(video_emb_dim+audio_emb_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 10)
        )
    def forward(self, audio_feat, video_feat):
        audio_emb = audio_feat.mean(1)
        video_emb = video_feat.mean(1)
        embed = torch.cat((audio_emb, video_emb), 1) #(128,1024)
        #先将两个模态的特征进行拼接然后再过一个全连接神经网络
        output = self.all_embed(embed)
        return output
#late_fusion 的模型
class late_fusion(nn.Module):
    def __init__(self, audio_emb_dim, video_emb_dim, num_classes,alpha = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.outputlayer_video = nn.Sequential(
            nn.Linear(128, 256),
            nn.Linear(256, self.num_classes),
        )
        self.outputlayer_audio = nn.Sequential(
            nn.Linear(128, 256),
            nn.Linear(256, self.num_classes),
        )
    
    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)
        audio_emb = self.audio_embed(audio_emb)
        output_audio = self.outputlayer_audio(audio_emb)
        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)
        output_video = self.outputlayer_video(video_emb)
        #将两个子网络的输出按照权重比例进行融合
        output = (1-self.alpha)*output_audio + self.alpha*output_video
        return output
#baseline
class MeanConcatDense(nn.Module):
    
    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.outputlayer = nn.Sequential(
            nn.Linear(128, 128),
            nn.Linear(128, self.num_classes),
        )
    
    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)
        audio_emb = self.audio_embed(audio_emb)
        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)
        #单独使用视频特征进行分类
        embed = video_emb
        output = self.outputlayer(embed)
        return output
#LSTM特征处理的模型
class Mynet(nn.Module):
    
    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.rnn1 =  nn.LSTM(512,256,num_layers=2,bidirectional=True,batch_first=True)
        self.rnn2 =  nn.LSTM(512,256,num_layers=2,bidirectional=True,batch_first=True)
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.outputlayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
    
    def forward(self, audio_feat, video_feat):
        #将音频特征通过LSTM
        audio_emb,_ = self.rnn1(audio_feat)
        audio_emb = torch.flatten(audio_emb,1)
        audio_emb = self.audio_embed(audio_emb)
        #将视频特征通过LSTM
        video_emb,_ = self.rnn2(video_feat)
        video_emb = torch.flatten(video_emb,1)
        video_emb = self.video_embed(video_emb)
        
        embed = torch.cat((audio_emb, video_emb), 1)

        output = self.outputlayer(embed)
        return output

