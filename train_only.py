#本文件将训练和测试进行了合并
import imp
from pathlib import Path
import os
import argparse
import pdb
import time
import random

from tqdm import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import SceneDataset 
import models
import utils
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import wandb
wandb.init(project="av_class", entity="zzh12138")
parser = argparse.ArgumentParser(description='training networks')
parser.add_argument('--config_file', type=str,default='configs/baseline.yaml', required=False)
parser.add_argument('--seed', type=int, default=0, required=False,
                    help='set the seed to reproduce result')
parser.add_argument('--alpha', type=float, default=0.5, required=False,
                    help='set the seed to reproduce result')
args = parser.parse_args()
wandb.config = {
  "alpha": args.alpha,
  "seed": args.seed
}
wandb.init(config=args)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(args.config_file, "r") as reader:
    config = yaml.load(reader, Loader=yaml.FullLoader)

mean_std_audio = np.load(config["data"]["audio_norm"])
mean_audio = mean_std_audio["global_mean"]
std_audio = mean_std_audio["global_std"]
mean_std_video = np.load(config["data"]["video_norm"])
mean_video = mean_std_video["global_mean"]
std_video = mean_std_video["global_std"]

audio_transform = lambda x: (x - mean_audio) / std_audio
video_transform = lambda x: (x - mean_video) / std_video

tr_ds = SceneDataset(config["data"]["train"]["audio_feature"],
                     config["data"]["train"]["video_feature"],
                     audio_transform,
                     video_transform)
tr_dataloader = DataLoader(tr_ds, shuffle=True, **config["data"]["dataloader_args"])

cv_ds = SceneDataset(config["data"]["cv"]["audio_feature"],
                     config["data"]["cv"]["video_feature"],
                     audio_transform,
                     video_transform)
cv_dataloader = DataLoader(cv_ds, shuffle=False, **config["data"]["dataloader_args"])

model = models.Mynet(49152, 49152, config["num_classes"])
print(model)

output_dir = 'experiments\lstm'
Path(output_dir).mkdir(exist_ok=True, parents=True)
logging_writer = utils.getfile_outlogger(os.path.join(output_dir, "train.log"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = getattr(optim, config["optimizer"]["type"])(
    model.parameters(),
    **config["optimizer"]["args"])

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    **config["lr_scheduler"]["args"])

print('-----------start training-----------')


def train(epoch):
    model.train()
    train_loss = 0.
    start_time = time.time()
    count = len(tr_dataloader) * (epoch - 1)
    loader = tqdm(tr_dataloader)
    for batch_idx, batch in enumerate(loader):
        count = count + 1
        audio_feat = batch["audio_feat"].to(device)
        video_feat = batch["video_feat"].to(device)
        target = batch["target"].to(device)

        # training
        optimizer.zero_grad()

        logit = model(audio_feat, video_feat)
        loss = loss_fn(logit, target)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} |'.format(
                epoch, batch_idx + 1, len(tr_dataloader),
                elapsed * 1000 / (batch_idx + 1), loss.item()))

    train_loss /= (batch_idx + 1)
    logging_writer.info('-' * 99)
    logging_writer.info('| epoch {:3d} | time: {:5.2f}s | training loss {:5.2f} |'.format(
        epoch, (time.time() - start_time), train_loss))
    return train_loss

def validate(epoch):
    model.eval()
    validation_loss = 0.
    start_time = time.time()
    # data loading
    cv_loss = 0.
    targets = []
    preds = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(cv_dataloader):
            audio_feat = batch["audio_feat"].to(device)
            video_feat = batch["video_feat"].to(device)
            target = batch["target"].to(device)
            logit = model(audio_feat, video_feat)
            loss = loss_fn(logit, target)
            pred = torch.argmax(logit, 1)
            targets.append(target.cpu().numpy())
            preds.append(pred.cpu().numpy())
            cv_loss += loss.item()

    cv_loss /= (batch_idx+1)
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    accuracy = accuracy_score(targets, preds)
    logging_writer.info('| epoch {:3d} | time: {:5.2f}s | cv loss {:5.2f} | cv accuracy: {:5.2f} |'.format(
            epoch, time.time() - start_time, cv_loss, accuracy))
    logging_writer.info('-' * 99)

    return cv_loss


training_loss = []
cv_loss = []


with open(os.path.join(output_dir, 'config.yaml'), "w") as writer:
    yaml.dump(config, writer, default_flow_style=False)

not_improve_cnt = 0
for epoch in range(1, config["epoch"]):
    print('epoch', epoch)
    training_loss.append(train(epoch))
    cv_loss.append(validate(epoch))
    print('-' * 99)
    print('epoch', epoch, 'training loss: ', training_loss[-1], 'cv loss: ', cv_loss[-1])

    if cv_loss[-1] == np.min(cv_loss):
        # save current best model
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
        print('best validation model found and saved.')
        print('-' * 99)
        not_improve_cnt = 0
    else:
        not_improve_cnt += 1
    
    lr_scheduler.step(cv_loss[-1])
    
    if not_improve_cnt == config["early_stop"]:
        break


minmum_cv_index = np.argmin(cv_loss)
minmum_loss = np.min(cv_loss)
plt.plot(training_loss, 'r')
#plt.hold(True)
plt.plot(cv_loss, 'b')
plt.axvline(x=minmum_cv_index, color='k', linestyle='--')
plt.plot(minmum_cv_index, minmum_loss,'r*')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.savefig(os.path.join(output_dir, 'loss.png'))
torch.multiprocessing.set_sharing_strategy('file_system')

with open(os.path.join(output_dir, "config.yaml"), "r") as reader:
    config = yaml.load(reader, Loader=yaml.FullLoader)

mean_std_audio = np.load(config["data"]["audio_norm"])
mean_std_video = np.load(config["data"]["video_norm"])
mean_audio = mean_std_audio["global_mean"]
std_audio = mean_std_audio["global_std"]
mean_video = mean_std_video["global_mean"]
std_video = mean_std_video["global_std"]

audio_transform = lambda x: (x - mean_audio) / std_audio
video_transform = lambda x: (x - mean_video) / std_video

tt_ds = SceneDataset(config["data"]["test"]["audio_feature"],
                     config["data"]["test"]["video_feature"],
                     audio_transform,
                     video_transform)
config["data"]["dataloader_args"]["batch_size"] = 1
tt_dataloader = DataLoader(tt_ds, shuffle=False, **config["data"]["dataloader_args"])
model = models.Mynet(49152, 49152, config["num_classes"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(
    os.path.join(output_dir, "best_model.pt"), "cpu")
)

model = model.to(device).eval()

targets = []
probs = []
preds = []
aids = []

with torch.no_grad():
    tt_dataloader = tqdm(tt_dataloader)
    for batch_idx, batch in enumerate(tt_dataloader):
        audio_feat = batch["audio_feat"].to(device)
        video_feat = batch["video_feat"].to(device)
        target = batch["target"].to(device)
        logit = model(audio_feat, video_feat)
        pred = torch.argmax(logit, 1)
        targets.append(target.cpu().numpy())
        probs.append(torch.softmax(logit, 1).cpu().numpy())
        preds.append(pred.cpu().numpy())
        aids.append(np.array(batch["aid"]))


targets = np.concatenate(targets, axis=0)
preds = np.concatenate(preds, axis=0)
probs = np.concatenate(probs, axis=0)
aids = np.concatenate(aids, axis=0)

writer = open(os.path.join(output_dir, "result.txt"), "w")
cm = confusion_matrix(targets, preds)
keys = ['airport',
        'bus',
        'metro',
        'metro_station',
        'park',
        'public_square',
        'shopping_mall',
        'street_pedestrian',
        'street_traffic',
        'tram']
dic_false = {}
counter = 0
for key in keys:
    dic_false[key] = 0
scenes_pred = [keys[pred] for pred in preds]
scenes_label = [keys[target] for target in targets]
for i in range(len(scenes_pred)):
    if scenes_label[i] == 'bus' and scenes_pred[i]!= 'bus':
        dic_false[scenes_pred[i]] += 1
        counter += 1
for key in dic_false.keys():
    dic_false[key]/=counter
print(dic_false)
pred_dict = {"aid": aids, "scene_pred": scenes_pred, "scene_label": scenes_label}
for idx, key in enumerate(keys):
    pred_dict[key] = probs[:, idx]
pd.DataFrame(pred_dict).to_csv(os.path.join(output_dir, "prediction.csv"),
                               index=False,
                               sep="\t",
                               float_format="%.3f")


print(classification_report(targets, preds, target_names=keys), file=writer)

df_cm = pd.DataFrame(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
    index=keys, columns=keys)
plt.figure(figsize=(15, 12))
sn.heatmap(df_cm, annot=True)
plt.savefig(os.path.join(output_dir, 'cm.png'))

acc = accuracy_score(targets, preds)
wandb.log({"average_acc": acc})
print('  ', file=writer)
print(f'accuracy: {acc:.3f}', file=writer)
logloss = log_loss(targets, probs)
print(f'overall log loss: {logloss:.3f}', file=writer)
