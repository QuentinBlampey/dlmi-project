import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
from datetime import datetime

from models.aggregators import MeanAggregator, DotAttentionAggregator
from models.back_bone import BackBone
from models.cnn import BaselineCNN
from models.top_head import FullyConnectedHead
from dataset import LymphDataset, get_transform


def main(args):
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"> Device: {device}\n")
    
    ### Loaders construction

    files = []
    for dirname, _, filenames in os.walk('../3md3070-dlmi'):
        for filename in filenames:
            files.append(os.path.join(dirname, filename))

    clinical_df = pd.read_csv("../3md3070-dlmi/clinical_annotation.csv", index_col="ID")
    df_test = clinical_df[clinical_df["LABEL"] == -1]
    df = clinical_df[clinical_df["LABEL"] != -1]

    ID_train, ID_val = train_test_split(df.index.values, test_size=0.2, random_state=1, stratify=df.LABEL.values)

    df_train = df.loc[ID_train]
    df_val = df.loc[ID_val]

    path_train = [[file for file in files if p_id + '/' in file] for p_id in df_train.index]
    path_val = [[file for file in files if p_id + '/' in file] for p_id in df_val.index]
    path_test = [[file for file in files if p_id + '/' in file] for p_id in df_test.index]

    train_dst = LymphDataset(path_train, df_train, get_transform(True))
    val_dst = LymphDataset(path_val, df_val, get_transform(False))
    test_dst = LymphDataset(path_test, df_test, get_transform(False))

    train_loader = DataLoader(train_dst, batch_size=1, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dst, batch_size=1, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dst, batch_size=1, shuffle=False, num_workers=args.num_workers)


    ### CNN
    if args.cnn == 'baseline':
        cnn = BaselineCNN(size=args.size)
    else:
        raise NameError('Invalid cnn name')

    ### Aggregator
    if args.aggregator == 'mean':
        aggregator = MeanAggregator()
    elif args.aggregator == 'dot':
        aggregator = DotAttentionAggregator(args.size)
    else:
        raise NameError('Invalid aggregator name')

    ### Top head
    if args.top_head == 'fc':
        top_head = FullyConnectedHead(args.size)
    else:
        raise NameError('Invalid top_head name')

    ### Training

    model = BackBone(cnn, aggregator, top_head, device).to(device)

    pos_weight = torch.tensor([50/113]).to(device)
    loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train_and_eval(train_loader, val_loader, args.epochs, loss_fct, args.learning_rate)

    predictions = model.predict(test_loader)
    path_submission = os.path.join('..', 'submissions', f"{datetime.now().strftime('%y-%m-%d_%Hh%Mm%Ss')}.csv")
    submission = pd.DataFrame({'ID': test_dst.df.index.values,
                   'Prediction': predictions})
    submission.to_csv(path_submission, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cnn", type=str, default="baseline", choices=['baseline'],
        help="cnn name")
    parser.add_argument("-a", "--aggregator", type=str, default="mean",
        choices=['baseline', 'dot'], help="aggregator name")
    parser.add_argument("-t", "--top_head", type=str, default="fc",
        choices=['fc'], help="top head name")
    parser.add_argument("-e", "--epochs", type=int, default=10, 
        help="number of epochs")
    parser.add_argument("-s", "--size", type=int, default=16, 
        help="cnn output size")
    parser.add_argument("-nw", "--num_workers", type=int, default=8, 
        help="number of workers")
    parser.add_argument("-lr", "--learning_rate", type=float, default=2e-4, 
        help="dataset learning rate")

    args = parser.parse_args()
    print(f"> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    main(args)