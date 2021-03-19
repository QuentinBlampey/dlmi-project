import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import LymphDataset, get_transform
from models.train_utils import get_args, build_model


def cross_validate(model_factory, df, files, k, n_epochs, loss_function, learning_rate, weight_decay, num_workers,
                   preprocess, batch_size):
    kf = KFold(k, random_state=0, shuffle=True)
    accuracies = []
    for n, (train_index, val_index) in enumerate(kf.split(df.index.values)):
        print(f"Fold {n + 1}")
        df_train = df.iloc[train_index]
        df_val = df.iloc[val_index]
        path_train = [[file for file in files if p_id + '/' in file] for p_id in df_train.index]
        path_val = [[file for file in files if p_id + '/' in file] for p_id in df_val.index]
        train_dst = LymphDataset(path_train, df_train, get_transform(True), preprocess=preprocess)
        val_dst = LymphDataset(path_val, df_val, get_transform(False), preprocess=preprocess)
        train_loader = DataLoader(train_dst, batch_size=1, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dst, batch_size=1, shuffle=False, num_workers=num_workers)
        model = model_factory()
        val_acc = model.train_and_eval(train_loader, val_loader, n_epochs, loss_function, learning_rate, weight_decay,
                                       [args.lambda1, args.lambda2, args.lambda3], batch_size)
        accuracies.append(val_acc)
    return accuracies


def main(args):
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"> Device: {device}\n")

    ### Loaders construction

    files = []
    for dirname, _, filenames in os.walk(args.dataset):
        for filename in filenames:
            files.append(os.path.join(dirname, filename))

    clinical_df = pd.read_csv(os.path.join(args.dataset, "clinical_annotation.csv"), index_col="ID")
    clinical_df['AGE'] = clinical_df['DOB'].apply(lambda dob: 2021 - int(dob[-4:]))
    clinical_df['GENDER'] = clinical_df['GENDER'].apply(lambda gender: 1 if gender == "M" else -1)

    scaler = StandardScaler()
    clinical_df[['LYMPH_COUNT', 'AGE']] = scaler.fit_transform(clinical_df[['LYMPH_COUNT', 'AGE']])

    df = clinical_df[clinical_df["LABEL"] != -1]

    def model_factory():
        return build_model(args.cnn, args.aggregator, args.top_head, args.size, device)

    if args.loss_weighting:
        pos_weight = torch.tensor([50 / 113]).to(device)
        loss_fct = nn.BCEWithLogitsLoss(pos_weight)
    else:
        loss_fct = nn.BCEWithLogitsLoss()
    accuracies = cross_validate(model_factory, df, files, int(args.kfolds), args.epochs, loss_fct, args.learning_rate,
                                args.weight_decay,
                                args.num_workers, args.preprocess, args.batch_size)
    print(f"\nAverage accuracy: {np.mean(accuracies)}, ({accuracies})")


if __name__ == "__main__":
    args = get_args()
    print(f"> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    main(args)
