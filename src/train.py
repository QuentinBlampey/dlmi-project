import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import argparse

from trainer import BaselineModel
from models import BaselineNN
from dataset import LymphDataset, get_transform


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    files = []
    for dirname, _, filenames in os.walk('../3md3070-dlmi'):
        for filename in filenames:
            files.append(os.path.join(dirname, filename))

    clinical_df = pd.read_csv("../3md3070-dlmi/clinical_annotation.csv", index_col="ID")
    clinical_df = clinical_df[clinical_df["LABEL"] != -1]

    ids = list(clinical_df.index)

    ID_train, ID_test = train_test_split(ids, test_size=0.1, random_state=0)
    ID_train, ID_val = train_test_split(ID_train, test_size=0.2, random_state=1)

    df_train = clinical_df.loc[ID_train]
    df_val = clinical_df.loc[ID_val]
    df_test = clinical_df.loc[ID_test]

    path_train = [[file for file in files if p_id + '/' in file] for p_id in ID_train]
    path_val = [[file for file in files if p_id + '/' in file] for p_id in ID_val]
    path_test = [[file for file in files if p_id + '/' in file] for p_id in ID_test]

    train_dst = LymphDataset(path_train, df_train["LYMPH_COUNT"].values,df_train["GENDER"].values, df_train["DOB"].values, df_train["LABEL"], get_transform(True))
    val_dst = LymphDataset(path_val, df_val["LYMPH_COUNT"].values,df_val["GENDER"].values, df_val["DOB"].values, df_val["LABEL"], get_transform(False))
    test_dst = LymphDataset(path_test, df_test["LYMPH_COUNT"].values,df_test["GENDER"].values, df_test["DOB"].values, df_test["LABEL"], get_transform(False))

    # Data loaders
    torch.manual_seed(1)

    data_loader = torch.utils.data.DataLoader(
        train_dst, batch_size=1, shuffle=True, num_workers=8)

    val_data_loader = torch.utils.data.DataLoader(
        val_dst, batch_size=1, shuffle=True, num_workers=8)

    test_data_loader = torch.utils.data.DataLoader(
        test_dst, batch_size=1, shuffle=False, num_workers=8)

    if args.model == 'BaselineNN':
        model = BaselineModel(BaselineNN(), device)
    else:
        raise NameError('Invalid model name')

    loss_fct = nn.BCELoss()
    model.train(data_loader, val_data_loader, 10, loss_fct, 0.0002)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="BaselineNN", help="model name")

    main(parser.parse_args())