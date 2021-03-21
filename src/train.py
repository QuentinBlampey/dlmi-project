import json
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import LymphDataset, get_transform
from train_utils import get_args, build_model


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

    df_test = clinical_df[clinical_df["LABEL"] == -1]
    df = clinical_df[clinical_df["LABEL"] != -1]

    path_train = [[file for file in files if p_id + '/' in file] for p_id in df.index]
    path_test = [[file for file in files if p_id + '/' in file] for p_id in df_test.index]

    train_dst = LymphDataset(path_train, df, get_transform(True), preprocess=args.preprocess)
    test_dst = LymphDataset(path_test, df_test, get_transform(False), preprocess=args.preprocess)

    train_loader = DataLoader(train_dst, batch_size=1, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dst, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.cnn, args.aggregator, args.top_head, args.size, device)

    if args.loss_weighting:
        pos_weight = torch.tensor([50 / 113]).to(device)
        loss_fct = nn.BCEWithLogitsLoss(pos_weight)
    else:
        loss_fct = nn.BCEWithLogitsLoss()
    model.train_only(train_loader, args.epochs, loss_fct, args.learning_rate, args.weight_decay, [args.lambda1, args.lambda2, args.lambda3], args.cutting_threshold, args.batch_size)

    predictions = model.predict(test_loader, args.cutting_threshold)
    path_submission = os.path.join(args.submission, f"{datetime.now().strftime('%y-%m-%d_%Hh%Mm%Ss')}.csv")
    print('\npath_submission:', path_submission)
    submission = pd.DataFrame({'ID': test_dst.df.index.values,
                               'Predicted': predictions})
    submission.to_csv(path_submission, index=False)


if __name__ == "__main__":
    args = get_args()
    print(f"> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    main(args)
