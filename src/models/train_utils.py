import argparse

from models.aggregators import MeanAggregator, DotAttentionAggregator
from models.back_bone import BackBone
from models.cnn import BaselineCNN, PretrainedCNN
from models.top_head import FullyConnectedHead, LinearHead, GatedHead


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cnn", type=str, default="baseline",
                        help="cnn name")
    parser.add_argument("-a", "--aggregator", type=str, default="mean",
                        choices=['mean', 'dot'], help="aggregator name")
    parser.add_argument("-t", "--top_head", type=str, default="fc",
                        choices=['fc', 'linear', 'gated'], help="top head name")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="number of epochs")
    parser.add_argument("-s", "--size", type=int, default=16,
                        help="cnn output size")
    parser.add_argument("-nw", "--num_workers", type=int, default=8,
                        help="number of workers")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
                        help="dataset learning rate")
    parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4,
                        help="optimizer weight decay")
    parser.add_argument("-k", "--kfolds", type=float, default=5,
                        help="Number of folds")
    parser.add_argument("-ct", "--cutting_threshold", type=float, default=0.5,
                        help="cutting threshold")
    parser.add_argument("-ts", "--test_size", type=float, default=0.2,
                        help="dataset learning rate")
    parser.add_argument("-d", "--dataset", type=str, default="../3md3070-dlmi/",
                        help="path to the dataset")
    parser.add_argument("-sub", "--submission", type=str, default="../submissions",
                        help="path to submission folder")
    parser.add_argument("-p", "--preprocess", type=bool, default=False, const=True, nargs="?",
                        help="whether or not to add image preprocessing")
    parser.add_argument("-b", "--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("-lw", "--loss_weighting", type=bool, default=False,
                        help="Weight the loss to account unbalance between positive and negative")
    return parser.parse_args()


def build_model(cnn, aggregator, top_head, size, device):
    ### CNN
    if cnn == 'baseline':
        cnn = BaselineCNN(size=size)
    else:
        cnn = PretrainedCNN(size=size, cnn=cnn)

    ### Aggregator
    if aggregator == 'mean':
        aggregator = MeanAggregator()
    elif aggregator == 'dot':
        aggregator = DotAttentionAggregator(size)
    else:
        raise NameError('Invalid aggregator name')

    ### Top head
    if top_head == 'fc':
        top_head = FullyConnectedHead(size)
    elif top_head == 'linear':
        top_head = LinearHead(size)
    elif top_head == "gated":
        top_head = GatedHead(size)
    else:
        raise NameError('Invalid top_head name')

    ### Training

    return BackBone(cnn, aggregator, top_head, device).to(device)