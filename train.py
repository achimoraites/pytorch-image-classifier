import torch
import argparse
from data_loader import load_dir
from classifier import Classifier


def main():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description="Train your Image classifier")
    parser.add_argument('data_dir', action="store", type=str,
                        default="flowers", help='path to the folder of images')
    parser.add_argument('--arch', type=str, default='alexnet',
                        help='CCN Model architecture to use', choices=['vgg16', 'alexnet'])
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='set the folder that will be used to save the checkpoints')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='set the learning rate')
    parser.add_argument('--hidden_units', type=int, default=1024,
                        help='set the hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=1,
                        help='set the number of epochs')
    parser.add_argument('--gpu', action='store_true', help='train on gpu')
    in_arg = parser.parse_args()

    # DATA LOADING
    dataloaders, image_datasets = load_dir(in_arg.data_dir)

    # Use GPU if it's available
    if in_arg.gpu:
        print('Training using gpu...')
        device = torch.device("cuda")
    else:
        print('Training using cpu...')
        device = torch.device("cpu")

    # MODEL
    classifier = Classifier(device=device, arch=in_arg.arch,
                            hidden_layers=in_arg.hidden_units)

    # TRAIN
    classifier.train(dataloaders, in_arg.learning_rate,
                     in_arg.epochs, len(dataloaders['train']))

    # SAVE
    class_to_idx = image_datasets['train'].class_to_idx
    classifier.save_to_checkpoint(class_to_idx, in_arg.save_dir)


# Call to main function to run the program
if __name__ == "__main__":
    main()
