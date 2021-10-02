import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models

from PIL import Image
import numpy as np

from data_loader import image_tranformer


class Classifier:
    def __init__(self, device, arch='vgg16',  hidden_layers=1024,  outputs=102):

        self.device = device
        self.arch = arch
        self.hidden_layers = hidden_layers
        self.outputs = outputs

        if arch == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.inputs = 25088
        elif arch == 'alexnet':
            self.model = models.alexnet(pretrained=True)
            self.inputs = 9216
        else:
            raise Exception(
                "Arch type of {} is not supported !!!".format(arch))

        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = self.create_model_classifier()
        self.model.to(device)

    def create_model_classifier(self):

        return nn.Sequential(nn.Linear(self.inputs, self.hidden_layers),
                             nn.ReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(self.hidden_layers, self.outputs),
                             nn.LogSoftmax(dim=1))

    def train(self, dataloaders, learning_rate=0.001, epochs=1, print_every=20):
        self.learning_rate = learning_rate
        self.epochs = epochs

        criterion = nn.NLLLoss()
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(
            self.model.classifier.parameters(), lr=learning_rate)

        steps = 0
        running_loss = 0
        for epoch in range(epochs):
            for inputs, labels in dataloaders['train']:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                logps = self.model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in dataloaders['valid']:
                            inputs, labels = inputs.to(
                                self.device), labels.to(self.device)
                            logps = self.model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(
                                equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                          f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                    running_loss = 0
                    self.model.train()
        self.optimizer = optimizer

    def predict(self, image_path, topk=5):

        image = Image.open(image_path)
        tranformer = image_tranformer(resize=256)
        image = tranformer(image)
        image = image.unsqueeze_(0)
        image = image.float()
        inputs = image.to(self.device)

        with torch.no_grad():
            output = self.model.forward(inputs)

        results = F.softmax(output.data, dim=1)

        top_predictions = results.cpu().topk(topk)

        results = list()
        for pred in top_predictions:
            results.append(pred.data.numpy().squeeze().tolist())

        return tuple(results)

    def save_to_checkpoint(self, class_to_idx, folder_path='checkpoints'):
        self.model.class_to_idx = class_to_idx

        checkpoint = {
            'arch': self.arch,
            'hidden_layers': self.hidden_layers,
            'outputs': self.outputs,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'classifier': self.model.classifier,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict(),
            'class_to_idx': self.model.class_to_idx,
        }

        torch.save(checkpoint, folder_path + '/' +
                   self.arch + '_checkpoint.pth')


def load_from_checkpoint(file_path, device):

    if device == torch.device("cuda"):
        def map_location(storage, loc): return storage.cuda()
    else:
        map_location = 'cpu'

    checkpoint = torch.load(file_path,  map_location=map_location)
    classifier = Classifier(
        device, checkpoint['arch'], checkpoint['hidden_layers'], checkpoint['outputs'])
    classifier.model.classifier = checkpoint['classifier']
    classifier.model.load_state_dict(checkpoint['state_dict'], strict=False)
    classifier.model.class_to_idx = checkpoint['class_to_idx']

    return classifier
