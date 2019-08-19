import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from lookups import lookup
from lookups import lookup_reverse
import math


num_train_cases = 100
batch_size = 32
num_batches = int(math.ceil(num_train_cases / batch_size))


class Net(nn.Sequential):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(in_features=10, out_features=512)
        self.relu1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(in_features=512, out_features=1024)
        self.relu2 = nn.LeakyReLU()

        self.fc3 = nn.Linear(in_features=1024, out_features=1024)
        self.relu3 = nn.LeakyReLU()

        self.fc4 = nn.Linear(in_features=1024, out_features=1024)
        self.relu4 = nn.LeakyReLU()

        self.fc5 = nn.Linear(in_features=1024, out_features=2048)
        self.relu5 = nn.LeakyReLU()

        self.fc6 = nn.Linear(in_features=2048, out_features=4096)
        self.relu6 = nn.LeakyReLU()

        self.fc7 = nn.Linear(in_features=4096, out_features=1024)
        self.relu7 = nn.LeakyReLU()

        self.fc8 = nn.Linear(in_features=1024, out_features=32)

    def forward(self, input):
        output = self.fc1(input)
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = self.relu5(output)

        output = self.fc6(output)
        output = self.relu6(output)

        output = self.fc7(output)
        output = self.relu7(output)

        output = self.fc8(output)

        return output


def train(num_epochs):
    model.train()
    best_acc = 0.0
    best_acc_train = 0.0

    for epoch in range(num_epochs):
        train_acc = 0.0
        train_loss = 0.0

        # for i, (images, labels) in enumerate(train_loader):
        #     if cuda_avail:
        #         images = Variable(images.cuda())
        #         labels = Variable(labels.cuda())

        # images = torch.FloatTensor([6,7,8,6,8,65,7,8,9,0,6,5,4,6,7,8,9,8,7,6])
        # labels = torch.FloatTensor([7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9])

        #images = torch.FloatTensor([[1,2,3,5,1,4,2,1,43,344,1,2,4,5,1,2,4,2,4,1],[6,7,8,6,8,65,7,8,9,0,6,5,4,6,7,8,9,8,7,6]])
        #labels = torch.FloatTensor([[2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3,3,3],[7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9]])

        for batch_num in range(1,num_batches+1):
            range_end = int(batch_num * batch_size) - 1
            range_start = range_end - batch_size + 1

            inp = train_input[range_start:range_end+1]
            targ = train_solutions[range_start:range_end+1]

            inp = torch.FloatTensor(inp)
            targ = torch.FloatTensor(targ)

            if cuda_avail:
                inp = Variable(inp.cuda())
                targ = Variable(targ.cuda())

            # inp = "this is a test..aabc"
            # targ = "c72bd8e501effc3679f403da1534b297"

            # images = encode_string(inp)
            # labels = encode_string(targ)

            # Clear all accumulated gradients.
            optimizer.zero_grad()

            # Predict classes using images from the training set.
            outputs = model(inp)

            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, targ)

            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            # print(outputs)
            # print(targ)

            train_acc += calculate_accuracy(outputs, targ)
            train_loss += loss.cpu().data * inp.size(0)






        if epoch == (num_epochs-1):
            torch.save(model.state_dict(), "md5_{}.model".format(epoch))
            print("Model saved.")

        # Call the learning rate adjustment function
        # adjust_learning_rate(epoch)

        # Evaluate on the test set
        #test_acc = test()

        # Save the model if the test acc is greater than our current best
        # if test_acc >= best_acc or train_acc >= best_acc_train:
        #     save_models(epoch, train_acc, test_acc)
        #     best_acc = test_acc
        #     best_acc_train = train_acc

        # Print metrics for epoch.
        print("Epoch {}, Train Accuracy: {}, TrainLoss: {}".format(epoch, train_acc, train_loss)) #, test_acc))


def encode_string(str):
    result = []
    for chr in str:
        result.append(lookup[chr])

    #result = torch.FloatTensor(result)

    return result


def decode_tensor(tensor):
    result = ""
    for t in tensor:
        for chr in t.tolist():
            try:
                result += lookup_reverse[round(chr)]
            except KeyError as e:
                result += " "

    return result


def calculate_accuracy(output, target):
    total = 0
    match = 0
    for r in range(output.shape[0]):
        for c in range(32):
            total += 1
            if round(output[r][c].item()) == target[r][c].item():
                match += 1

    return (match  / total) * 100


def load_data_file(file):
    input = []
    output = []
    for line in open(file):
        line = line.rstrip('\n')
        input.append(encode_string(line.split("\t")[0]))
        output.append(encode_string(line.split("\t")[1]))

    return input, output


if __name__ == "__main__":
    cuda_avail = torch.cuda.is_available()

    model = Net()

    if cuda_avail:
        model.cuda()

    optimizer = Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    loss_fn = nn.MSELoss()

    train_input, train_solutions = load_data_file("data/train/input_pairs.txt")

    print("Starting training.")
    train(10000000)
