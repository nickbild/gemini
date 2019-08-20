import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from lookups import lookup
from lookups import lookup_reverse
import math


input_seq_length = 6
output_seq_length = 32

data_file_train = "data/train/input_pairs.txt"
num_samples_train = 100000
batch_size_train = 32 # Divisible by num_samples_train.
num_batches_train = int(math.ceil(num_samples_train / batch_size_train))

data_file_test = "data/test/input_pairs.txt"
num_samples_test = 4992
batch_size_test = 32 # Divisible by num_samples_test.
num_batches_test = int(math.ceil(num_samples_test / batch_size_test))


class Net(nn.Sequential):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(in_features=input_seq_length, out_features=512)
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

        self.fc8 = nn.Linear(in_features=1024, out_features=output_seq_length)

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


def test():
    test_acc = 0.0

    for batch_num in range(1,num_batches_test+1):
        range_end = int(batch_num * batch_size_test) - 1
        range_start = range_end - batch_size_test + 1
        range_len = range_end - range_start + 1

        input = test_input.narrow(0, range_start, range_len)
        target = test_solutions.narrow(0, range_start, range_len)

        output = model(input)

        test_acc += count_matches(output, target)

    return test_acc


def train(num_epochs):
    model.train()
    best_acc = 0.0

    for epoch in range(num_epochs):
        train_acc = 0.0
        train_loss = 0.0

        for batch_num in range(1,num_batches_train+1):
            range_end = int(batch_num * batch_size_train) - 1
            range_start = range_end - batch_size_train + 1
            range_len = range_end - range_start + 1

            input = train_input.narrow(0, range_start, range_len)
            target = train_solutions.narrow(0, range_start, range_len)

            # Clear all accumulated gradients.
            optimizer.zero_grad()

            # Predict classes using images from the training set.
            output = model(input)

            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(output, target)

            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_acc += count_matches(output, target)
            train_loss += loss.cpu().data * input.size(0)

        # Call the learning rate adjustment function
        # adjust_learning_rate(epoch)

        # Evaluate on the test set
        test_acc = test()
        test_acc_pct = (test_acc / (num_samples_test * output_seq_length)) * 100

        # Save the model if the test accuracy is greater than our current best.
        if test_acc_pct >= best_acc:
            print("Saving model...")
            torch.save(model.state_dict(), "md5_{}_{}.model".format(epoch, test_acc_pct))
            best_acc = test_acc_pct

        # Print metrics for epoch.
        train_acc_pct = (train_acc / (num_samples_train * output_seq_length)) * 100
        print("Epoch: {}, Train Accuracy: {}, Train Loss: {}, Test Accuracy: {}".format(epoch, train_acc_pct, train_loss, test_acc_pct))


def encode_string(str):
    result = []
    for chr in str:
        result.append(lookup[chr])

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


def count_matches(output, target):
    match = 0
    for r in range(output.shape[0]):
        for c in range(output_seq_length):
            if round(output[r][c].item()) == target[r][c].item():
                match += 1

    return match


def load_data_file_to_tensor(file):
    input = []
    output = []
    for line in open(file):
        line = line.rstrip('\n')
        input.append(encode_string(line.split("\t")[0]))
        output.append(encode_string(line.split("\t")[1]))

    input = torch.FloatTensor(input)
    output = torch.FloatTensor(output)

    if cuda_avail:
        print("Transferring data to GPU...")
        input = Variable(input.cuda())
        output = Variable(output.cuda())

    return input, output


if __name__ == "__main__":
    cuda_avail = torch.cuda.is_available()

    model = Net()

    if cuda_avail:
        print("Transferring model to GPU...")
        model.cuda()

    optimizer = Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    loss_fn = nn.MSELoss()

    train_input, train_solutions = load_data_file_to_tensor(data_file_train)
    test_input, test_solutions = load_data_file_to_tensor(data_file_test)

    print("Starting training.")
    train(100000000)
