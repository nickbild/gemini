import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
import math


save_models = True
input_seq_length = 1
output_seq_length = 1
learning_rate = 0.0001

data_file_train = "data/train/input_pairs.txt"
num_samples_train = 500
batch_size_train = 50 # Divisible by num_samples_train.
num_batches_train = int(math.ceil(num_samples_train / batch_size_train))

data_file_test = "data/test/input_pairs.txt"
num_samples_test = 500
batch_size_test = 50 # Divisible by num_samples_test.
num_batches_test = int(math.ceil(num_samples_test / batch_size_test))

# Track recent training error values.
error_history = []
for i in range(2):
    error_history.append(0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Layers
        self.linear_input_to_64 = nn.Linear(in_features=input_seq_length, out_features=64)
        self.linear_64_to_128 = nn.Linear(in_features=64, out_features=128)
        self.linear_128_to_128 = nn.Linear(in_features=128, out_features=128)
        self.linear_128_to_192 = nn.Linear(in_features=128, out_features=192)
        self.linear_192_to_256 = nn.Linear(in_features=192, out_features=256)
        self.linear_256_to_128 = nn.Linear(in_features=256, out_features=128)
        self.linear_128_to_output = nn.Linear(in_features=128, out_features=output_seq_length)

        # Activations
        self.leaky_relu = nn.LeakyReLU()


    def forward(self, input):
        output = self.linear_input_to_64(input)
        output = self.leaky_relu(output)

        output = self.linear_64_to_128(output)
        output = self.leaky_relu(output)

        output = self.linear_128_to_128(output)
        output = self.leaky_relu(output)

        output = self.linear_128_to_128(output)
        output = self.leaky_relu(output)

        output = self.linear_128_to_128(output)
        output = self.leaky_relu(output)

        output = self.linear_128_to_192(output)
        output = self.leaky_relu(output)

        output = self.linear_192_to_256(output)
        output = self.leaky_relu(output)

        output = self.linear_256_to_128(output)
        output = self.leaky_relu(output)

        output = self.linear_128_to_output(output)

        return output


def test():
    model.eval()
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


def train(num_epochs, train_input, train_solutions, test_input, test_solutions):
    best_acc = 90.0

    for epoch in range(num_epochs):
        model.train()

        train_acc = 0.0
        train_loss = 0.0

        if epoch % 100 == 0:
            rate = update_learning_rate()
            for param_group in optimizer.param_groups:
                param_group["lr"] = rate
                print("Learning rate set to: {}".format(param_group["lr"]))

        # Shuffle training data.
        idx = torch.randperm(train_input.nelement())
        train_input = train_input.view(-1)[idx].view(train_input.size())
        train_solutions = train_solutions.view(-1)[idx].view(train_solutions.size())

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

        error_history.append(train_loss.item())
        error_history.pop(0)

        loss_reduction = adjust_learning_rate(learning_rate, error_history)
        # scheduler.step(train_loss)

        # Evaluate on the test set
        test_acc = test()
        test_acc_pct = (test_acc / num_samples_test) * 100

        # Save the model if the test accuracy is greater than our current best.
        if test_acc_pct >= best_acc and save_models:
            print("Saving model...")
            torch.save(model.state_dict(), "gemini_{}_{}.model".format(epoch, test_acc_pct))
            best_acc = test_acc_pct

        # Print metrics for epoch.
        train_acc_pct = (train_acc / num_samples_train) * 100
        print("Epoch: {},\tTrain Accuracy: {},\tTrain Loss: {},\tLoss Reduction: {},\tTest Accuracy: {}".format(epoch, round(train_acc_pct, 3), round(train_loss.item(), 8), round(loss_reduction,3), round(test_acc_pct, 3)))


def adjust_learning_rate(rate, error_history):
    # Adapt learning rate based on rate of error change.
    mean1 = error_history[0]
    mean2 = error_history[1]
    reduction = mean1 - mean2

    # if reduction < 0.0001:
    #     rate = rate / 10
    # if mean2 < 0.0003:
    #     rate = rate / 1000
    # elif mean2 < 0.00088:
    #     rate = rate / 100
    # elif mean2 < 0.003:
    #     rate = rate / 10

    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = rate

    return reduction


def count_matches(output, target):
    match = 0
    for r in range(output.shape[0]):
        if round(output[r].item()) == round(target[r].item()):
            match += 1

    return match


def load_data_file_to_tensor(file):
    input = []
    output = []
    for line in open(file):
        line = line.rstrip('\n')
        input.append(float(line.split("\t")[0]))
        output.append(float(line.split("\t")[1]))

    input = torch.FloatTensor(input).view(-1,1)
    output = torch.FloatTensor(output).view(-1,1)

    if cuda_avail:
        print("Transferring data to GPU...")
        input = Variable(input.cuda())
        output = Variable(output.cuda())

    return input, output


# Allow for manual updates to learning rate during training.
def update_learning_rate():
    rate = open("learning_rate.txt").readline().strip()

    return float(rate)


if __name__ == "__main__":
    cuda_avail = torch.cuda.is_available()

    model = Net()

    if cuda_avail:
        print("Transferring model to GPU...")
        model.cuda()

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    loss_fn = nn.SmoothL1Loss()
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, cooldown=20, factor=0.1, verbose=True)

    train_input, train_solutions = load_data_file_to_tensor(data_file_train)
    test_input, test_solutions = load_data_file_to_tensor(data_file_test)

    print("Starting training.")
    train(100000000000, train_input, train_solutions, test_input, test_solutions)
