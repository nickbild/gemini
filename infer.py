import sys
import torch
from train import Net
from lookups import lookup
from lookups import lookup_reverse


# Load the saved model.
checkpoint = torch.load("gemini_161960_96.0.model")
model = Net()
model.load_state_dict(checkpoint)
model.eval()


def predict_output(input):
    input = torch.FloatTensor(input).view(-1,1)
    output = model(input)

    return output.item()


if __name__ == "__main__":
    input = int(sys.argv[1])

    output = round(predict_output([input]))

    print(output)
