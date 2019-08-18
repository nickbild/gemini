import torch
from train import Net
from lookups import lookup
from lookups import lookup_reverse


# Load the saved model.
checkpoint = torch.load("md5_499.model")
model = Net()
model.load_state_dict(checkpoint)
model.eval()


def predict_image_class():
    input = encode_string("this is a test..aabc")
    #input = torch.FloatTensor([1,2,3,5,1,4,2,1,43,344,1,2,4,5,1,2,4,2,4,1])
    #image_tensor.cuda()

    # Turn the input into a Variable.
    #input = Variable(image_tensor)

    # Predict the class of the image.
    output = model(input)
    print(decode_tensor(output))
    #index = output.data.numpy().argmax()
    #score = output[0, index].item()

    #return index, score


def encode_string(str):
    result = []
    for chr in str:
        result.append(lookup[chr])

    result = torch.FloatTensor(result)

    return result


def decode_tensor(tensor):
    result = ""
    for chr in tensor.tolist():
        try:
            result += lookup_reverse[round(chr)]
        except KeyError as e:
            result += ""

    return result


if __name__ == "__main__":
    predict_image_class()
