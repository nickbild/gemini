import torch
from train import Net


# Load the saved model.
checkpoint = torch.load("md5_499.model")
model = Net()
model.load_state_dict(checkpoint)
model.eval()


def predict_image_class():

    input = torch.FloatTensor([1,2,3,5,1,4,2,1,43,344,1,2,4,5,1,2,4,2,4,1])
    #image_tensor.cuda()

    # Turn the input into a Variable.
    #input = Variable(image_tensor)

    # Predict the class of the image.
    output = model(input)
    print(output)
    #index = output.data.numpy().argmax()
    #score = output[0, index].item()

    #return index, score


if __name__ == "__main__":
    predict_image_class()
