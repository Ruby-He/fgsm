from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Download MNIST datasets
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# There are only three inputs for this tutorial, and are defined as follows:
# 
# -  **epsilons** - List of epsilon values to use for the run. It is important to keep 0 in the list because it represents the model performance on the original test set. 
#    Also, intuitively we would expect the larger the epsilon, the more noticeable the perturbations but the more effective the attack in terms of degrading model
#    accuracy. Since the data range here is :math:`[0,1]`, no epsilon value should exceed 1.
# 
# -  **pretrained_model** - path to the pretrained MNIST model which was trained already
#
# -  **use_cuda** - boolean flag to use CUDA if desired and available

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "data/model/lenet_mnist_model.pth"
use_cuda=True

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

# MNIST Test dataset and dataloader declaration
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
#             transforms.ToTensor(),
#             ])), 
#         batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist/', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            ])), 
        batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
# model.state_dict()返回的是一个OrderDict，存储了网络结构的名字和对应的参数，下面看看源代码如何实现的
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers.一般test阶段使用model.eval()，train阶段用model.train()
model.eval()


######################################################################
# FGSM Attack
# ~~~~~~~~~~~
# 
# Now, we can define the function that creates the adversarial examples by
# perturbing the original inputs. The ``fgsm_attack`` function takes three
# inputs, *image* is the original clean image (:math:`x`), *epsilon* is
# the pixel-wise perturbation amount (:math:`\epsilon`), and *data_grad*
# is gradient of the loss w.r.t the input image
# (:math:`\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y)`). The function
# then creates perturbed image as
# 
# .. math:: perturbed\_image = image + epsilon*sign(data\_grad) = x + \epsilon * sign(\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y))
# 
# Finally, in order to maintain the original range of the data, the
# perturbed image is clipped to range :math:`[0,1]`.
# 

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


######################################################################
# Testing Function
# ~~~~~~~~~~~~~~~~
# 
# Finally, the central result of this tutorial comes from the ``test`` function. Each call to this test function performs a full test step on
# the MNIST test set and reports a final accuracy. 
#
# However, notice that this function also takes an *epsilon* input. This is because the ``test`` function reports the accuracy of a model 
# that is under attack from an adversary with strength :math:`\epsilon`. 
#
# More specifically, for each sample in the test set, the function computes the gradient of the loss w.r.t the input data (:math:`data\_grad`), 
# creates a perturbed image with ``fgsm_attack`` (:math:`perturbed\_data`), then checks to see if the perturbed example is adversarial. 
# In addition to testing the accuracy of the model, the function also saves and returns some successful adversarial examples to be visualized later.

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


######################################################################
# Run Attack
# ~~~~~~~~~~
# 
# The last part of the implementation is to actually run the attack. Here,we run a full test step for each epsilon value in the *epsilons* input.
# For each epsilon we also save the final accuracy and some successful dversarial examples to be plotted in the coming sections. Notice how the 
# printed accuracies decrease as the epsilon value increases. Also,note the :math:`\epsilon=0` case represents the original test accuracy,with no attack.
# 

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


######################################################################
# Results
# -------
# 
# Accuracy vs Epsilon
# ~~~~~~~~~~~~~~~~~~~
# 
# The first result is the accuracy versus epsilon plot. As alluded to
# earlier, as epsilon increases we expect the test accuracy to decrease.
# This is because larger epsilons mean we take a larger step in the
# direction that will maximize the loss. Notice the trend in the curve is
# not linear even though the epsilon values are linearly spaced. For
# example, the accuracy at :math:`\epsilon=0.05` is only about 4% lower
# than :math:`\epsilon=0`, but the accuracy at :math:`\epsilon=0.2` is 25%
# lower than :math:`\epsilon=0.15`. Also, notice the accuracy of the model
# hits random accuracy for a 10-class classifier between
# :math:`\epsilon=0.25` and :math:`\epsilon=0.3`.
# 

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
# plt.show()
plt.savefig('/home/hyr/codes/fgsm/output/lenet_Accuracy.png')
plt.close()

######################################################################
# Sample Adversarial Examples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Remember the idea of no free lunch? In this case, as epsilon increases
# the test accuracy decreases **BUT** the perturbations become more easily
# perceptible. In reality, there is a tradeoff between accuracy
# degredation and perceptibility that an attacker must consider. Here, we
# show some examples of successful adversarial examples at each epsilon
# value. Each row of the plot shows a different epsilon value. The first
# row is the :math:`\epsilon=0` examples which represent the original
# “clean” images with no perturbation. The title of each image shows the
# “original classification -> adversarial classification.” Notice, the
# perturbations start to become evident at :math:`\epsilon=0.15` and are
# quite evident at :math:`\epsilon=0.3`. However, in all cases humans are
# still capable of identifying the correct class despite the added noise.
# 

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
# plt.show()
plt.savefig('/home/hyr/codes/fgsm/output/lenet_adversarial.png')
plt.close()
