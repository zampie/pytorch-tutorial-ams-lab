import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import time

# -------------------------------------------------------------------------------------------------
# 定义重要参数

init_time = time.time()

root = './data/hy6/train'

max_epoch = 20

# image_size = 128
batch_size = 64
lr = 0.002

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# -------------------------------------------------------------------------------------------------
# 数据读取

transform = transforms.Compose([
    # transforms.Resize(image_size),
    # transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 图像的增强，变形，转型，归一化

dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)  # 创建文件夹型的数据集
# dataset = torchvision.datasets.CelebA('./data/', transform=transform, download=True)
# dataset = torchvision.datasets.CIFAR10('./data/', transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # 读取数据，设置读取方式 注意shuffle默认为False

classes = dataset.classes  # 获取数据集的标签名称
num_classes = len(classes)  # 获取数据集的类数


# -------------------------------------------------------------------------------------------------
# 显示一个batch
real_batch = next(iter(dataloader))

print(' '.join(classes[i] for i in real_batch[1]))

plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Training Images")
plt.imshow(
    np.transpose(vutils.make_grid(real_batch[0], nrow=8, padding=0, normalize=True).cpu(), (1, 2, 0)))
plt.show()

# -------------------------------------------------------------------------------------------------
# 定义网络

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, num_classes),
            # nn.Softmax(1)
        )

    def forward(self, input):
        x = self.feature(input)
        x = x.view(-1, 32 * 8 * 8)
        x = self.classifier(x)
        return x


net = Net().to(device)
print(net)

# 定义优化器和损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# -------------------------------------------------------------------------------------------------
# 训练
iteration = 0
loss_list = []
str_time = time.time()
print("Starting Training Loop...")

for epoch in range(max_epoch):
    for i, data in enumerate(dataloader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # optimizer.zero_grad()
        net.zero_grad()

        outputs = net(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if iteration % 1 == 0:
                predicted = outputs.argmax(1)
                correct = predicted == labels
                accuracy = correct.sum().item() / len(correct)

                it_time = time.time()
                print('[%d, %5d] loss: %.3f accuracy:%.3f%% time:%.3f' %
                      (epoch, i, loss.item(), accuracy * 100, it_time - str_time))

            loss_list.append(loss.item())
            iteration += 1

print('Finished Training')

# -------------------------------------------------------------------------------------------------
# 画Loss图
plt.figure(figsize=(5, 5))
plt.title('Training Loss')
plt.plot(loss_list)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('Loss.jpg')
plt.show()


# -------------------------------------------------------------------------------------------------
# 测试模型

print("Starting Test...")

root_test = root = './data/hy6/test'

test_set = torchvision.datasets.ImageFolder(root=root_test, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)

        predicted = torch.max(outputs, 1)[1]

        correct += (predicted == labels).sum().item()
        total += len(labels)

    accuracy = correct / total
    print('Accuracy of all test images: %.2f %%' % (accuracy * 100))

    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(num_classes):
        print('Accuracy of %8s : %.2f %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

# -------------------------------------------------------------------------------------------------
# 抽查
print("Samples:")

test_num = 8
data_iter = iter(test_loader)
images, labels = data_iter.next()
images, labels = images.to(device), labels.to(device)

# print(labels[:test_num])
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(test_num)))

outputs = net(images)

predicted = torch.max(outputs, 1)[1]
# print(predicted[:test_num])
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(test_num)))

correct = predicted == labels

accuracy = correct.sum().item() / len(correct)
print('Accuracy of these samples: %.2f%%' % (accuracy * 100))

plt.figure(figsize=(10, 10))
plt.imshow(
    np.transpose(torchvision.utils.make_grid(images[:test_num], nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()
