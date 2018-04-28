import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data_utils
import torchvision.datasets as dset
from PIL import Image
import os
import cv2
import tqdm
import glob

def print_network(model, name):
    num_params=0
    for p in model.parameters():
        num_params+=p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params))


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=12,
#                                          shuffle=True, num_workers=2)


def read_images(path_images):
  im_list = os.listdir(path_images)
  return im_list

#label_list= os.listdir('./train_128')
#print(len(label_list))
    
#for j in range(len(label_list)):
#    print(label_list[j])
#    ims_list= read_images('./train_128/'+label_list[j])
#    print(len(ims_list))
#    if j==0:
#      ims = np.uint8(np.empty((128,128,3,len(ims_list)*len(label_list))))
#    print(ims.shape)
    
#    for i in range(len(ims_list)):
      #print('./train_128/'+label_list[j]+'/'+ims_list[i])
      #print(ims.shape[3])
#      ims[:,:,:,(i+(len(ims_list)*(j-1)))]=cv2.imread('./train_128/'+label_list[j]+'/'+ims_list[i],1)
#      ims[:,:,:,(i+(len(ims_list)*(j-1)))]=np.squeeze(ims[:,:,[2,1,0],(i+(len(ims_list)*(j-1)))])
      
#    print('Size in ims {}'.format(ims.shape))
    
#ims = np.moveaxis(ims,-1,0)
#ims = np.moveaxis(ims,-1,1)

#print('Size in ims {}'.format(ims.shape))

#ims = torch.from_numpy(ims.astype('float32'))
#lab=label_list[j]
#numoflabel=lab.split('_')
#print(numoflabel[1])
#numlabel=int(numoflabel[1])
#print(numlabel)
#train_label = torch.from_numpy(np.squeeze(np.ones(len(ims_list))*numlabel).astype('float32'))
#data_train = torch.utils.data.TensorDataset(ims,train_label)
data_train = torchvision.datasets.ImageFolder(root="train_128", transform=transform)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=50,
                                        shuffle=True)


#ims_list= read_images('./test_128/label_0')
#print(len(ims_list))

#ims = np.uint8(np.empty((128,128,3,len(ims_list))))
#print(ims.shape)

#for i in range(len(ims_list)):
#  ims[:,:,:,i]=cv2.imread('./test_128/label_0/'+ims_list[i],1)
  
#  ims[:,:,:,i]=np.squeeze(ims[:,:,[2,1,0],i])
  
#print('Size in ims {}'.format(ims.shape))

#ims = np.moveaxis(ims,-1,0)
#ims = np.moveaxis(ims,-1,1)

#print('Size in ims {}'.format(ims.shape))

#ims = torch.from_numpy(ims.astype('float32'))
#train_label = torch.from_numpy(np.squeeze(np.ones(len(ims_list))).astype('float32'))

#data_test = torch.utils.data.TensorDataset(ims,train_label)

#testset = torchvision.datasets(root='./test', train=False, transform=transform)
#testset=dset.  (root='./test',transform=transform)
testset = torchvision.datasets.ImageFolder(root="./val_128", transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                        shuffle=False)


classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9',
           '10', '11', '12', '13', '14', '15',
           '16', '17', '18', '19', '20', '21',
           '22', '23', '24', '25')

def imshow(img):
    img = img / 2 + 0.5     #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#print(labels)

# show image
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
__all__ = ['AlexNet', 'alexnet']


#model_urls = {
#   'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
#}


class AlexNet(nn.Module):

    def __init__(self, num_classes=25):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32,96, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 3 * 3, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        x = x.view(x.size(0), 128 * 3 * 3)
        x = self.classifier(x)
        return x
        
    def training_params(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
        self.Loss = nn.CrossEntropyLoss()
        
net = AlexNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train(data_loader, model, epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TRAIN] Epoch: {}".format(epoch)):
        data = data.cuda(); data = Variable(data)
        target = target.cuda(); target = Variable(target)

        output = model(data)
        model.optimizer.zero_grad()
        loss = model.Loss(output,target)   
        loss.backward()
        model.optimizer.step()
        loss_cum.append(loss.data.cpu()[0])
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
    
    print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))

def test(data_loader, model, epoch):
    model.eval()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TEST] Epoch: {}".format(epoch)):
        data = data.cuda(); data = Variable(data, volatile=True)
        target = target.cuda(); target = Variable(target, volatile=True)
        # print(data)
        
        output = model(data)
        loss = model.Loss(output,target)   
        loss_cum.append(loss.data.cpu()[0])
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
        # print(arg_max_out)
    
    print("Loss Test: %0.3f | Acc Test: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))
    
    
def test2():
    print("Begin Test.")    
    model.eval()
    strrta='';
    files = glob.glob('test_128/label_0/*.jpg')
    for file in files:
        img = Image.open(file)
        img = transform(img)
        if img.size(0) == 1:
            img = torch.stack([img] * 3, dim=1).squeeze()
        img=Variable(img, volatile=True).unsqueeze(0).cuda()
        
        output = model(img)
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        splitted=file.split('/')
        name=splitted[2]
        splitted=name.split('.')
#        print(splitted[1])
        strrta+=splitted[0]+','+"%d"%(int(arg_max_out))+'\n'
        #print(splitted[0]+','+"%d"%(int(arg_max_out)))
    print("saving file...")    
    with open("Output.txt", "w") as text_file:
        text_file.write(strrta)
    print("File saved...")
        
if __name__=='__main__':
    epochs=50
    batch_size=1000
    TEST=True
    
    model = net
    model.cuda()

    model.training_params()
    print_network(model, 'Conv network')    

    #Exploring model
    #data, _ = next(iter(train_loader))
    #_ = model(Variable(data.cuda(), volatile=True), verbose=True)

    for epoch in range(epochs):
        train(train_loader, model, epoch)
        if TEST: test(test_loader, model, epoch);
        if epoch>30:
            text = raw_input("Continue?:")
            if text=='0': break
    if TEST: test2()
        
def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model