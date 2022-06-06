from PIL import Image
import torch.utils.data as data
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class load_data(data.Dataset):
    def __init__(self, image_dir, label_dir, subfolder='train'):
        super(load_data, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)
        self.label_path = os.path.join(label_dir, subfolder)
        self.img = [os.listdir(os.path.join(self.input_path,x)) for x in os.listdir(self.input_path) if not x.startswith('.')]
        self.label = [os.listdir(os.path.join(self.label_path,x)) for x in os.listdir(self.label_path) if not x.startswith('.')]
        self.image_filenames = [x.split('_')[0]+'/'+x for x in self.img[0]]
        self.label_filenames = [x.split('_')[0]+'/'+"_".join(x.split('_')[0:-1])+'_gtFine_color.png' for x in self.img[0]]
             

    def __getitem__(self, index):
        # Load Image
        #print(self.image_filenames)
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        label_fn = os.path.join(self.label_path, self.label_filenames[index])
        img = Image.open(img_fn).convert('RGB')
        resize = transforms.Resize([1024, 2048])
        img = resize(img)
        tx = transforms.ToTensor()
        img = tx(img)
        label = Image.open(label_fn).convert('RGB')
        label = resize(label)
        label = tx(label)
        #print(img_fn,label_fn)
        
        return img,label
    
    def __len__(self):
        return len(self.image_filenames)
