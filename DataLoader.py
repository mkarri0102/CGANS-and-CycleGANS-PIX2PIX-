from PIL import Image
import torch.utils.data as data
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy import misc

class load_data(data.Dataset):
    def __init__(self, experiment_name ,image_dir, label_dir, subfolder='train'):
        super(load_data, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)
        self.label_path = os.path.join(label_dir, subfolder)
        self.experiment_name=experiment_name
        if(experiment_name=="cityscapes"):
            self.img = [os.listdir(os.path.join(self.input_path,x)) for x in os.listdir(self.input_path) if not x.startswith('.')]
            self.label = [os.listdir(os.path.join(self.label_path,x)) for x in os.listdir(self.label_path) if not x.startswith('.')]
            self.image_filenames = [x.split('_')[0]+'/'+x for x in self.img[0]]
            self.label_filenames = [x.split('_')[0]+'/'+"_".join(x.split('_')[0:-1])+'_gtFine_color.png' for x in self.img[0]]
        elif(experiment_name=="shoes_edges" or experiment_name=="maps"):
            self.image_filenames=[x for x in os.listdir(self.input_path) if not x.startswith('.')]
            self.label_filenames=[x for x in os.listdir(self.label_path) if not x.startswith('.')]
        elif(experiment_name=="summer_winter"):
            self.input_path+="summer/"
            self.label_path+="winter/"
            self.image_filenames=[x for x in os.listdir(self.input_path) if not x.startswith('.')]
            self.label_filenames=[x for x in os.listdir(self.label_path) if not x.startswith('.')]
            
    def __getitem__(self, index):
        # Load Image
        #print(self.image_filenames)
            
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        label_fn = os.path.join(self.label_path, self.label_filenames[index])
        img = Image.open(img_fn).convert('RGB')
        label = Image.open(label_fn).convert('RGB')
        
        width,height=img.size
        if(self.experiment_name=="shoes_edges" or self.experiment_name=="maps"):
            img=img.crop((0,0,width/2,height))
            label=label.crop((width/2,0,width,height))
        
            
        resize = transforms.Resize([height, width])
        tx = transforms.ToTensor()
        
        img = resize(img)
        img = tx(img)
        
        label = resize(label)
        label = tx(label)
        #print(img_fn,label_fn)
        
        return img,label
    
    def __len__(self):
        return len(self.image_filenames)

   