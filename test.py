"""
Show how to connect to keypress events
"""
from build_vocab import get_image_caption
import sys
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_loader, FlickerDataset
from torchvision import transforms
import pickle
from vocabulary import Vocabulary
import torchvision
from PIL import Image
import os
from model import lstm_combine
import torch
from torch.autograd import Variable 

id = 0
dataset =[]
ax = 0
image_names= 0
fig=0
transform =0
text=0
def press(event):
    global id
    global dataset
    print('press', event.key)
    if event.key == 'z':
        id -= 1
        show_im(id)
    if event.key == 'x':
        id += 1
        show_im(id)

def show_im(id):
    global ax
    global image_names
    global fig, transform
    # im, caption, im_name = dataset[0]
    # im= im.numpy().reshape(224,224,3)
    # print(im)
    # plt.clf()
    im=Image.open(os.path.join(image_dir, image_names[id]))
    ax.imshow(np.asarray(im))
    im = transform(im)
    im = Variable(im.cuda())
    im = im.unsqueeze(0)
    feature = model.encoder(im)
    sampled_ids = model.decoder.sample(feature)
    caption = vocab.convert_sentence(sampled_ids)
    text.set_text(caption)
    print(image_names[id], caption)
    fig.canvas.draw()

# a = Data()
transform = transforms.Compose([ 
    transforms.RandomCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
        (0.229, 0.224, 0.225))])


    

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
image_name_path = './data/Flickr8k_text/Flickr_8k.testImages.txt'
image_dir = './data/Flickr8k_Dataset/'
caption_path = './data/Flickr8k_text/Flickr8k.token.txt'
image_names, caption = get_image_caption(image_name_path, caption_path)

model = lstm_combine(256, 512, len(vocab))
model.load_state_dict(torch.load('./models/base_linebk3/112400.pt'))
model.cuda()
model.eval()

# im.show()

# im, caption, im_name = dataset[0]
# __import__('pdb').set_trace()

# show_im(0,dataset)
# print(np.asarray((im)).shape)
fig, ax = plt.subplots()
text = ax.text(-1.5,-1.5, "")
# text = fig.text(0.5,0.5, '')
fig.canvas.mpl_connect('key_press_event', press)
# ax.imshow(np.asarray((im)))
plt.show()
