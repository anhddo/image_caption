import argparse
import torch
import glob
import torch.nn as nn
import numpy as np
import os
import pickle
from vocabulary import Vocabulary
from data_loader import get_loader, FlickerDataset
from model import lstm_combine
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from train_log import TrainLog

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 1)

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def build_data_loader(path, vocab, transform, args):
    return get_loader(
            args.image_dir,
            path,
            args.caption_path, 
            vocab, 
            transform,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers) 

def build_test_dataset( vocab, transform, args):
    return FlickerDataset(image_dir=args.image_dir,
            image_name_path = args.test_image_name_path,
            caption_path=args.caption_path,
            vocab=vocab,
            transform = transform)

def validate(n_batch, criterion, data_loader, model):
    model.eval()
    total_loss = 0
    for i, (images, captions, image_names, lengths) in enumerate(data_loader):
        # Set mini-batch dataset
        images = to_var(images)
        captions = to_var(captions)#[B, L]
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
    model.zero_grad()
    outputs = model(images, captions, lengths)
    total_loss += criterion(outputs, targets)[0]

    return total_loss / n_batch


def test(n_images, flicker, model, vocab):
    result = []
    for i in range(n_images):
        im, cap, im_name = flicker[i]
        im = to_var(im)
        im = im.unsqueeze(0)
        feature = model.encoder(im)
        sampled_ids = model.decoder.sample(feature)
        result.append((im_name, vocab.convert_sentence(sampled_ids)))
    return result

def remove_old_model(setting_path, keep_last_model):
    file_names = sort_saved_model_names(setting_path)
    for i in range(len(file_names) - keep_last_model):
        os.remove(file_names[i])

def sort_saved_model_names(setting_path):
    file_names = glob.glob('%s/*.pt' % (setting_path))
    # file_names = setting_path
    names = [int(os.path.splitext(os.path.basename(n))[0]) for n in file_names]
    file_names = [file_names[i] for i in np.argsort(names)]
    return file_names


def main(args):
    setting_path = os.path.join(args.model_dir, args.setting_name) 
    train_log_path = os.path.join(setting_path,'train_log.pkl')

    # Create model directory
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(setting_path):
        os.makedirs(setting_path)

    train_log = TrainLog()
    train_log.load(train_log_path)


    
    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    test_dataset= build_test_dataset(vocab, transform, args)
    # Build data loader
    data_loader = build_data_loader(args.train_image_name_path, vocab, transform, args)
    eval_data_loader = build_data_loader(args.validate_image_name_path, vocab, transform, args)
    # # Build the models
    model = lstm_combine(args.embed_size, args.hidden_size, len(vocab),\
            str2bool(args.weight_finetuned))

    last_model_paths = sort_saved_model_names(setting_path)
    last_model_path = ''
    if last_model_paths:
        last_model_path = last_model_paths[-1]

    if os.path.exists(last_model_path):
        # __import__('pdb').set_trace()
        torch_object = torch.load(last_model_path)
        model.load_state_dict(torch_object)
    
    if torch.cuda.is_available():
        model.cuda()

    # # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    
    # Train the Models
    total_batch = len(data_loader)
    while train_log.epoch < args.num_epochs:
        for i, (images, captions, image_names, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            images = to_var(images)
            captions = to_var(captions)#[B, L]
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            model.zero_grad()
            outputs = model(images, captions, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            # weight_norm =  np.sqrt(np.sum([ np.square(torch.norm(p)[0]) for p in model.parameters()]))
            optimizer.step()
            batch_idx = train_log.epoch * total_batch + i
            # Print log info
            # if True:
            if batch_idx % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(train_log.epoch, args.num_epochs, i, total_batch, 
                        loss.data[0], np.exp(loss.data[0]))) 

                train_log.train_loss.append(loss.data[0])
                model.eval()
                # validate(10, criterion, eval_data_loader, model)
                translate_test = test(3, test_dataset, model, vocab)
                for sen in translate_test:
                    print(sen)
                model.train()
                train_log.save(train_log_path)
                
            # Save the models
            if batch_idx % args.save_step == 0:
                # path = os.path.join(args.setting_path, 'combine-%d-%d.pkl' %(epoch+1, i+1))
                torch.save(model.state_dict(), os.path.join(setting_path,
                     str(batch_idx) + '.pt'))
                remove_old_model(setting_path, args.keep_last_model)

        train_log.epoch += 1

        torch.save(model.state_dict(), os.path.join(setting_path,
            str(batch_idx) + '.pt'))
        remove_old_model(setting_path, args.keep_last_model)
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--setting_name', type=str, default='base_line')
    parser.add_argument('--image_dir', type=str, default='./data/resized256' ,
                        help='directory for resized images')
    parser.add_argument('--train_image_name_path', type=str,
            default='./data/Flickr8k_text/Flickr_8k.trainImages.txt')
    parser.add_argument('--validate_image_name_path', type=str,
            default='./data/Flickr8k_text/Flickr_8k.devImages.txt')
    parser.add_argument('--test_image_name_path', type=str,
            default='./data/Flickr8k_text/Flickr_8k.testImages.txt')
    parser.add_argument('--model_dir', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--caption_path', type=str,
                        default='./data/Flickr8k_text/Flickr8k.token.txt',
                        help='path for train annotation json file')

    parser.add_argument('--keep_last_model', type=int, default=3)
    parser.add_argument('--weight_finetuned',  default=True,\
            help = 'init CNN weight with pretrain on imagenet')


    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=2000,
                        help='step size for saving trained models')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=128 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=128 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
