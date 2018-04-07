import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
def base_combine(embed_size, hidden_size, vocab_size):
    return CombineModel(
            EncoderCNN(embed_size), 
            DecoderRNN(embed_size, hidden_size, vocab_size)
        )

def lstm_combine(embed_size, hidden_size, vocab_size, finetuned = True):
    return CombineModel(
            EncoderCNN(embed_size, finetuned), 
            DecoderLSTM(embed_size, hidden_size, vocab_size)
        )

class CombineModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(CombineModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, lengths):
        #images: [B,3,W,H]
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs
        

class EncoderCNN(nn.Module):
    def __init__(self, output_size, finetuned):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.alexnet = models.alexnet(pretrained = finetuned)

        linear = nn.Linear(4096, output_size)
        linear.weight.data.normal_(0.0, 0.02)
        linear.bias.data.fill_(0)

        classifier = [i for i in self.alexnet.classifier][:-1]
        classifier.append(linear)
        self.alexnet.classifier = nn.Sequential(*classifier)

        if not finetuned:
            self.init_weights()
        
    def init_weights(self):
        # self.linear.weight.data.normal_(0.0, 0.02)
        # self.linear.bias.data.fill_(0)
        for param in self.parameters():
            param.data.uniform_(-0.08, 0.08)
        pass
        
    def forward(self, images):
        """Extract the image feature vectors."""
        encode = self.alexnet(images)
        return encode
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers = 1):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers = num_layers, batch_first = True);
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.rnn(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.rnn(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.data[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        return sampled_ids


class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers = 1):
        """Set the hyper-parameters and build the layers."""
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers = num_layers, batch_first = True);
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.rnn(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        # H, C = None, None
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.rnn(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.data[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        return sampled_ids
