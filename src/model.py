import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
from heapsort import heapsort
from tqdm import tqdm

class EncoderCNN(nn.Module):
    def __init__(self, EMBEDDING_DIM):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, EMBEDDING_DIM)
        self.bn = nn.BatchNorm1d(EMBEDDING_DIM, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, num_layers, max_seq_length=30):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, num_layers, batch_first=True)
        self.linear = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, BEAM_SIZE, states=None):
        """Generate captions for given image features using greedy search."""
        inputs = features.unsqueeze(1)

        with tqdm(range(self.max_seg_length)) as pbar:
            pbar.set_description("[Infering]")
            for j in pbar:
                if j == 0:
                    # prepare the first beam
                    hiddens, states = self.lstm(inputs, states)                             # hiddens: (1, 1, HIDDEN_DIM)
                    outputs = self.linear(hiddens.squeeze(1))                               # outputs:  (1, VOCAB_SIZE)
                    VOCAB_SIZE = len(outputs[0])
                    vocab_idx = list(range(VOCAB_SIZE))
                    ret = heapsort(list(zip(vocab_idx,list(outputs[0]))),BEAM_SIZE)   # ret: (BEAM_SIZE)
                    vocab_idx, prob = zip(*ret)
                    sampled_ids = [(torch.Tensor([v]).long(), prob) for v in vocab_idx]
                    beam = [(self.embed(s).unsqueeze(1), states) for s,_ in sampled_ids]        # beam: [(inputs, states)] * BEAM_SIZE
                else:
                    states_list = []
                    prob_list = []
                    for i,(inputs, states) in enumerate(beam):
                        # if the last word is end
                        if sampled_ids[i][0][-1] == VOCAB_SIZE-2:
                            states_list.append(states)
                            prob_list.extend(list(zip(zip([i],[VOCAB_SIZE-1]),[sampled_ids[i][1]])))
                        else:
                            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, HIDDEN_DIM)
                            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, VOCAB_SIZE)
                            states_list.append(states)
                    
                            idxs = zip([i] * VOCAB_SIZE, list(range(VOCAB_SIZE)))
                            prob_list.extend(list(zip(idxs,list(outputs[0]))))
                    ret = heapsort(prob_list, BEAM_SIZE)              # ret: [((beam_idx, vocab_idx), prob)] * (BEAM_SIZE)
                    predicted, prob = zip(*ret)
                    beam_idx, vocab_idx = zip(*predicted)

                    beam = []
                    tmp_sampled_ids = []
                    for i in range(BEAM_SIZE):
                        word_id = torch.Tensor([vocab_idx[i]]).long()
                        tmp_sampled_ids.append((torch.cat((sampled_ids[beam_idx[i]][0], word_id),0), prob[i]))
                        inputs = self.embed(word_id)                       # inputs: (batch_size, EMBEDDING_DIM)
                        inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, EMBEDDING_DIM)
                        beam.append((inputs, states_list[beam_idx[i]]))
                    sampled_ids = tmp_sampled_ids
        # sampled_ids = torch.stack(sampled_ids[0], 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    # def sample(self, features, BEAM_SIZE, states=None):
    #     """Generate captions for given image features using greedy search."""
    #     sampled_ids = []
    #     inputs = features.unsqueeze(1)
    #     for i in range(self.max_seg_length):
    #         hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, HIDDEN_DIM)
    #         outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, VOCAB_SIZE)
    #         idxs = [i] * len(outputs[0])
    #         enu = list(range(len(idxs)))
    #         pred = zip(idxs,enu)
    #         predicted = heapsort(list(zip(pred,list(outputs[0]))),1) 
    #         print(predicted)                       # predicted: (batch_size)
    #         _, predicted = zip(*predicted)
    #         predicted = torch.Tensor([predicted[0]]).long()
    #         # predicted = predicted.view(-1)[:1]
    #         # _, predicted = outputs.max(1)                        # predicted: (batch_size)
    #         sampled_ids.append(predicted)
    #         inputs = self.embed(predicted)                       # inputs: (batch_size, EMBEDDING_DIM)
    #         inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, EMBEDDING_DIM)
    #     sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
    #     return sampled_ids