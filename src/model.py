import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
from .heapsort import heapsort
from tqdm import tqdm

class EncoderCNN(nn.Module):
    def __init__(self, EMBEDDING_DIM):
        # Load the pretrained ResNet-152 and replace top fc layer.
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        # Delete the last fc layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, EMBEDDING_DIM)
        self.bn = nn.BatchNorm1d(EMBEDDING_DIM, momentum=0.01)
        
    def forward(self, images):
        # Extract feature vectors from input images.
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, num_layers, MAX_SEG_LENGTH=20):
        # Set the hyper-parameters and build the layers.
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, num_layers, batch_first=True)
        self.linear = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.MAX_SEG_LENGTH = MAX_SEG_LENGTH
        self.VOCAB_SIZE = VOCAB_SIZE
        
    def forward(self, features, captions, lengths):
        # Decode image feature vectors and generates captions.
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def beam_search(self, features, BEAM_SIZE, END_ID, device, states=None):
        # Generate captions for given image features using beam search.
        inputs = features.unsqueeze(1)
        VOCAB_SIZE = self.VOCAB_SIZE

        with tqdm(range(self.MAX_SEG_LENGTH)) as pbar:
            pbar.set_description("[Infering]")
            for j in pbar:
                if j == 0:
                    # Prepare the first beam
                    # We expect the first token is <start>, so we choose only the one with the highest probability (it should be <start>)
                    hiddens, states = self.lstm(inputs, states)                             # hiddens: (1, 1, HIDDEN_DIM)
                    outputs = self.linear(hiddens.squeeze(1))                               # outputs: (1, VOCAB_SIZE)
                    outputs = self.logsoftmax(outputs)

                    prob, predicted = outputs.max(1)                                        # predicted: (1)
                    sampled_ids = [(predicted, prob)]
                    beam = [(self.embed(s).unsqueeze(1), states) for s,_ in sampled_ids]    # beam: [(inputs, states)]

                else:
                    states_list = []
                    prob_list = []
                    for i,(inputs, states) in enumerate(beam):
                        # If the last word is end, skip infering
                        if sampled_ids[i][0][-1] == END_ID:
                            states_list.append(states)
                            prob_list.extend(list(zip(zip([i],[END_ID]),[sampled_ids[i][1]])))
                        else:
                            hiddens, states = self.lstm(inputs, states)                     # hiddens: (1, 1, HIDDEN_DIM)
                            outputs = self.linear(hiddens.squeeze(1))                       # outputs: (1, VOCAB_SIZE)
                            outputs = self.logsoftmax(outputs) + sampled_ids[i][1]
                            states_list.append(states)
                    
                            idxs = zip([i] * VOCAB_SIZE, list(range(VOCAB_SIZE)))           # idx: [(beam_idx, vocab_idx)] * (VOCAB_SIZE)
                            prob_list.extend(list(zip(idxs,list(outputs[0]))))              # prob_list: [((beam_idx, vocab_idx), prob)] * (all inferred results of this layer)
                    ret = heapsort(prob_list, BEAM_SIZE)                                    # ret: [((beam_idx, vocab_idx), prob)] * (BEAM_SIZE)
                    predicted, prob = zip(*ret)
                    beam_idx, vocab_idx = zip(*predicted)

                    beam = []
                    tmp_sampled_ids = []
                    for i in range(BEAM_SIZE):
                        word_id = torch.Tensor([vocab_idx[i]]).to(device).long()
                        tmp_sampled_ids.append((torch.cat((sampled_ids[beam_idx[i]][0], word_id),0), prob[i]))
                        inputs = self.embed(word_id)                                        # inputs: (1, EMBEDDING_DIM)
                        inputs = inputs.unsqueeze(1)                                        # inputs: (1, 1, EMBEDDING_DIM)
                        beam.append((inputs, states_list[beam_idx[i]]))                     # beam: [(inputs, states)] * (BEAM_SIZE)
                    sampled_ids = tmp_sampled_ids                                           # sampled_ids: [(predicted ids, prob)] * (BEAM_SIZE)
        return sampled_ids