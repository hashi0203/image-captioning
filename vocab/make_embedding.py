import pickle
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

# from torch.autograd import detect_anomaly

class CBOW(nn.Module):

    def __init__(self, VOCAB_SIZE, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, embedding_dim)
        self.linear1 = nn.Linear(2 * context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, VOCAB_SIZE)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.linear1(embeds)
        out = F.relu(out)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def embedding():
    cdir = os.path.dirname(os.path.abspath(__file__))+'/'
    with open(cdir+'word_to_id.pkl', 'rb') as f:
        word_to_id = pickle.load(f)

    embeds = nn.Embedding(len(word_to_id), 10)
    print(embeds.weight)

    lookup_tensor = torch.tensor([word_to_id["above"]], dtype=torch.long)
    above_embed = embeds(lookup_tensor)
    print(above_embed)



def train(CONTEXT_SIZE, EMBEDDING_DIM, device, cdir):
    with open(cdir+'word_to_id.pkl', 'rb') as f:
        word_to_id = pickle.load(f)
    VOCAB_SIZE = len(word_to_id)
    with open(cdir+'filename_token.pkl', 'rb') as f:
        filename_token = pickle.load(f)

    # arrange tokens to Bag of Words
    print("Processing data.")
    data = []
    for j,im in enumerate(filename_token):
        tokens = [jj-1 for jj in im[1]]
        for i in range(CONTEXT_SIZE, len(tokens)-CONTEXT_SIZE):
            context = [tokens[i-CONTEXT_SIZE:i] + tokens[i+1:i+CONTEXT_SIZE+1]]
            target = tokens[i]
            data.append((context,target))

    # define loss function and optimization algorithm
    model = CBOW(VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_SIZE).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # start training
    print("Start Training.")
    losses = []

    for epoch in range(10):
        total_loss = torch.Tensor([0])
        with tqdm(data) as pbar:
            pbar.set_description("[Epoch %d]" % (epoch + 1))
            for context, target in pbar:
                # with detect_anomaly():
                context_idxs = torch.tensor(context, dtype=torch.long).to(device)
                target_idxs = torch.tensor([target], dtype=torch.long).to(device)
                model.zero_grad
                log_probs = model(context_idxs)
                loss = loss_function(log_probs, target_idxs)
                # if torch.isnan(loss):
                #     print(log_probs)
                #     print("loss")
                loss.backward()
                # if torch.isnan(loss):
                #     print("loss2")
                optimizer.step()

                total_loss += loss.item()
            losses.append(total_loss)

            print("[Epoch %d]: total_loss %4f" % ((epoch + 1), total_loss[0]))
            model_path = cdir+'model/model-'+str(CONTEXT_SIZE)+'-'+str(EMBEDDING_DIM)+'-'+str(epoch+1)+'.pth'
            torch.save(model.to('cpu').state_dict(), model_path)
            model.to(device)
    [print(loss) for loss in losses]

def infer(CONTEXT_SIZE, EMBEDDING_DIM, device, cdir):
    with open(cdir+'word_to_id.pkl', 'rb') as f:
        word_to_id = pickle.load(f)
    VOCAB_SIZE = len(word_to_id)
    with open(cdir+'id_to_word.pkl', 'rb') as f:
        id_to_word = pickle.load(f)
    with open(cdir+'filename_token.pkl', 'rb') as f:
        filename_token = pickle.load(f)

    model = CBOW(VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_SIZE).to(device)
    model.load_state_dict(torch.load(cdir+'model/model-'+CONTEXT_SIZE+'-'+EMBEDDING_DIM+'-10.pth'))

    data = []
    for i,im in enumerate(filename_token):
        if i > 40:
            break
        tokens = im[1]
        for i in range(CONTEXT_SIZE, len(tokens)-CONTEXT_SIZE):
            context = [tokens[i-CONTEXT_SIZE:i] + tokens[i+1:i+CONTEXT_SIZE+1]]
            target = tokens[i]
            data.append((context,target))

    for context, target in data:
        context_idxs = torch.tensor(context, dtype=torch.long).to(device)
        log_probs = model(context_idxs)
        _, index = log_probs[0].max(0)
        print([id_to_word[c] for c in context[0]])
        print(id_to_word[target])
        print(id_to_word[index.item()+1])

if __name__ == "__main__":
    CONTEXT_SIZE = 3
    EMBEDDING_DIM = 100
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running in "+dev+".")
    device = torch.device(dev)
    cdir = os.path.dirname(os.path.abspath(__file__))+'/'

    train(CONTEXT_SIZE, EMBEDDING_DIM, device, cdir)

    infer(CONTEXT_SIZE, EMBEDDING_DIM, device, cdir)
