with open('./dataset/poetry.txt', 'r',encoding='UTF-8') as f:
    poetry_corpus = f.read()

print(poetry_corpus[:100])
print(len(poetry_corpus))

poetry_corpus = poetry_corpus.replace('\n', ' ').replace('\r', ' ').replace('，', ' ').replace('。', ' ')
poetry_corpus[:100]

import numpy as np

class TextConverter(object):
    def __init__(self, text_path, max_vocab=5000):
        """建立一个字符索引转换器
        
        Args:
            text_path: 文本位置
            max_vocab: 最大的单词数量
        """
        
        with open(text_path, 'r',encoding='UTF-8') as f:
            text = f.read()
        text = text.replace('\n', ' ').replace('\r', ' ').replace('，', ' ').replace('。', ' ')
        # 去掉重复的字符
        vocab = set(text)

        # 如果单词总数超过最大数值，去掉频率最低的
        vocab_count = {}
        
        # 计算单词出现频率并排序
        for word in vocab:
            vocab_count[word] = 0
        for word in text:
            vocab_count[word] += 1
        vocab_count_list = []
        for word in vocab_count:
            vocab_count_list.append((word, vocab_count[word]))
        vocab_count_list.sort(key=lambda x: x[1], reverse=True)
        
        # 如果超过最大值，截取频率最低的字符
        if len(vocab_count_list) > max_vocab:
            vocab_count_list = vocab_count_list[:max_vocab]
        vocab = [x[0] for x in vocab_count_list]
        self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)


convert = TextConverter('./dataset/poetry.txt', max_vocab=10000)

# 原始文本字符
txt_char = poetry_corpus[:11]
print(txt_char)

# 转化成数字
print(convert.text_to_arr(txt_char))

n_step = 20

# 总的序列个数
num_seq = int(len(poetry_corpus) / n_step)

# 去掉最后不足一个序列长度的部分
text = poetry_corpus[:num_seq*n_step]

print(num_seq)

import torch

arr = convert.text_to_arr(text)
arr = arr.reshape((num_seq, -1))
arr = torch.from_numpy(arr)

print(arr.shape)
print(arr[0, :])

class TextDataset(object):
    def __init__(self, arr):
        self.arr = arr
        
    def __getitem__(self, item):
        x = self.arr[item, :]
        
        # 构造 label
        y = torch.zeros(x.shape)
        # 将输入的第一个字符作为最后一个输入的 label
        y[:-1], y[-1] = x[1:], x[0]
        return x, y
    
    def __len__(self):
        return self.arr.shape[0]


train_set = TextDataset(arr)

x, y = train_set[0]
print(convert.arr_to_text(x.numpy()))
print(convert.arr_to_text(y.numpy()))

from torch import nn
from torch.autograd import Variable

use_gpu = True

class CharRNN(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_size, 
                 num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.word_to_vec = nn.Embedding(num_classes, embed_dim)
        #self.rnn = nn.GRU(embed_dim, hidden_size, num_layers, dropout)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers)
        self.project = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hs=None):
        batch = x.shape[0]
        if hs is None:
            hs = Variable(
                torch.zeros(self.num_layers, batch, self.hidden_size))
            if use_gpu:
                hs = hs.cuda()
        word_embed = self.word_to_vec(x)  # (batch, len, embed)
        word_embed = word_embed.permute(1, 0, 2)  # (len, batch, embed)
        out, h0 = self.rnn(word_embed, hs)  # (len, batch, hidden)
        le, mb, hd = out.shape
        out = out.view(le * mb, hd)
        out = self.project(out)
        out = out.view(le, mb, -1)
        out = out.permute(1, 0, 2).contiguous()  # (batch, len, hidden)
        return out.view(-1, out.shape[2]), h0

from torch.utils.data import DataLoader

batch_size = 128
#train_data = DataLoader(train_set, batch_size, True, num_workers=4)
train_data = DataLoader(train_set, batch_size, True)

print("train data is: ",train_data)

from mxtorch.trainer import ScheduledOptim

#model = CharRNN(convert.vocab_size, 512, 512, 2, 0.5)

model = CharRNN(convert.vocab_size, 512, 512, 2, 0.5)

if use_gpu:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()

basic_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = ScheduledOptim(basic_optimizer)

epochs = 20
for e in range(epochs):
    train_loss = 0
    for data in train_data:
        x, y = data
        y = y.long()
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        x, y = Variable(x), Variable(y)

        # Forward.
        score, _ = model(x)
        loss = criterion(score, y.view(-1))

        #print(loss)

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        # Clip gradient.
        nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()

        #train_loss += loss.data[0]
        train_loss += loss.item()

    print('epoch: {}, perplexity is: {:.3f}, lr:{:.1e}'.format(e+1, np.exp(train_loss / len(train_data)), optimizer.lr))

def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c

begin = '天青色等烟雨'
text_len = 30

model = model.eval()
samples = [convert.word_to_int(c) for c in begin]
input_txt = torch.LongTensor(samples)[None]
if use_gpu:
    input_txt = input_txt.cuda()
input_txt = Variable(input_txt)
_, init_state = model(input_txt)
result = samples
model_input = input_txt[:, -1][:, None]
for i in range(text_len):
    out, init_state = model(model_input, init_state)
    pred = pick_top_n(out.data)
    model_input = Variable(torch.LongTensor(pred))[None]
    if use_gpu:
        model_input = model_input.cuda()
    result.append(pred[0])
text = convert.arr_to_text(result)
print('Generated text is: {}'.format(text))



