import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path






class AbstractEncoder(BasicModule):
    def __init__(self, drop_out, emb_dim, lstm_hdim, embeddings):
        super(AbstractEncoder, self).__init__()
        self.embeddings = self.load_embeddings(embeddings)
        self.word_lstm = nn.LSTM(emb_dim, hidden_size=lstm_hdim, num_layers=1, batch_first=True, bidirectional=True)
        self.embedding_dropout = nn.Dropout(p=drop_out)
        self.lstm_hid_dim = lstm_hdim

    def init_hidden(self, sent_len):
        return (torch.randn(2, sent_len, self.lstm_hid_dim).cuda(),
                torch.randn(2, sent_len, self.lstm_hid_dim).cuda())

    def load_embeddings(self, embeddings):
        """Load the embeddings based on flag"""
        embeddings_ = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        embeddings_.weight = torch.nn.Parameter(embeddings)
        return embeddings_

    def forward(self, x, abstract_encoder=False):
        #print(x.shape)
        sent_len = len(x)
        embeddings = self.embeddings(x)
        embeddings = self.embedding_dropout(embeddings)
        #print(embeddings.shape)
        #embeddings = embeddings.expand(1, embeddings.size(0), embeddings.size(1))
        # step1 get LSTM outputs
        hidden_state = self.init_hidden(sent_len)
        outputs, hidden_state = self.word_lstm(embeddings, hidden_state)
        return outputs 
        
        
        
class word_span_encoder(BasicModule):
    def __init__(self, dropout, word_rnn_size ,word_att_size,span_class):
        super(word_span_encoder,self).__init__()
        
        self.q = nn.Linear(2*word_rnn_size ,200)
        self.k = nn.Linear(2*word_rnn_size ,200)
        self.v = nn.Linear(2*word_rnn_size ,200)
        
        self.word_att = nn.Linear(2*word_rnn_size ,200)
        self.word_att = nn.Linear(200 ,1)
        
        self.conv_att = nn.Conv1d(256,200,4, stride =4)
        
        self.tanh = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(dropout)
        
        self.span_out = nn.Linear(200,span_class)
        
        
    def forward(self,x):
        
        q = self.tanh(self.q(x))
        k = self.tanh(self.k(x).permute(0,2,1))
        v = self.tanh(self.v(x))
            
        
        soft = self.softmax(torch.bmm(q,k))
        
        self_att = torch.bmm(soft,v)
        
        #avg_embed = torch.mean(self_att,1)
        
        #print(self_att.shape)
        
        #span_out = self.sigmoid(self.span_out(avg_embed))
        
        #print(span_out.shape)
        
        batch = x.shape[0]
        
        self_att = self_att.reshape(batch,-1,4,16)
        
        word_att = x.reshape(batch,-1,64)
        
        self_att  = torch.mean(self_att,2)
        
        
        self_att = self_att.reshape(batch,16,-1)
        
        conv_att = self.conv_att(word_att)
        
        conv_att = conv_att.reshape(batch,16,-1)
        
        sent_level = conv_att + self_att
        #sent_level = conv_att 
        #sent_level = self_att
        
        
        #return sent_level,span_out
        
        return sent_level
        
        
        
        
        
class sent_encoder(BasicModule):
    def __init__(self, dropout, word_rnn_size , sent_rnn_size, sent_att_size):
        super(sent_encoder,self).__init__()
        
        self.sent_lstm = nn.LSTM(200, hidden_size=100, num_layers=1, batch_first=True, bidirectional=True)
        
        self.q = nn.Linear(200,128)
        self.k = nn.Linear(200,128)
        self.v = nn.Linear(200,128)
        
        self.sent_att = nn.Linear(2*sent_rnn_size, sent_att_size)
        
        self.conv_att = nn.Conv1d(200,128,16, stride =16)
        
        self.tanh = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(dropout)
        
        
        sent_att_size = 128
        #emotion
        
        self.emoti_fc = nn.Linear(128,64)
        self.emoti_out = nn.Linear(64, 7 )
        
        #sentiment 
        
        self.senti_fc = nn.Linear(128, 64)
        self.senti_out = nn.Linear(64, 3)
        
        #sarcasm
        
        self.sar_fc = nn.Linear(128, 64)
        self.sar_out = nn.Linear(64,2)
        
        #bully
        
        self.bully_fc = nn.Linear(128, 64)
        self.bully_out = nn.Linear(64+3,2)
        
        #span
        self.span_out = nn.Linear(128,64)
        
        
        #attention
        
          
        self.alphas_b = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(1)])
        self.alphas_s = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(1)])
        self.alphas_e = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(1)])
        self.alphas_sa = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(1)])
        
        
        
        
        
    def forward(self,x,span=None):
    
        #print(x.shape)
    
        batch = x.shape[0]
         
        sent_lstm, _= self.sent_lstm(x)
        
        
    
        q = self.tanh(self.q(sent_lstm))
        k = self.tanh(self.k(sent_lstm).permute(0,2,1))
        v = self.tanh(self.v(sent_lstm))
            
        
        soft = self.softmax(torch.bmm(q,k))
        
        self_att = torch.bmm(soft,v)
        
        #print(self_att.shape)
        
        self_att = self_att.reshape(batch,-1,16,1)
        
        self_att = torch.mean(self_att,2)
        
        self_att = self_att.reshape(batch,-1)
        
        
        x = x.reshape(batch,-1,16)
        
        #doc_level = torch.mean(self_att,1)
        conv_att = self.conv_att(x)
        
        
        conv_att = conv_att.reshape(batch,-1)
        
        doc_level = self_att + conv_att
        #doc_level = conv_att
        #doc_level = self_att
        
        #doc_level = self.dropout(doc_level)
        
            
        #emoti_fc = self.tanh(self.emoti_fc(self.dropout(doc_level)))
        senti_fc = self.tanh(self.senti_fc(self.dropout(doc_level)))
        #sar_fc = self.tanh(self.sar_fc(self.dropout(doc_level)))
        bully_fc = self.tanh(self.bully_fc(self.dropout(doc_level)))
        
        #if span is None:
        #    span = self.sigmoid(self.span_out(self.dropout(doc_level)))
        
        #emoti_out = self.softmax(self.emoti_out(emoti_fc))
        senti_out = self.softmax(self.senti_out(senti_fc))
        #sar_out = self.softmax(self.sar_out(sar_fc))
        
                
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
       
        
        #span = torch.cat((span,torch.zeros((batch,36)).to(device)),1)
        
        #xsum = emoti_fc+senti_fc+sar_fc+bully_fc+span
        #xsum = bully_fc+span
        
        #xsum = doc_level + span
       
        #xsum = torch.cat((senti_out, bully_fc,span),1)
        xsum = torch.cat((senti_out, bully_fc),1)
        #xsum = torch.cat(( bully_fc,span),1)
        
        #xsum = torch.cat((doc_level,span),1)
        
        #xsum = senti_fc*self.alphas_s[0].expand_as(senti_fc) + bully_fc*self.alphas_b[0].expand_as(bully_fc) + sar_fc*self.alphas_sa[0].expand_as(sar_fc)+emoti_fc*self.alphas_e[0].expand_as(emoti_fc)
        
        bully_out = self.softmax(self.bully_out(xsum))
        
        #bully_out = self.softmax(self.bully_out(doc_level))
        
        
        #return emoti_out,senti_out,sar_out,bully_out
        #return span,senti_out,bully_out
        return senti_out,bully_out
        #return emoti_out,span,senti_out,bully_out
        #return span,bully_out
        #return bully_out
        
        
        
        
        
        
        
        
        
        
        
    
        
    
        
        
        