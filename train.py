import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import  mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


import os
import pickle
from lhan import AbstractEncoder, word_span_encoder, sent_encoder
#from src.hierarchical_att_model import HierAttNet

# import EarlyStopping
import os, sys
#sys.path.append('path_to_the_module/early-stopping-pytorch')
from pytorchtools import EarlyStopping
#from torchsample.callbacks import EarlyStopping


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
print(device)



output=open("/home/prince/prince/COLING/cyberbully/dataset-bully-word.pkl","rb")
mbert = pickle.load(output)


#for i in range(mbert["task-label"].shape[0]):
#   print(mbert["task-label"][i])



#print(len(mbert["task-label"]))
print(mbert["senti-label"].shape)

doc=mbert["doc"][:6084]
sent_per_doc=mbert["sent_per_doc"][:6084]
word_per_sent=mbert["word_per_sent"][:6084]
emoti=mbert["emoti-label"][:6084]
senti=mbert["senti-label"][:6084]
task = mbert["task-label"][:6084]
sarcasm=mbert["sarcasm-label"][:6084]
span=mbert["span"][:6084]
target = mbert["target-label"][:6084]

cnt=0

for i in range(task.shape[0]):
    if(task[i]==1):
        cnt=cnt+1
        
print(cnt)
    

#task =keras.utils.to_categorical(task)
#print(task.shape)


#senti_label={"negative":0,"neutral":1,"positive":2,"Negative":0,"Neutral":1,"Positive":2}
#emoti_label={"Happiness":0,"Sadness":1,"Anger":2,"Fear":3,"Surprise":4,"Disgust":5,"Others":9,"happiness":0,"sadness":1,"anger":2,"fear":3,"surprise":4,"disgust":5,"others":9,"Ridicule":6,"Anticipation":7,"Trust":8}




doc_train,doc_test,task_train,task_test=train_test_split(doc,task,test_size=0.2,stratify=task,random_state=1234)
sent_per_doc_train,sent_per_doc_test,task_train,task_test=train_test_split(sent_per_doc,task,test_size=0.2,stratify=task,random_state=1234)
word_per_sent_train,word_per_sent_test,task_train,task_test=train_test_split(word_per_sent,task,test_size=0.2,stratify=task,random_state=1234)
emoti_train,emoti_test,task_train,task_test=train_test_split(emoti,task,test_size=0.2,stratify=task,random_state=1234)
senti_train,senti_test,task_train,task_test=train_test_split(senti,task,test_size=0.2,stratify=task,random_state=1234)
sarcasm_train,sarcasm_test,task_train,task_test=train_test_split(sarcasm,task,test_size=0.2,stratify=task,random_state=1234)
span_train,span_test,task_train,task_test=train_test_split(span,task,test_size=0.2,stratify=task,random_state=1234)
target_train,target_test,task_train,task_test=train_test_split(target,task,test_size=0.2,stratify=task,random_state=1234)


task=task_train


doc_train,doc_val,task_train,task_val=train_test_split(doc_train,task,test_size=0.1,stratify=task,random_state=1234)
sent_per_doc_train,sent_per_doc_val,task_train,task_val=train_test_split(sent_per_doc_train,task,test_size=0.1,stratify=task,random_state=1234)
word_per_sent_train,word_per_sent_val,task_train,task_val=train_test_split(word_per_sent_train,task,test_size=0.1,stratify=task,random_state=1234)
emoti_train,emoti_val,task_train,task_val=train_test_split(emoti_train,task,test_size=0.1,stratify=task,random_state=1234)
senti_train,senti_val,task_train,task_val=train_test_split(senti_train,task,test_size=0.1,stratify=task,random_state=1234)
sarcasm_train,sarcasm_val,task_train,task_val=train_test_split(sarcasm_train,task,test_size=0.1,stratify=task,random_state=1234)
span_train,span_val,task_train,task_val=train_test_split(span_train,task,test_size=0.1,stratify=task,random_state=1234)
target_train,target_val,task_train,task_val=train_test_split(target_train,task,test_size=0.1,stratify=task,random_state=1234)


train={"doc":doc_train,"sent_per_doc":sent_per_doc_train,"word_per_sent":word_per_sent_train,"label":task_train,"target":target_train,"emotion":emoti_train,"sarcasm":sarcasm_train,"sentiment":senti_train,"span":span_train}
val={"doc":doc_val,"sent_per_doc":sent_per_doc_val,"word_per_sent":word_per_sent_val,"label":task_val,"target":target_val,"emotion":emoti_val,"sarcasm":sarcasm_val,"sentiment":senti_val,"span":span_val}
test={"doc":doc_test,"sent_per_doc":sent_per_doc_test,"word_per_sent":word_per_sent_test,"label":task_test,"target":target_test,"emotion":emoti_test,"sarcasm":sarcasm_test,"sentiment":senti_test,"span":span_test}





class Tweet(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data=data
        
    def __len__(self):
        return len(self.data["doc"])
    def __getitem__(self,idx):
    
        if(torch.is_tensor(idx)):
            idx=idx.tolist()
            
            
            
        doc=torch.tensor(self.data["doc"][idx].astype(np.float)).long().to(device)
        
        sent_per_doc=torch.tensor(self.data["sent_per_doc"][idx].astype(np.float)).long().to(device)
        word_per_sent=torch.tensor(self.data["word_per_sent"][idx].astype(np.float)).long().to(device)
        label=torch.tensor(self.data["label"][idx].astype(np.long)).float().to(device)
        emotion=torch.tensor(self.data["emotion"][idx]).long().to(device)
        sarcasm=torch.tensor(self.data["sarcasm"][idx].astype(np.long)).long().to(device)
        sentiment=torch.tensor(self.data["sentiment"][idx]).long().to(device)
        span=torch.tensor(self.data["span"][idx]).float().to(device)
        target=torch.tensor(self.data["target"][idx]).long().to(device)
        
    
        
        sample = {
            
            "doc":doc,
            "sent_per_doc":sent_per_doc,
            "word_per_sent":word_per_sent,
            "label":label,
            "emotion":emotion,
            "sarcasm":sarcasm,
            "sentiment":sentiment,
            "span":span,
            "target":target
        }
        
        return sample
        
        
        
        
        
 
tweet_train = Tweet(train)
dataloader_train = DataLoader(tweet_train, batch_size=32,shuffle=False, num_workers=0)

print("train_data loaded")

tweet_val = Tweet(val)
dataloader_val = DataLoader(tweet_val, batch_size=32,shuffle=False, num_workers=0)
print("validation_data loaded")


tweet_test = Tweet(test)
dataloader_test = DataLoader(tweet_test, batch_size=32,shuffle=False, num_workers=0,drop_last=True) 



 
n_classes=3
vocab_size=20668

# tweet 6000, unique words ->mapped 

#embeddings=open("/home/prince/prince/COLING/cyberbully/embedding_matrix_6436.pkl","rb")
embeddings=open("/home/prince/prince/COLING/cyberbully/embed_matrix_bert.npy","rb")
#embeddings=pickle.load(embeddings)
embeddings=np.load(embeddings)

embeddings=torch.tensor(embeddings).float().to(device)

emb_size=768
fine_tune=True
word_rnn_size=128
sentence_rnn_size=128
word_rnn_layers=1
sentence_rnn_layers=1
word_att_size=256
sentence_att_size=256
dropout= 0.3
span_class = 64
lr = 0.0003
        
abs_encoder = AbstractEncoder(dropout, emb_size,word_rnn_size,embeddings).to(device)

span_model = word_span_encoder(dropout, word_rnn_size, word_att_size, span_class).to(device)

multitask = sent_encoder(dropout, word_rnn_size, sentence_rnn_size, sentence_att_size).to(device)



combined_params = list(abs_encoder.parameters()) + list(span_model.parameters()) + list(multitask.parameters())

optimizer = torch.optim.Adam(combined_params, lr=lr, weight_decay=0) 

criterion_bce = nn.BCELoss() #Binary case
criterion_loss = nn.CrossEntropyLoss()


epochs = 20


exp_path = "coling2022"

def train_model(abs_encoder,span_model,multitask, patience, n_epochs):

    epochs = n_epochs
#     clip = 5

    train_acc_list=[]
    val_acc_list=[]
    train_loss_list=[]
    val_loss_list=[]
    
 
    
        # initialize the experiment path
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    # initialize early_stopping object
    span_file = os.path.join(exp_path, 'checkpoint_span.pt')
    multitask_file = os.path.join(exp_path, 'checkpoint_mutlitask.pt')
    abs_file = os.path.join(exp_path, 'checkpoint_absencoder.pt')
    early_stopping_span = EarlyStopping(patience=patience, verbose=True, path=span_file)
    early_stopping_multitask = EarlyStopping(patience=patience, verbose=True, path=multitask_file)
    early_stopping_abs = EarlyStopping(patience=patience, verbose=True, path=abs_file)


    abs_encoder.train()
    span_model.train()
    multitask.train()

    for i in range(epochs):
        total_loss_train = 0
        total_train = 0
        correct_train = 0

        for data in dataloader_train:

            doc = data['doc'].to(device)
            sent_per_doc = data['sent_per_doc'].to(device)
            word_per_sent = data['word_per_sent'].to(device)
            
            #print(word_per_sent.shape)
            
            sent_per_doc=sent_per_doc.squeeze()
            

            label_train = data['label'].to(device)
            emotion_train = data['target'].to(device)
            sentiment_train = data['sentiment'].to(device)
            span_train = data['span'].to(device)
            sarcasm_train = data['sarcasm'].to(device)
            target_train = data['target'].to(device)
            
            
            #print(label_train.shape)
            
            abs_encoder.zero_grad()
            span_model.zero_grad()
            multitask.zero_grad()
            
            #print(emotion_train.shape)


            
            batch_size = doc.shape[0]
            
            doc = doc.reshape(batch_size,-1)
            
            abs_encoded= abs_encoder(doc)
            
            #sent_level, span_out = span_model(abs_encoded)
            sent_level = span_model(abs_encoded)
            
            #print(sent_level.shape)
            
            
            
            #emoti_out,senti_out,sar_out,output = multitask(sent_level,span_out)
            #senti_out,output = multitask(sent_level,span_out)
            #span_out,senti_out,output = multitask(sent_level,span_out)
            #span_out,senti_out,output = multitask(sent_level)
            senti_out,output = multitask(sent_level)
            #emoti_out,span_out,senti_out,output = multitask(sent_level)
            #span_out,output = multitask(sent_level)
            #output = multitask(sent_level,span_out)
            #output = multitask(sent_level)
            
            #span_loss = criterion_bce(span_out,span_train)
            #emoti_loss = criterion_loss(emoti_out, emotion_train.long().squeeze(1))
            senti_loss = criterion_loss(senti_out, sentiment_train.long().squeeze(1))
            bully_loss = criterion_loss(output, label_train.long().squeeze(1))
            #sar_loss = criterion_loss(sar_out, sarcasm_train.long().squeeze(1))
            
            
            
            #loss = span_loss +  bully_loss 
            #loss = bully_loss 
            #loss = span_loss + emoti_loss + senti_loss + bully_loss + sar_loss 
            #loss = span_loss + senti_loss + bully_loss 
            #loss = span_loss + senti_loss + bully_loss + emoti_loss
            loss = senti_loss + bully_loss 
            
            
            
            
            #print(loss)
            
            loss.backward()
            
            optimizer.step()
            
            with torch.no_grad():
                _, predicted_train = torch.max(output.data, 1)
                #predicted_train = (output.data>=0.5).float()
                total_train += label_train.size(0)
                #total_train += emotion_train.size(0)
                
        
                correct_train += (predicted_train == label_train.squeeze(1)).sum().item()
                #correct_train += (predicted_train == emotion_train).sum().item()
    #                 out_val = (output.squeeze()>0.5).float()
    #                 out_final = ((out_val == 1).nonzero(as_tuple=True)[0])
    #                 print()
    #                 acc = torch.abs(output.squeeze() - label.float()).view(-1)
    #                 acc = (1. - acc.sum() / acc.size()[0])ret
    #                 total_acc_train += acc
                total_loss_train += loss.item()

        
        train_acc = 100 * correct_train / total_train
        train_loss = total_loss_train/total_train
        span_model.eval()
        abs_encoder.eval()
        multitask.eval()
#         total_acc_val = 0
        total_loss_val = 0
        total_val = 0
        correct_val = 0

        with torch.no_grad():
            for data in dataloader_val:                
#                 Clip features...                

                
                doc = data['doc'].to(device)
                sent_per_doc = data['sent_per_doc'].to(device)
                word_per_sent = data['word_per_sent'].to(device)

                sent_per_doc=sent_per_doc.squeeze()
                
                label_val = data['label'].to(device)
                emotion_val = data['target'].to(device)
                span_val = data['span'].to(device)
                sentiment_val = data['sentiment'].to(device)
                sarcasm_val = data['sarcasm'].to(device)
                target_val = data['target'].to(device)


                abs_encoder.zero_grad()
                span_model.zero_grad()
                multitask.zero_grad()
                
                
                batch_size = doc.shape[0]
                
                doc = doc.reshape(batch_size,-1)
                
                abs_encoded = abs_encoder(doc)
                
                #sent_level, span_out = span_model(abs_encoded)
                sent_level = span_model(abs_encoded)
                
                #print(sent_level.shape)
                
                
                
                #emoti_out,senti_out,sar_out,output = multitask(sent_level,span_out)
                #senti_out,output = multitask(sent_level,span_out)
                #span_out,senti_out,output = multitask(sent_level,span_out)
                #span_out,senti_out,output = multitask(sent_level)
                senti_out,output = multitask(sent_level)
                #emoti_out,span_out,senti_out,output = multitask(sent_level)
                #span_out,output = multitask(sent_level)
                #output = multitask(sent_level,span_out)
                #output = multitask(sent_level)
                
                #span_loss_val = criterion_bce(span_out,span_val)
                #emoti_loss_val = criterion_loss(emoti_out, emotion_val.long().squeeze(1))
                senti_loss_val = criterion_loss(senti_out, sentiment_val.long().squeeze(1))
                bully_loss_val = criterion_loss(output, label_val.long().squeeze(1))
                #sar_loss_val = criterion_loss(sar_out, sarcasm_val.long().squeeze(1))
                
                #val_loss = span_loss_val + emoti_loss_val+ senti_loss_val + bully_loss_val + sar_loss_val 
                #val_loss = span_loss_val + senti_loss_val + bully_loss_val  
                val_loss = senti_loss_val + bully_loss_val  
                #val_loss = span_loss_val + senti_loss_val + bully_loss_val + emoti_loss_val 
                #val_loss = span_loss_val + bully_loss_val 
                #val_loss = bully_loss_val 
                
                _, predicted_val = torch.max(output.data, 1)
                #predicted_val = (output.data>=0.5).float()
                
                #print(predicted_val)
                #print(output.shape)
                
                total_val += label_val.size(0)
                #total_val += emotion_val.size(0)
                #_, label_val = torch.max(label_val, 1)
                correct_val += (predicted_val == label_val.squeeze(1)).sum().item()                
                #correct_val += (predicted_val == emotion_val).sum().item()                
                total_loss_val += val_loss.item()
                
                
        val_acc = 100 * correct_val / total_val
        val_loss = total_loss_val/total_val
        
        print(val_acc)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        early_stopping_span(val_loss, span_model)
        early_stopping_multitask(val_loss, multitask)
        early_stopping_abs(val_loss, abs_encoder)
        
        #print("Saving model...") 
        #torch.save(model.state_dict(), os.path.join(exp_path, "final.pt"))
        
        if early_stopping_span.early_stop:
            print("Early stopping")
            break
            

            
        
        print(f'Epoch {i+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
        
       
        span_model.train()
        multitask.train()
        abs_encoder.train()
        
        torch.cuda.empty_cache()
    
    # load the last checkpoint with the best model
    span_model.load_state_dict(torch.load(span_file))
    multitask.load_state_dict(torch.load(multitask_file))
    abs_encoder.load_state_dict(torch.load(abs_file))
    #model.load_state_dict(torch.load(os.path.join(exp_path,"final.pt")))
    
    return  span_model, multitask, abs_encoder , train_acc_list, val_acc_list, train_loss_list, val_loss_list, i
                        
                    
                    
                    
                
                    
            
 


def test_model(span_model, multitask, abs_encoder):
    span_model.eval() 
    multitask.eval() 
    abs_encoder.eval()
    
    
    total_test = 0
    correct_test =0
    total_acc_test = 0
    total_loss_test = 0
    outputs = []
    outputs_senti = []
    test_labels=[]
    with torch.no_grad():
        for data in dataloader_test:

            doc = data['doc'].to(device)
            sent_per_doc = data['sent_per_doc'].to(device)
            word_per_sent = data['word_per_sent'].to(device)

            label_test = data['label'].to(device)
            span_test = data['span'].to(device)
            target_test = data["target"].to(device)
            
            
            batch_size = doc.shape[0]
                
            doc = doc.reshape(batch_size,-1)
            
            abs_encoded= abs_encoder(doc)
            
            #sent_level, span_out = span_model(abs_encoded)
            sent_level = span_model(abs_encoded)
            
            
            #_,_,_,out = multitask(sent_level,span_out)
            #_,out = multitask(sent_level,span_out)
            #_,_,out = multitask(sent_level,span_out)
            #_,senti_out,out = multitask(sent_level)
            #_,senti_out,out = multitask(sent_level)
            senti_out,out = multitask(sent_level)
            #senti_out,_,_,out = multitask(sent_level)
            #out = multitask(sent_level,span_out)
            #out = multitask(sent_level)
            
            
            outputs_senti+=list(senti_out.cpu().data.numpy())
            outputs += list(out.cpu().data.numpy())
            loss = criterion_loss(out, label_test.squeeze(1).long())
            
            _, predicted_test = torch.max(out.data, 1)
            #predicted_test = (out.data>=0.5).float()
            total_test += label_test.size(0)
            #_, label_test = torch.max(label_test, 1)
            correct_test += (predicted_test == label_test.squeeze(1)).sum().item()
#                 out_val = (output.squeeze()>0.5).float()
#                 out_final = ((out_val == 1).nonzero(as_tuple=True)[0])
#                 print()
#                 acc = torch.abs(output.squeeze() - label.float()).view(-1)
#                 acc = (1. - acc.sum() / acc.size()[0])
#                 total_acc_train += acc
            total_loss_test += loss.item()
            
            
#     #         print(label.float())
#             acc = torch.abs(out.squeeze() - label.float()).view(-1)
#     #         print((acc.sum() / acc.size()[0]))
#             acc = (1. - acc.sum() / acc.size()[0])
#     #         print(acc)
#             total_acc_test += acc
#             total_loss_test += loss.item()

    
    acc_test = 100 * correct_test / total_test
    loss_test = total_loss_test/total_test   
    
    print(f'acc: {acc_test:.4f} loss: {loss_test:.4f}')
    return outputs_senti,outputs




n_epochs = 100
# early stopping patience; how long to wait after last time validation loss improved.
patience = 10
span_model, multitask, abs_encoder , train_acc_list, val_acc_list, train_loss_list, val_loss_list, epoc_num = train_model(abs_encoder,span_model,multitask, patience, n_epochs)

"""
span_file = os.path.join(exp_path, 'checkpoint_span.pt')
multitask_file = os.path.join(exp_path, 'checkpoint_mutlitask.pt')
abs_file = os.path.join(exp_path, 'checkpoint_absencoder.pt')

span_model.load_state_dict(torch.load(span_file))
multitask.load_state_dict(torch.load(multitask_file))
abs_encoder.load_state_dict(torch.load(abs_file))
"""

#chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
#model.load_state_dict(torch.load(chk_file))
# Plot the training and validation curves


outputs_senti, outputs = test_model(span_model, multitask, abs_encoder)

y_pred=[]
for i in outputs:
     y_pred.append(np.argmax(i))
# # np.argmax(outputs[:])
# outputs

y_pred_senti=[]
for i in outputs_senti:
    y_pred_senti.append(np.argmax(i))


print(len(y_pred))
test_labels=[]
test_labels_senti=[]

for index in range(len(test["label"])-1):
    #test_labels.append(np.argmax(test["label"][index]))
    test_labels.append(test["label"][index])
    test_labels_senti.append(test["sentiment"][index])
    #test_labels.append(emoti_label[test["emotion"][index]])
    #test_labels.append(test["sarcasm"][index][0])
    
"""
for index, row in test_samples_frame.iterrows():
    lab = row['labels'][0]
    if lab=="not harmful":
        test_labels.append(0)
    elif lab=="somewhat harmful":
        test_labels.append(1)
    else:
        test_labels.append(2)
"""

# In[ ]:


def calculate_mmae(expected, predicted, classes):
    NUM_CLASSES = len(classes)
    count_dict = {}
    dist_dict = {}
    for i in range(NUM_CLASSES):
        count_dict[i] = 0
        dist_dict[i] = 0.0
    for i in range(len(expected)):
        dist_dict[expected[i]] += abs(expected[i] - predicted[i])
        count_dict[expected[i]] += 1
    overall = 0.0
    for claz in range(NUM_CLASSES): 
        class_dist =  1.0 * dist_dict[claz] / count_dict[claz] 
        overall += class_dist
    overall /= NUM_CLASSES
#     return overall[0]
    return overall


# In[ ]:


rec = np.round(recall_score(test_labels, y_pred, average="weighted"),4)
prec = np.round(precision_score(test_labels, y_pred, average="weighted"),4)
f1 = np.round(f1_score(test_labels, y_pred, average="weighted"),4)
# hl = np.round(hamming_loss(test_labels, y_pred),4)
acc = np.round(accuracy_score(test_labels, y_pred),4)
#mmae = np.round(calculate_mmae(test_labels, y_pred, [0,1]),4)
#mae = np.round(mean_absolute_error(test_labels, y_pred),4)
# print("recall_score\t: ",rec)
# print("precision_score\t: ",prec)
# print("f1_score\t: ",f1)
# print("hamming_loss\t: ",hl)
# print("accuracy_score\t: ",f1)

print(classification_report(test_labels, y_pred))


# In[ ]:


print("Acc, F1, Rec, Prec, MAE, MMAE")
#print(acc, f1, rec, prec, mae, mmae)
print(acc, f1, rec, prec)



rec = np.round(recall_score(test_labels, y_pred),4)
prec = np.round(precision_score(test_labels, y_pred),4)
f1 = np.round(f1_score(test_labels, y_pred),4)
# hl = np.round(hamming_loss(test_labels, y_pred),4)
acc = np.round(accuracy_score(test_labels, y_pred),4)

print(acc,f1,rec,prec)




rec = np.round(recall_score(test_labels, y_pred),4)
rec_senti = np.round(recall_score(test_labels_senti, y_pred_senti,average="weighted"),4)
prec = np.round(precision_score(test_labels, y_pred),4)
prec_senti = np.round(precision_score(test_labels_senti, y_pred_senti,average="weighted"),4)
f1 = np.round(f1_score(test_labels, y_pred),4)
f1_senti = np.round(f1_score(test_labels_senti, y_pred_senti,average="weighted"),4)
# hl = np.round(hamming_loss(test_labels, y_pred),4)
acc = np.round(accuracy_score(test_labels, y_pred),4)
acc_senti = np.round(accuracy_score(test_labels_senti, y_pred_senti),4)

print(acc_senti,f1_senti,rec_senti,prec_senti)
