import bcolz
import numpy as np
import pickle


vectors = bcolz.open('storagedat')[:]

words = pickle.load(open('worddump.pkl', 'rb'))

word2idx = pickle.load(open('vectordumpwordsID.pkl', 'rb'))



#TranslationDictionary = {w: vectors[word2idx[w]] for w in words}



Int2WordDictionary = {}
Number2Question = {}
#weights matrix is set up
weights_matrix=np.zeros((len(words)+2, 200))
#print(weights_matrix.shape)
for i in range(weights_matrix.shape[0]-2):
    weights_matrix[i]=vectors[i]

weights_matrix[len(words)+1] = np.random.rand(200) # value for unknown word
#note: weights_matrix[len(words)] should = 0 so that we can have a padding value


#%%

import torch
import re

text_in = open('../text_tokenized.txt', 'r', encoding = "utf8")


processed_line = [[]]




questionlist = [[]]
qnumber = []
line_counter = 0
unique_counter = 0

# lets parse the text and turn it into 3 sets, the first is the question number
# The second is the question text, and 3rd is the body which I will exclude here
intquestion = []

for line in text_in:
    intquestion = [] # lets prepare a list that will hold questions translated to integers
    num_q_body = re.split(r'\t+', line)
    qnumber.append(num_q_body[0]) # qnumber stores questions
    
    question = re.split('\W+',num_q_body[1])
    for i in question:
        try:
            intquestion.append(word2idx[i])
        except KeyError:
            intquestion.append(len(words)+1)
    
    #questionlist[line_counter] = question # questionlist stores parsed question text
    #questionlist.append([])
    Number2Question[int(num_q_body[0])] = intquestion 
    # make a dictionary with number as a key and  intquestion as a value
    
    
    counter = 0
    line_vec = []
    
    
    line_counter = line_counter + 1


# Now we have all of the questions listed by number with all of 
# the words converted to integers


        #try: 

         #   word_vec = TranslationDictionary[i]
          #  line_vec.append(word_vec)
           # counter = counter + 1
            
        #except KeyError:
        #    line_vec.append(np.zeros(200))
    
    
    #processed_data[line_counter] = line_vec
    #processed_data.append([])
    #line_counter = line_counter + 1
    #print(line_counter)


    
    #base = re.split('\W+', num_q_body[2])
#%%

#processed_data = np.asarray(processed_data, dtype=np.float32)
#qnumber = np.asarray(qnumber, dtype=np.int)


#weights_matrix=np.zeros((unique_counter, 200))

#for keys in Int2WordDictionary:
 #   weights_matrix[Int2WordDictionary[keys]] = TranslationDictionary[keys]

#weights_matrix[unique_counter] = np.zeros(1,200)
#%%

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim



def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))

    #emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    #emb_layer.load_state_dict({'weight': weights_matrix})
    
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class CNN(nn.Module):
    def __init__(self,  hidden_size, filter_size ):

        super(CNN,self).__init__()
        self.embedding, self.num_embeddings, self.embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size # hidden_size is the num_channel_out after conv layer
        self.filter_size = filter_size # kernel size for conv layer
        self.conv=nn.Conv2d(1, self.hidden_size, (self.filter_size, self.embedding_dim))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, input):
        d = self.embedding(input) # batch_size * seq_len * embedding_dim(200)
        d = torch.unsqueeze(d, 1) # batch_size * 1 * seq_len * embedding_dim(200)
        c = self.conv(d) # output has dimension (batch_size * hidden_size * seq_len * 1)
        h = self.relu(c.squeeze(3)) # (batch_size * hidden_size * seq_len); relu doesnt change shape
        mypool=nn.MaxPool1d(kernel_size=h.size()[2]) # the size of the window to take a max over is the 3rd dim of h
        h=mypool(h) # (batch_size * hidden_size * 1)
        h=h.squeeze(2) # (batch_size * hidden_size)
        h = self.dropout(h) # (batch_size * hidden_size)
        return h

#%%
        

hidden_size = 100
learning_rate = 1e-3 #another option is 3e-4
wd = 0
n_filter = 4
n_negative = 20 # number of negative questions
#dropout = 0.1 # two more options 0.2 and 0.3

cnnmodel=CNN(hidden_size, n_filter)

lossfunction = nn.MultiMarginLoss(p=1, margin=0.2)

optimizer = optim.Adam(cnnmodel.parameters(), lr=learning_rate, weight_decay=wd)



#%%
import sys


def cnntrain(query_embedding, positive_embedding, negative_embedding):
    n_batch = len(query_embedding)
    optimizer.zero_grad()
    similarity_matrix = Variable(torch.zeros(n_batch, n_negative+1))
    
	#query_vec = cnnmodel(Variable(torch.FloatTensor(query_embedding)))
    query_vec = cnnmodel(Variable(torch.LongTensor(query_embedding)))
    #postive_vec = cnnmodel(Variable(torch.FloatTensor(positive_embedding)))
    postive_vec = cnnmodel(Variable(torch.LongTensor(positive_embedding)))
    #ForReading=Variable(torch.LongTensor(query_embedding))
    #ForReadingpositive=Variable(torch.LongTensor(positive_embedding))
    #query_vec = torch.squeeze(query_vec)
    #postive_vec = torch.squeeze(postive_vec)
    #print(Variable(torch.FloatTensor(query_vec)).shape)
    #print(Variable(torch.FloatTensor(postive_vec)).shape)
    #print("hello")
    #print(Variable(torch.LongTensor(negative_embedding)).shape)
    cosqp = nn.functional.cosine_similarity(query_vec, postive_vec)
    #print(cosqp.shape)
    similarity_matrix[:,0] = cosqp
    #print(Variable(torch.LongTensor(negative_embedding[0])).shape)
    for i in range(n_negative):
        negative_vec = cnnmodel(Variable(torch.LongTensor(negative_embedding[i:500:20])))
        #print("is i")
        cosqn = nn.functional.cosine_similarity(query_vec, negative_vec)
        similarity_matrix[:,i+1] = cosqn

    target = [0 for i in range(n_batch)]
    target = Variable(torch.LongTensor(target))

    loss = lossfunction(similarity_matrix, target)
    loss.backward()
    optimizer.step()
    s = torch.sum(cnnmodel.conv.weight.data)
    print(s)
    return loss.mean()

def ModelTraining():
    training_file = open("../train_random.txt", "r", encoding = "utf8")
    batch_size = 25
    line_counter = 0
    negative_collection = []
    positive_collection = [[]]
    query_collection = [[]]
    negative_total = 0
    batch_count = 0
    bundle_counter = 0
    for line in training_file:
        #if bundle_counter % (batch_size) == 0;:
         #   maxlength=0;
        question_set = re.split(r'\t+', line)
        question = question_set[0] 
        positive = question_set[1].split()
        negative = question_set[2].split()
        
        
        
        
        question_text = Number2Question[int(question)]
        #if len(question_text)>maxlength:
        #    maxlength = len(question_text)
        #print(negative_total)
        #print(Number2Question[int(negative[0])])
        #print(negative)
        negative_counter = 0
        positive_counter = 0
        # negative_text = [[]]
        for i in positive:

            for counter,j in enumerate(negative):
                negative_collection.append(Number2Question[int(j)][:])
                #if len(negative_collection[negative_total])>maxlength:
                 #   maxlength = len(negative_collection[negative_total])
                counter+=1
                if (counter==n_negative):
                    break
                
        #negative_collection.remove(negative_total)
        
        
        for i in positive:
           # print(int(i))
            #print("are positive")
            #print(int(question))
            #print("is question")
            positive_collection[bundle_counter] =  Number2Question[int(i)][:]
            
            query_collection[bundle_counter]=Number2Question[int(question)][:]
            query_collection.append([])
            positive_collection.append([])
            bundle_counter = bundle_counter + 1
            batch_count = batch_count + 1
            
        
            #print(int(j))
            #print(" are negative")
        
        # line_counter = line_counter + 1
        # if(line_counter==2):
        #     sys.exit()
        
            #for j in range(negative_counter):
             #   negative_collection[negative_total] = negative_text[j]
              #  negative_collection.append([])
               # negative_total = negative_total +1;
                
            #if len(positive_text)>maxlength:
             #   maxlength = len(positive_text)
        
    #del negative_collection[negative_total]
    del positive_collection[bundle_counter]
    del query_collection[bundle_counter]
    for i in range(bundle_counter):

        #let's pad the batches now
        if i % (batch_size) == 0:
            maxlength=0
            
        for j in range(i*n_negative, (i+1)*n_negative):
            if len(negative_collection[j])>maxlength:
                maxlength = len(negative_collection[j])
        
        if len(positive_collection[i])>maxlength:
            maxlength = len(positive_collection[i]) 
        
        if len(query_collection[i])>maxlength:
            maxlength = len(query_collection[i])
        
        if i % (batch_size) == batch_size-1:
            for j in range(i-batch_size+1, i+1):
                if len(query_collection[j])<maxlength:
                    buffer = [len(words)]*(maxlength - len(query_collection[j]))
                    query_collection[j].extend(buffer)
                if len(positive_collection[j])<maxlength:
                    buffer = [len(words)]*(maxlength - len(positive_collection[j]))
                    positive_collection[j].extend(buffer)
                for k in range(j*n_negative, (j+1)*n_negative):
                    if len(negative_collection[k])<maxlength:
                        buffer = [len(words)]*(maxlength - len(negative_collection[k]))
                        negative_collection[k].extend(buffer)
            
        




	#batch_size = 25
    batch= int(bundle_counter/batch_size)
    #print(batch)
    all_loss = []
    for t in range(15):
        print('iteration=', t)
		
        for i in range(batch):
           # print('batch =', i)
            s = i*batch_size
            ie = (i+1)*batch_size
            q_batch = query_collection[s:ie]
            p_batch = positive_collection[s:ie]
            n_batch = negative_collection[n_negative*s:n_negative*ie]
			#q_batch = query_embedding[s:ie,:,:]
			#p_batch = positive_embedding[s:ie,:,:]
			#n_batch = negative_embedding[s:ie,:,:,:]

            loss=cnntrain(q_batch, p_batch, n_batch)
            print(loss)
            all_loss.append(loss)

    modelname = 'cnn_saved_model'
    torch.save(cnnmodel, modelname)

    import matplotlib.pyplot as plt
    plt.plot(all_loss)
    plt.show()
    
    
#%%
    
ModelTraining()

#%%



