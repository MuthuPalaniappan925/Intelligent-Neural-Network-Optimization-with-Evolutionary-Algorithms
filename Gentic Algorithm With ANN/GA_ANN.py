#!/usr/bin/env python
# coding: utf-8
# User:@MuthuPalaniappan M

# # Importing Packages

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import utils,metrics
from sklearn import preprocessing as pp
import scipy
from statistics import mean
plt.style.use('dark_background')


# # Reading the Dataset

# In[4]:


data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
data.head()


# # Loading the Data

# In[5]:


x = data.drop(['Personal Loan','ID'],axis=1).values
x


# In[6]:


y = data['Personal Loan'].values
y


# # Train Test Split

# In[7]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=9)


# # Data Pre-Processing

# In[8]:


SC = StandardScaler()
x_train = SC.fit_transform(x_train)
x_test = SC.fit_transform(x_test)


# # Class GeneticANN

# In[32]:


class geneticANN:
    #init method
    def __init__(self,input_layer_size,hidden_layer_size,output_layer_size,population_size,low_limit,upper_limit,mutation_rate,crossover_rate):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.population_size = population_size
        self.low_limit = low_limit
        self.upper_limit = upper_limit
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    #Method to define the initial population
    def init_population(self):
        p_params = [dict() for _ in range(self.population_size)]
        for i in range(self.population_size):
            #Hidden and output layers weights
            w_hidden = np.random.uniform(self.low_limit,self.upper_limit,size=(self.hidden_layer_size,self.input_layer_size))
            w_outputs = np.random.uniform(self.low_limit,self.upper_limit,size=(self.output_layer_size,self.hidden_layer_size))
            #Hidden and output layers bias
            b_hidden = np.random.uniform(self.low_limit,self.upper_limit,size=(self.hidden_layer_size,1))
            b_output = np.random.uniform(self.low_limit,self.upper_limit,size=(self.output_layer_size,1))

            #Adding to Params
            p_params[i] = {"w_hidden":w_hidden,"w_outputs":w_outputs,"b_hidden":b_hidden,"b_output":b_output}

        return p_params
    #Calculating the loss and accuracy
    def acc_loss(self,current_generation,x,y):
        acc = [float() for _ in range(self.population_size)]
        loss = [float() for _ in range(self.population_size)]
        
        #Forward-Propagation
        for i in range(self.population_size):
            w_hidden = current_generation[i]['w_hidden']
            b_hidden = current_generation[i]['b_hidden']
            
            z_h = np.dot(w_hidden,x.T) + b_hidden
            ##Output of the hidden Layer
            activation_h = np.maximum(z_h,0)
            
            w_outputs = current_generation[i]['w_outputs']
            b_output = current_generation[i]['b_output']
            
            z_o = np.dot(w_outputs,activation_h) + b_output
            z_o = z_o.T
            #Softmax activation
            activation_o = scipy.special.softmax(z_o)
            
            #Getting the Index of the maximum probablity output values
            y_pred = np.argmax(activation_o,axis=1)
            y_pred = pp.label_binarize(y_pred,classes=[0,1])
            
            ##Calculating the acc and loss
            acc[i] = metrics.accuracy_score(y,y_pred)
            loss[i] = metrics.log_loss(y,activation_o)
            
        return (loss,acc)
    
    #Fitness Function
    def fitness_function(self,loss,acc):
        fitness_values = [float() for _ in range(self.population_size)]
        #Calculating Inverted Loss
        inv_loss = [1/x for x in loss]
        sum_inv_los = np.sum(inv_loss)
        
        #Calculating Fitness
        for i in range(self.population_size):
            fitness_values[i] = (inv_loss[i] / sum_inv_los) * 100
            
        return fitness_values
    
    #Parent Selection Method
    def parent_selection(self,fitness_values):
        ##Rouletee wheel selection
        count = 0
        p1 = -1
        p2 = -1
        
        while count != 2:
            rand_val = np.random.random_sample() * 100
            cul_freq = 0
            i = 0
            while cul_freq < rand_val and i < self.population_size:
                cul_freq = cul_freq + fitness_values[i]
                i+=1
            if p1 == -1:
                p1 = i
                count+=1
            
            elif p2 == -1 and p1 != i:
                ##Checking second parent should be the same
                p2 = i
                count = count + 1
        
        return (p1-1,p2-1)
    
    #Flatten Chromosomes
    def flatten_chromosomes(self,chromo_mat):
        flat_chromo = []
        for i in chromo_mat:
            flat_chromo = np.concatenate((flat_chromo,chromo_mat[i].flatten()))
        return flat_chromo
    
    #Unflatten Chromosomes
    def unflat_chromosomes(self,flat_chromo,iCtr,hCtr,oCtr):
        temp=np.split(flat_chromo,[hCtr*iCtr])
        wHflat=temp[0]
        flat_chromo=temp[1]

        temp=np.split(flat_chromo,[oCtr*hCtr])
        wOflat=temp[0]
        flat_chromo=temp[1]

        temp=np.split(flat_chromo,[hCtr])
        bHflat=temp[0]
        flat_chromo=temp[1]

        bOflat=flat_chromo

        w_hidden=np.array(wHflat).reshape(hCtr,iCtr)
        w_outputs=np.array(wOflat).reshape(oCtr,hCtr)
        b_hidden=np.array(bHflat).reshape(hCtr,1)
        b_output=np.array(bOflat).reshape(oCtr,1)

        chromoMat = {"w_hidden": w_hidden, "w_outputs": w_outputs, "b_hidden": b_hidden, "b_output": b_output}

        return chromoMat
    
    #Flatten Generation
    def flat_gen(self,current_generation):
        cur_flat_gen = []
        for i in current_generation:
            c_flat = self.flatten_chromosomes(i)
            cur_flat_gen.append(c_flat)
        return cur_flat_gen
    
    ##Un-Flatten Generations
    def un_flat_gen(self,cur_flat_gen,iCtr,hCtr,oCtr):
        current_generation = []
        for i in cur_flat_gen:
            c = self.unflat_chromosomes(i,iCtr,hCtr,oCtr)
            current_generation.append(c)
        return current_generation
    
    #Single-Point-Crossover
    def SPC(self,p1,p2,total_length):
        rd_idx = np.random.randint(1,total_length)
        
        child_1=np.concatenate((p2[0:rd_idx],p1[rd_idx:]))
        child_2=np.concatenate((p1[0:rd_idx],p2[rd_idx:]))
    
        return child_1,child_2
    
    #Mutation Operation
    def mutation(self,child,total_length):
        for i in range(total_length):
            rd_number = np.random.rand()
            if rd_number < self.mutation_rate:
                child[i] = float(np.random.uniform(self.low_limit,self.upper_limit,(1,1)))###shape = (1,1)
        return child
    
    #Form-Next_Population
    def next_generation(self,cur_flat_gen,fitness_values,loss,total_length):
        #Elitism Method
        elite = np.argsort(loss)[0:4]
        next_flat_gen = cur_flat_gen
        i = 0
        while i < 4:
            next_flat_gen[i] = cur_flat_gen[elite[i]]
            i+=1
        while i < self.population_size:
            p1,p2 = self.parent_selection(fitness_values)
            next_flat_gen[i] = cur_flat_gen[p1]
            next_flat_gen[i+1] = cur_flat_gen[p2]
            
            rand_val = np.random.rand()
            if rand_val < self.crossover_rate:
                child_1,child_2 = self.SPC(cur_flat_gen[p1],cur_flat_gen[p2],total_length)
                next_flat_gen[i] = child_1
                next_flat_gen[i+1] = child_2
            
            next_flat_gen[i] = self.mutation(next_flat_gen[i],total_length)
            next_flat_gen[i+1] = self.mutation(next_flat_gen[i+1],total_length)
            
            i+=2
        
        return next_flat_gen
    
    
    ##Ploting the Metrices
    def plot_Metrics(self, i, metricDict, metricName, maxGen, xStep, yStep):
    
        plt.figure(i)

        xMaxVal = maxGen + 1
        yMaxVal = max(metricDict['max']) + 1

        # Generate x-axis ticks as multiples of 10
        xTicks = np.arange(0, xMaxVal, 10)

        plt.xlim(0, xMaxVal)
        plt.ylim(0, yMaxVal)

        # Use the modified xTicks to set the x-axis ticks
        plt.xticks(xTicks)
        plt.yticks(np.arange(0, yMaxVal, yStep))

        plt.xlabel("Generation")
        plt.ylabel(metricName)
        plt.title("ANN %s vs Generation" % (metricName))

        maxColor = "green"
        minColor = "red"
        meanColor = "orange"

        m1, = plt.plot(metricDict['max'], color=maxColor)
        m2, = plt.plot(metricDict['min'], color=minColor)
        m3, = plt.plot(metricDict['mean'], color=meanColor)

        plt.legend([m1, m2, m3], ["Max " + metricName, "Min " + metricName, "Average " + metricName], loc="upper right")

        return


# # Defining the Hyper Parameters

# In[48]:


input_layer_size = x_train.shape[1]
hidden_layer_size = 8
output_layer_size = np.unique(y).shape[0]
population_size = 30
low_limit = -2.0
upper_limit = 2.0
mutation_rate = 0.02
crossover_rate = 0.9
number_of_generations = 50


# # Creating the Instances of the geneticANN Class

# In[40]:


gaann = geneticANN(input_layer_size,hidden_layer_size,output_layer_size,population_size,low_limit,upper_limit,mutation_rate,crossover_rate)


# # Preprocessing

# In[41]:


encoder = pp.LabelBinarizer()
y = encoder.fit_transform(y)
X_shuffle, Y_shuffle = utils.shuffle(x, y)


# # Dictionary of Lists to store population metrics for plotting graphs for Loss,Fitness and Accuracy 

# In[42]:


LossLines = {'max':[],'min':[],'mean':[]}
FitnessLines = {'max':[],'min':[],'mean':[]}
AccuracyLines = {'max':[],'min':[],'mean':[]}


# # Calculating the Total Params

# In[43]:


total_length = input_layer_size * hidden_layer_size + hidden_layer_size * output_layer_size + hidden_layer_size + output_layer_size


# # Genetic with ANN

# In[49]:


print("\n  Generating initial population...")
init_population = gaann.init_population()
curGen = init_population
print("\n  Creating next generations...")
for i in range(0,number_of_generations):
    loss,acc = gaann.acc_loss(curGen,X_shuffle,Y_shuffle)
    fitness = gaann.fitness_function(loss,acc)
    
    cur_gen_flat = gaann.flat_gen(curGen)
    next_gen_flat = gaann.next_generation(cur_gen_flat,fitness,loss,total_length)
    nexGEN = gaann.un_flat_gen(next_gen_flat,input_layer_size,hidden_layer_size,output_layer_size)
    
    print("\n\tGen %d Done!"%(i+1),end="")
    curGen = nexGEN
    
    LossLines['max'].append(max(loss))
    LossLines['min'].append(min(loss))
    LossLines['mean'].append(mean(loss))

    FitnessLines['max'].append(max(fitness))
    FitnessLines['min'].append(min(fitness))
    FitnessLines['mean'].append(mean(fitness))
    
    AccuracyLines['max'].append(max(acc)*100)
    AccuracyLines['min'].append(min(acc)*100)
    AccuracyLines['mean'].append(mean(acc)*100)
    
xstep = 100
gaann.plot_Metrics(0,LossLines,"LOSS",number_of_generations,100,1)
gaann.plot_Metrics(1,FitnessLines,"FITNESS",number_of_generations,100,1)
gaann.plot_Metrics(2,AccuracyLines,"ACCURACY",number_of_generations,100,5)

plt.show()


# In[ ]:




