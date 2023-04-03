##Importing Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pp
from mpl_toolkits.mplot3d import Axes3D
import scipy
from statistics import mean
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import utils,metrics
import streamlit as st
import seaborn as sns
plt.style.use('dark_background')
old_settings = np.seterr(all='ignore')

data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
x = data.drop(['Personal Loan','ID'],axis=1).values
y = data['Personal Loan'].values

##Spliting into Training and Testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=9)

##Pre_processing the Dataset
SC = StandardScaler()
x_train = SC.fit_transform(x_train)
x_test = SC.fit_transform(x_test)
## Class for PCO with ANN
class PSOANN:
    
    #init_method
    def __init__(self,input_layer_size,hidden_layer_size,output_layer_size,particle_size,max_iterations,c_1,c_2,inertial_weight):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.particle_size = particle_size
        self.max_iterations = max_iterations
        self.c_1 = c_1
        self.c_2 = c_2
        self.inertial_weight = inertial_weight
        
        #Social learning and personal learning parameters definition
        self.global_fitness = float('inf')
        self.global_best_position = np.zeros((self.input_layer_size * self.hidden_layer_size) + (self.hidden_layer_size * self.output_layer_size))
        self.particles = np.zeros((particle_size,(input_layer_size * hidden_layer_size) + (hidden_layer_size * output_layer_size)))
        self.velocities = np.zeros((particle_size,(input_layer_size * hidden_layer_size) + (hidden_layer_size * output_layer_size)))
        self.local_best_positions = np.zeros((particle_size,(input_layer_size * hidden_layer_size) + (hidden_layer_size * output_layer_size)))
        self.local_best_fitness = np.full(particle_size,float('inf'))
        
        #Defining the hidden weights and the output layer weights
        self.hidden_layer_weights = np.zeros((input_layer_size,hidden_layer_size))
        self.output_layer_weights = np.zeros((hidden_layer_size,output_layer_size))
        
    #Activation Function
    def sigmoid(self,x):
        act_val = 1 / (1+np.exp(-x))
        return act_val
    
    #Forward-Propagation
    def forward(self,input_data):
        hidden_layer = self.sigmoid(np.dot(input_data,self.hidden_layer_weights))
        output_layer = np.dot(hidden_layer,self.output_layer_weights)
        return output_layer
    
    #Defining the fitness Function
    def fitness_function(self,particle):
        self.hidden_layer_weights = np.reshape(particle[:self.input_layer_size * self.hidden_layer_size],(self.input_layer_size,self.hidden_layer_size))
        self.output_layer_weights = np.reshape(particle[self.input_layer_size * self.hidden_layer_size:],(self.hidden_layer_size,self.output_layer_size))
        pred = np.round(self.forward(x_train))
        acc = np.sum(pred == y_train) / len(y_train)
        return 1 - acc
    
    #Training PHASE
    def train(self):
        g_best_ = []
        lb_p_ = []

        self.particles = np.random.uniform(low=-1,high=1,size=(self.particle_size,(self.input_layer_size * self.hidden_layer_size) + (self.hidden_layer_size * self.output_layer_size)))
        #For each and every particle
        for i in range(self.particle_size):
            fitness_value = self.fitness_function(self.particles[i])
            #Updating the global and local best position and fitness value
            #Personal-Learning params
            if fitness_value < self.local_best_fitness[i]:
                self.local_best_fitness[i] = fitness_value
                self.local_best_positions[i] = self.particles[i]
            
            #Social-Learning params
            if fitness_value < self.global_fitness:
                self.global_fitness = fitness_value
                self.global_best_position = self.particles[i]
                
        for i in range(self.max_iterations):
            #Over-Iterations
            for j in range(self.particle_size):
                #For-Each-Every-Particle
                r1 = np.random.uniform(0,1)
                r2 = np.random.uniform(0,1)
                #Velocity of the next iterations will be the combination of Inertia + Personal_Learning + Social_Learning
                new_vel = (self.inertial_weight * self.velocities[j]) + (self.c_1 * r1 * (self.local_best_positions[j] - self.particles[j])) + (self.c_2 * r2 * (self.global_best_position - self.particles[j]))
                #New Position will be New_Position + Velocity of the next iterations
                self.particles[j] = self.particles[j] + new_vel
                #Update the social learning and global learning
                fitness_value = self.fitness_function(self.particles[j])
                if fitness_value < self.local_best_fitness[j]:
                    self.local_best_fitness[j] = fitness_value
                    self.local_best_positions[j] = self.particles[j]
                    lb_p_.append(self.local_best_positions[j])
                if fitness_value < self.global_fitness:
                    self.global_fitness = fitness_value
                    self.global_best_position = self.particles[j]
                    g_best_.append(self.global_best_position)

        #Plotting the global and local best positions
        g_best_ = np.array(g_best_)
        lb_p_ = np.array(lb_p_)
        
        print(f'Global Best Shape After Training: {g_best_.shape}')
        print(f'Local Best Shape After Training: {lb_p_.shape}')
                    
    #Predict Function
    def predict(self,x_test):
        hidden_layer = self.sigmoid(np.dot(x_test,self.hidden_layer_weights))
        output_layer = np.dot(hidden_layer,self.output_layer_weights)
        return np.round(output_layer)
    
    #Visualize the particles after the optimization to see the movement of the particles/swarm
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = [p[0] for p in self.particles]
        y = [p[1] for p in self.particles]
        z = [p[2] for p in self.particles]
        ax.scatter(x, y, z, c='r', s=50)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('Particle distribution in 3D')
        ax.set_facecolor('white')
        ax.grid(True)
        st.pyplot(fig)

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
    def plot_Metrics(self,i,metricDict, metricName, maxGen, xStep, yStep):

        st.set_option('deprecation.showPyplotGlobalUse', False)

        xMaxVal = maxGen + 1
        yMaxVal = max(metricDict['max']) + 1

        # Generate x-axis ticks as multiples of 10
        xTicks = np.arange(0, xMaxVal, 10)

        fig, ax = plt.subplots()

        ax.set_xlim(0, xMaxVal)
        ax.set_ylim(0, yMaxVal)

        # Use the modified xTicks to set the x-axis ticks
        ax.set_xticks(xTicks)
        ax.set_yticks(np.arange(0, yMaxVal, yStep))

        ax.set_xlabel("Generation")
        ax.set_ylabel(metricName)
        ax.set_title("ANN %s vs Generation" % (metricName))

        maxColor = "green"
        minColor = "red"
        meanColor = "orange"

        m1, = ax.plot(metricDict['max'], color=maxColor)
        m2, = ax.plot(metricDict['min'], color=minColor)
        m3, = ax.plot(metricDict['mean'], color=meanColor)

        ax.legend([m1, m2, m3], ["Max " + metricName, "Min " + metricName, "Average " + metricName], loc="upper right")

        st.pyplot(fig)

        return


LossLines = {'max':[],'min':[],'mean':[]}
FitnessLines = {'max':[],'min':[],'mean':[]}
AccuracyLines = {'max':[],'min':[],'mean':[]}



class ANN_Model:
    #Init Class
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.uniform(-1, 1, size=(input_size, hidden_size))
        self.biases1 = np.random.uniform(-1, 1, size=hidden_size)
        self.weights2 = np.random.uniform(-1, 1, size=(hidden_size, output_size))
        self.biases2 = np.random.uniform(-1, 1, size=output_size)

    def forward(self, X):
        hidden = np.dot(X, self.weights1) + self.biases1
        hidden = 1 / (1 + np.exp(-hidden))
        output = np.dot(hidden, self.weights2) + self.biases2
        return output


class ACONN:
    #init_Method
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)
        
    ##Forward Propagation
    def forward_Propagation(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        self.y_hat = self.sigmoid(self.z2)
        return self.y_hat
    
    #Sigmoid Activation Function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    #Fitness Function / Cost Function
    def cost_function(self, X, y):
        self.y_hat = self.forward_Propagation(X)
        J = 0.5 * np.sum((y - self.y_hat) ** 2)
        return J

    def ant_colony_optimization(self, X, y, max_iterations, num_ants, alpha, beta, rho, q):
        # Initialize the list to store the cost over generations
        cost_history_run = []
        pheromone = np.ones((self.input_layer_size, self.hidden_layer_size, self.output_layer_size))
        for i in range(max_iterations):
            ants = []
            for j in range(num_ants):
                ant = {}
                ant['path'] = []
                ant['pheromone'] = []
                ant['cost'] = 0
                ant['W1'] = np.zeros((self.input_layer_size, self.hidden_layer_size))
                ant['W2'] = np.zeros((self.hidden_layer_size, self.output_layer_size))
                ant['path'].append(np.random.randint(self.input_layer_size))
                ant['path'].append(np.random.randint(self.hidden_layer_size))
                for k in range(self.output_layer_size):
                    ant['path'].append(np.random.randint(self.hidden_layer_size))
                for k in range(self.input_layer_size):
                    for l in range(self.hidden_layer_size):
                        ant['pheromone'].append(pheromone[k][l][0])
                for k in range(self.hidden_layer_size):
                    for l in range(self.output_layer_size):
                        ant['pheromone'].append(pheromone[0][k][l])
                for k in range(self.input_layer_size):
                    for l in range(self.hidden_layer_size):
                        if np.random.rand() < ant['pheromone'][k * self.hidden_layer_size + l]:
                            ant['W1'][k][l] = np.random.randn()
                for k in range(self.hidden_layer_size):
                    for l in range(self.output_layer_size):
                        if np.random.rand() < ant['pheromone'][self.input_layer_size * self.hidden_layer_size + k * self.output_layer_size + l]:
                            ant['W2'][k][l] = np.random.randn()
                ant['cost'] = self.cost_function(X, y)
                ants.append(ant)
            ants = sorted(ants, key=lambda x: x['cost'])
            best_ant = ants[0]
            for k in range(self.input_layer_size):
                for l in range(self.hidden_layer_size):
                    pheromone[k][l][0] = (1 - rho) * pheromone[k][l][0] + q / best_ant['cost'] * (best_ant['W1'][k][l] != 0)
            for k in range(self.hidden_layer_size):
                for l in range(self.output_layer_size):
                    pheromone[0][k][l] = (1 - rho) * pheromone[0][k][l] + q / best_ant['cost'] * (best_ant['W2'][k][l] != 0)
                    self.W1 = best_ant['W1']
                    self.W2 = best_ant['W2']
            print(f"Generation {i+1}, Best Cost: {best_ant['cost']}")
            cost_history_run.append(best_ant['cost'])
        plt.plot(cost_history_run)
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.title("Cost History Run")
        st.pyplot()
                    
    def train(self, X, y, max_iterations, num_ants, alpha, beta, rho, q):
        self.ant_colony_optimization(X, y, max_iterations, num_ants, alpha, beta, rho, q)

    def predict(self, X):
        y_hat = self.forward_Propagation(X)
        predictions = np.round(y_hat)
        return predictions


##Define the Streamlit app
def app():
    evolutionary_algorithms = ['Genetic Algorithm', 'Cultural Evolution', 'Particle Swarm Optimization', 'Ant Colony Optimization']
    st.title('Intelligent Neural Network Optimization with Evolutionary Algorithm')
    selected_algorithm = st.selectbox('Select an evolutionary algorithm:', evolutionary_algorithms)
    ##Loading the Dataset
    data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
    x = data.drop(['Personal Loan','ID'],axis=1).values
    y = data['Personal Loan'].values

    ##Spliting into Training and Testing sets
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=9)

    ##Pre_processing the Dataset
    SC = StandardScaler()
    x_train = SC.fit_transform(x_train)
    x_test = SC.fit_transform(x_test)
    st.info("Dataset Information")
    st.dataframe(data)

    if selected_algorithm == 'Particle Swarm Optimization':
        count=0
        st.sidebar.header("PSO-ANN Control Parameters")
        count+=1
        n_particles = st.sidebar.slider("Number of particles", min_value=10, max_value=100, value=50,key = count)
        count+=1
        n_iterations = st.sidebar.slider("Number of iterations/epochs", min_value=10, max_value=1000, value=100,key = count)
        count+=1
        c1 = st.sidebar.slider("C1 - acceleration coefficient", min_value=0.0, max_value=2.0, value=0.5,key = count)
        count+=1
        c2 = st.sidebar.slider("C2 - acceleration coefficient", min_value=0.0, max_value=2.0, value=0.5,key = count)
        count+=1
        w = st.sidebar.slider("Inertia weight", min_value=0.0, max_value=1.0, value=0.9,key = count)
        count+=1
        h_layer = st.sidebar.slider("Hidden Layer Size", min_value=1, max_value=10, value=5, key=count)
        count+=1

        if st.sidebar.button('Perform PSO Technique for ANN'):
            data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
            x = data.drop(['Personal Loan','ID'],axis=1).values
            y = data['Personal Loan'].values

            ##Spliting into Training and Testing sets
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=9)

            ##Pre_processing the Dataset
            SC = StandardScaler()
            x_train = SC.fit_transform(x_train)
            x_test = SC.fit_transform(x_test)
            psoann = PSOANN(input_layer_size=x_train.shape[1],hidden_layer_size = h_layer, output_layer_size=1,particle_size = n_particles, max_iterations = n_iterations, c_1 = c1,c_2 = c2,inertial_weight = w)
            with st.spinner("Running PSO for weights Optimization"):

                psoann.train()
                st.info("Here is the particle distribution in 3D")
                psoann.plot()
                y_pred = psoann.predict(x_test)
                accuracy = accuracy_score(y_pred, y_test)
                st.success(f"The accuracy score is: {accuracy}")
                ## Calculate confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_title("Confusion Matrix", fontsize=16)
                sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False, ax=ax)
                ax.set_xlabel("Predicted Labels", fontsize=14)
                ax.set_ylabel("True Labels", fontsize=14)
                st.pyplot(fig)
                parameters = {'Parameter': ['Number of particles', 'Number of iterations', 'C1', 'C2', 'Inertia weight','Hidden Layers Size'],
                'Value': [n_particles, n_iterations, c1, c2, w,h_layer]}

                # create a pandas dataframe from the dictionary
                #st.title("Summary of the Params")
            df = pd.DataFrame(parameters)
            st.info("Parameters")
            st.dataframe(df)

    if selected_algorithm == 'Genetic Algorithm':
        count=0
        st.sidebar.header("GA-ANN Control Parameters")
        count+=1
        n_particles = st.sidebar.slider("Population Size", min_value=10, max_value=100, value=30,key = count)
        count+=1
        number_of_generations = st.sidebar.slider("Number of Generations", min_value=10, max_value=100, value=50,key = count)
        count+=1
        low_limit = st.sidebar.slider("Lower Limit coefficient", min_value=-2.0, max_value=0.0, value=-0.5,key = count)
        count+=1
        upper_limit = st.sidebar.slider("Upper Limit coefficient", min_value=0.0, max_value=2.0, value=0.5,key = count)
        count+=1
        mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.0, max_value=0.2, value=0.01,key = count)
        count+=1
        crossover_rate = st.sidebar.slider("Cross-Over Rate", min_value=0.5, max_value=0.9, value=0.7, key=count)
        count+=1
        h_layer = st.sidebar.slider("Hidden Layer Size", min_value=1, max_value=10, value=5, key=count)
        count+=1
        output_layer_size = 1
        input_layer_size = x_train.shape[1]
        total_length = input_layer_size * h_layer + h_layer * output_layer_size + h_layer + output_layer_size

        if st.sidebar.button('Perform Genetic Technique for ANN'):
            ##Creating the Instance for Genetic Algorithm
            gaann = geneticANN(input_layer_size,h_layer,output_layer_size,n_particles,low_limit,upper_limit,mutation_rate,crossover_rate)
            ##Pre-Processing Steps
            encoder = pp.LabelBinarizer()
            y = encoder.fit_transform(y)
            X_shuffle, Y_shuffle = utils.shuffle(x, y)
            with st.spinner('Creating Initial generations...'):
                init_population = gaann.init_population()
                curGen = init_population
            with st.spinner('Creating next generations...'):
                for i in range(0,number_of_generations):
                    loss,acc = gaann.acc_loss(curGen,X_shuffle,Y_shuffle)
                    fitness = gaann.fitness_function(loss,acc)
                    
                    cur_gen_flat = gaann.flat_gen(curGen)
                    next_gen_flat = gaann.next_generation(cur_gen_flat,fitness,loss,total_length)
                    nexGEN = gaann.un_flat_gen(next_gen_flat,input_layer_size,h_layer,output_layer_size)
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
                st.info("Completed - %d Generations"%number_of_generations)
                st.success("Loss VS Generation")
                gaann.plot_Metrics(0,LossLines,"LOSS",number_of_generations,100,1)
                st.success("Fitness Vs Generation")
                gaann.plot_Metrics(1,FitnessLines,"FITNESS",number_of_generations,100,1)
                st.success("Accuracy Vs Generation")
                gaann.plot_Metrics(2,AccuracyLines,"ACCURACY",number_of_generations,100,5)

    if selected_algorithm == 'Cultural Evolution':
        count=0
        st.sidebar.header("CA-ANN Control Parameters")
        count+=1
        pop_size = st.sidebar.slider("Population Size", min_value=10, max_value=100, value=30,key = count)
        count+=1
        num_generations = st.sidebar.slider("Number of Generations", min_value=10, max_value=100, value=50,key = count)
        count+=1
        low_limit = st.sidebar.slider("Lower Limit coefficient", min_value=-2.0, max_value=0.0, value=-0.5,key = count)
        count+=1
        upper_limit = st.sidebar.slider("Upper Limit coefficient", min_value=0.0, max_value=2.0, value=0.5,key = count)
        count+=1
        p_accept = st.sidebar.slider("Acceptance Rate", min_value=0.0, max_value=1.0, value=0.5,key = count)
        count+=1
        step_size = st.sidebar.slider("Step Size", min_value=0.0, max_value=1.0, value=0.1, key=count)
        count+=1
        h_layer = st.sidebar.slider("Hidden Layer Size", min_value=1, max_value=10, value=5, key=count)
        count+=1
        output_layer_size = 1
        if st.sidebar.button('Perform Cultural Algorithm for ANN'):
            # Loading the dataset
            data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
            X = data.drop(['Personal Loan','ID'],axis=1).values
            y = data['Personal Loan'].values
            input_size = X.shape[1]
            output_size = len(np.unique(y))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            ann = ANN_Model(input_size,h_layer,output_size)
            # Define the fitness function to get the accuracy
            def fitness_function(solution):
                ann.weights1 = np.reshape(solution[:input_size*h_layer], (input_size, h_layer))
                ann.biases1 = np.reshape(solution[input_size*h_layer:(input_size+1)*h_layer], h_layer)
                ann.weights2 = np.reshape(solution[(input_size+1)*h_layer:(input_size+1)*h_layer+h_layer*output_size], (h_layer, output_size))
                ann.biases2 = np.reshape(solution[(input_size+1)*h_layer+h_layer*output_size:], output_size)
                y_pred = np.argmax(np.round(ann.forward(X_train)), axis=1)
                accuracy = accuracy_score(y_train, y_pred)
                return accuracy
            num_weights = input_size*h_layer + h_layer + h_layer*output_size + output_size
            population = [(np.random.uniform(-1, 1, size=num_weights), -1) for i in range(pop_size)]

            # Initialize the best solution and its fitness score
            best_solution = None
            best_fitness = -1

            # Initialize the list to store the accuracy over generations
            accuracy_history_run = []
            # Iterate for the specified number of generations
            for t in range(num_generations):
                # Evaluate the fitness of each solution in the population
                for i in range(pop_size):
                    population[i] = (population[i][0], fitness_function(population[i][0]))

                # Sort the population by fitness score
                population = sorted(population, key=lambda x: x[1], reverse=True)

                # Update the best solution and its fitness score
                if population[0][1] > best_fitness:
                    best_solution = population[0][0]
                    best_fitness = population[0][1]
                    # Initialize the new population
                new_population = []

                # Elitism: Add the best solution from the previous generation to the new population
                new_population.append((best_solution, best_fitness))

                # Generate the new solutions using the cultural algorithm
                for i in range(1, pop_size):
                    # Select two parents
                    parent1, _ = population[np.random.randint(pop_size)]
                    parent2, _ = population[np.random.randint(pop_size)]

                    # Perform crossover and mutation
                    child = np.zeros(num_weights)
                    for j in range(num_weights):
                        if np.random.rand() < p_accept:
                            child[j] = parent1[j]
                        else:
                            child[j] = parent2[j]
                        if np.random.rand() < step_size:
                            child[j] += np.random.normal(scale=0.1)

                    # Evaluate the fitness of the child
                    fitness = fitness_function(child)

                    # Add the child to the new population
                    new_population.append((child, fitness))

                # Set the population to the new population
                population = new_population
                st.set_option('deprecation.showPyplotGlobalUse', False)
                # Print the best fitness score in the current generation
                st.write(f'Generation {t+1}: Best fitness = {best_fitness}', end='\r')

                # Append the accuracy of the best solution to the accuracy history
                accuracy_history_run.append(best_fitness)
            # Evaluate the accuracy of the best solution on the test set
            ann.weights1 = np.reshape(best_solution[:input_size*h_layer], (input_size, h_layer))
            ann.biases1 = np.reshape(best_solution[input_size*h_layer:(input_size+1)*h_layer], h_layer)
            ann.weights2 = np.reshape(best_solution[(input_size+1)*h_layer:(input_size+1)*h_layer+h_layer*output_size], (h_layer, output_size))
            ann.biases2 = np.reshape(best_solution[(input_size+1)*h_layer+h_layer*output_size:], output_size)
            y_pred = np.argmax(np.round(ann.forward(X_test)), axis=1)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f'Test accuracy for the ANN = {accuracy}')

            # Plot the accuracy history RUN
            plt.plot(accuracy_history_run)
            plt.axvspan(-0.5, len(accuracy_history_run)-1+0.5, facecolor='grey', alpha=0.1)
            plt.xlabel('Generation')
            plt.ylabel('Accuracy')
            plt.title('Accuracy History')
            st.pyplot()

    if selected_algorithm == 'Ant Colony Optimization':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        count=0
        st.sidebar.header("ACO-ANN Control Parameters")
        count+=1
        num_ants = st.sidebar.slider("Number of Ants", min_value=10, max_value=100, value=30,key = count)
        count+=1
        num_generations = st.sidebar.slider("Number of Generations", min_value=10, max_value=100, value=50,key = count)
        count+=1
        alpha = st.sidebar.slider("Alpha", min_value=-2.0, max_value=0.0, value=-0.5,key = count)
        count+=1
        beta = st.sidebar.slider("Beta", min_value=0.0, max_value=2.0, value=0.5,key = count)
        count+=1
        rho = st.sidebar.slider("rho VAlue", min_value=0.0, max_value=1.0, value=0.5,key = count)
        count+=1
        q = st.sidebar.slider("q value", min_value=0.0, max_value=1.0, value=0.1, key=count)
        count+=1
        h_layer = st.sidebar.slider("Hidden Layer Size", min_value=1, max_value=10, value=5, key=count)
        count+=1
        output_layer_size = 1

        if st.sidebar.button('Perform ACO Algorithm for ANN'):
            data = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
            x = data.drop(['Personal Loan','ID'],axis=1).values
            y = data['Personal Loan'].values
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=9)
            SC = StandardScaler()
            x_train = SC.fit_transform(x_train)
            x_test = SC.fit_transform(x_test)
            #Creating an instance for ACO-ANN Class
            aconn = ACONN(input_layer_size=x_train.shape[1], hidden_layer_size=h_layer, output_layer_size=1)
            #Training the Networks
            with st.spinner("Performing ACO for ANN Weights Optimisation"):
                aconn.train(x_train, y_train, max_iterations=num_generations, num_ants=num_ants, alpha=alpha, beta=beta, rho=rho, q=q)
            #Predicting the Values
            y_pred = aconn.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f'Accuracy:{accuracy}')


if __name__ == '__main__':
    app()