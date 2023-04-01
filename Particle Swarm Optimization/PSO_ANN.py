##Importing Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns

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

        #Getting the global and local best positions
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
        fig.show()
        
##Creating and instances of PSOANN Class
psoann = PSOANN(input_layer_size=x_train.shape[1],hidden_layer_size=5,output_layer_size=1,particle_size=10,max_iterations=30,c_1=1.9,c_2=1.2,inertial_weight=0.7)
##Training
psoann.train()
##Particle Distribution in 3-D
psoann.plot()
##Prediction
y_pred = psoann.predict(x_test)
##Performance Metrics
accuracy_score = accuracy_score(y_pred,y_test)
print(f"Accuracy Score: {accuracy_score}")
print(classification_report(y_test,y_pred))
##Plotting Confusion MAtrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Confusion Matrix", fontsize=16)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False, ax=ax)
ax.set_xlabel("Predicted Labels", fontsize=14)
ax.set_ylabel("True Labels", fontsize=14)
fig.show()