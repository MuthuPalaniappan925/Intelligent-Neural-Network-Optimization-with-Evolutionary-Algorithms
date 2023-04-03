##Importing Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
plt.style.use('dark_background')

##Defining the ACO_NN class
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
        plt.show()
                    
    def train(self, X, y, max_iterations, num_ants, alpha, beta, rho, q):
        self.ant_colony_optimization(X, y, max_iterations, num_ants, alpha, beta, rho, q)

    def predict(self, X):
        y_hat = self.forward(X)
        predictions = np.round(y_hat)
        return predictions
    
##Loading the dataset
data = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
x = data.drop(['Personal Loan','ID'],axis=1).values
y = data['Personal Loan'].values

##Train Test Split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=9)

##Data Pre-Processing
SC = StandardScaler()
x_train = SC.fit_transform(x_train)
x_test = SC.fit_transform(x_test)

##Creating the Instance For ACO_NN class
aconn = ACONN(input_layer_size=x_train.shape[1], hidden_layer_size=4, output_layer_size=1)

## Training the network
aconn.train(x_train, y_train, max_iterations=10, num_ants=10, alpha=1, beta=1, rho=0.5, q=1)

## Predicting the Values
y_pred = aconn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)