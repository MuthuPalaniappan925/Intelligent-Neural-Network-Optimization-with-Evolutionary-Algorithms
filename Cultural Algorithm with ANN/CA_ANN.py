import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
old_settings = np.seterr(all='ignore')
plt.style.use('dark_background')

# Define the ANN class
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

# Define the fitness function to get the accuracy
def fitness_function(solution):
    ann.weights1 = np.reshape(solution[:input_size*hidden_size], (input_size, hidden_size))
    ann.biases1 = np.reshape(solution[input_size*hidden_size:(input_size+1)*hidden_size], hidden_size)
    ann.weights2 = np.reshape(solution[(input_size+1)*hidden_size:(input_size+1)*hidden_size+hidden_size*output_size], (hidden_size, output_size))
    ann.biases2 = np.reshape(solution[(input_size+1)*hidden_size+hidden_size*output_size:], output_size)
    y_pred = np.argmax(np.round(ann.forward(X_train)), axis=1)
    accuracy = accuracy_score(y_train, y_pred)
    return accuracy

# Defining the Cultural Algorithm parameters
pop_size = 50
num_generations = 100
p_accept = 0.5
step_size = 0.1

# Loading the dataset
data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
X = data.drop(['Personal Loan','ID'],axis=1).values
y = data['Personal Loan'].values
input_size = X.shape[1]
output_size = len(np.unique(y))

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the ANN and the population
hidden_size = 10
ann = ANN_Model(input_size, hidden_size, output_size)
num_weights = input_size*hidden_size + hidden_size + hidden_size*output_size + output_size
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

    # Print the best fitness score in the current generation
    print(f'Generation {t+1}: Best fitness = {best_fitness}')

    # Append the accuracy of the best solution to the accuracy history
    accuracy_history_run.append(best_fitness)

# Evaluate the accuracy of the best solution on the test set
ann.weights1 = np.reshape(best_solution[:input_size*hidden_size], (input_size, hidden_size))
ann.biases1 = np.reshape(best_solution[input_size*hidden_size:(input_size+1)*hidden_size], hidden_size)
ann.weights2 = np.reshape(best_solution[(input_size+1)*hidden_size:(input_size+1)*hidden_size+hidden_size*output_size], (hidden_size, output_size))
ann.biases2 = np.reshape(best_solution[(input_size+1)*hidden_size+hidden_size*output_size:], output_size)
y_pred = np.argmax(np.round(ann.forward(X_test)), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy for the ANN = {accuracy}')

# Plot the accuracy history RUN
plt.plot(accuracy_history_run)
plt.axvspan(-0.5, len(accuracy_history_run)-1+0.5, facecolor='grey', alpha=0.1)
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.show()