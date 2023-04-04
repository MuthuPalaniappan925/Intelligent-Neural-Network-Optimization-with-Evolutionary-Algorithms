
# Intelligent-Neural-Network-Optimization-with-Evolutionary-Algorithms


Intelligent Neural Network Optimization with Evolutionary Algorithms is a project aimed at exploring the use of evolutionary algorithms to optimize the performance of neural networks (Conf of weights). Neural networks are powerful machine learning models that have shown great success in a variety of domains, including image and speech recognition, natural language processing, and more. However, designing and training neural networks can be a challenging task, particularly when dealing with large, complex models.

The aim of this project is to investigate how evolutionary algorithms can be used to optimize neural network layers weights and hyperparameters. This involves developing a framework that integrates neural network with evolutionary algorithms, such as genetic algorithms or particle swarm optimization. The framework will be used to evaluate the performance of different optimization strategies on a range of Bank loan classification dataset.


# Artifical Neural Network (ANN) Architecture

The white box neural network architecture for the bank loan dataset is a supervised learning model that utilizes 12 input nodes in the input layer to receive information about the loan applicants, such as age, income, credit score, and other relevant factors.

 The input layer feeds into a hidden layer with 8 nodes, which performs complex computations and transforms the inputs to create useful features for the output layer.The output layer consists of a single node that predicts the likelihood of a loan application being approved or denied based on the input data.

```
Input Layer - 12 Nodes.
Hidden Layer - 8 Nodes.
Output Layer - 1 Node.
```
```
Activation Functions

- Hidden Layer: ReLU (Rectified Linear Unit) activation function.
- Output Layer: Sigmoid activation function (binary classification problems).
```
# Evolutionary Algorithms

Evolutionary algorithms are a class of optimization algorithms that are inspired by biological evolution. They work by maintaining a population of candidate solutions and iteratively applying selection, mutation, and crossover operations to produce new generations of solutions. These algorithms have been used successfully in a variety of optimization problems, including those in machine learning.

The List of Evolutionary algorithms used in this project

- [Ant Colony Optimization](https://github.com/MuthuPalaniappan925/Intelligent-Neural-Network-Optimization-with-Evolutionary-Algorithms/tree/main/Ant%20Colony%20Optimization%20with%20ANN)
- [Cultural Algorithm](https://github.com/MuthuPalaniappan925/Intelligent-Neural-Network-Optimization-with-Evolutionary-Algorithms/tree/main/Cultural%20Algorithm%20with%20ANN)
- [Genetic Algorithm](https://github.com/MuthuPalaniappan925/Intelligent-Neural-Network-Optimization-with-Evolutionary-Algorithms/tree/main/Gentic%20Algorithm%20With%20ANN)
- [Particle Swarm Optimization](https://github.com/MuthuPalaniappan925/Intelligent-Neural-Network-Optimization-with-Evolutionary-Algorithms/tree/main/Particle%20Swarm%20Optimization)

## Run Locally

Clone the project

```bash
gh repo clone MuthuPalaniappan925/Intelligent-Neural-Network-Optimization-with-Evolutionary-Algorithms
```

Go to the project directory

```bash
  cd Intelligent-Neural-Network-Optimization-with-Evolutionary-Algorithms
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the Streamlit server

```bash
  streamlit run evolutionary_algorithm.py
```
