# Language Recognition Using Neural Networks

This project implements a **feedforward neural network from scratch** to classify words into different languages **without using pre-built machine learning libraries**. It was developed as part of an Extended Project Qualification (EPQ) during A-levels in 2017, for which I received an A\*.

While originally an academic project, this implementation remains a pure, low-level demonstration of neural networks, covering forward propagation, backpropagation, weight optimisation, and training strategies—all coded manually without TensorFlow or PyTorch.

## Why This Project Stands Out
- No ML frameworks used – The neural network, including backpropagation, was coded from scratch using just Python & NumPy.
- Hyperparameter tuning – Investigated effects of hidden layers, neurons per layer, and learning rate on accuracy.
- Real experimentation – Evaluated model performance over 20 iterations per configuration, testing six languages.
- Overcoming challenges – Switched focus when early attempts at Hangman difficulty prediction failed, demonstrating adaptability.
- Relevance today – Though created in 2017, the project showcases core ML principles still used in modern AI.

## Project Paper
For a detailed explanation of the methodology, implementation, and results, read the full project paper [here](./docs/Neural_Network_Language_Classifier.pdf).

## Results
- Best accuracy: 69.3% (compared to 16.7% random baseline).
- Languages tested: English, French, German, Latin, Italian, Greek.
- Tuned network structures: 1-3 hidden layers, 40-200 neurons per layer.
- Key findings: More hidden layers increased overfitting, while a 200-neuron, 1-hidden-layer model performed best.

## Installation & Usage
### Requirements
- Python 3.0 or later
- NumPy

### Running the Program
1. Install dependencies:
```bash
python3 -m pip install numpy
```

2. Clone the repository:
```bash
git clone https://github.com/mbeardwell/language-guesser.git
cd language-guesser
```

3.  Run the artefact:
```bash
python3 artefact.py
```

Further Improvements
-----------------------

*   Increase training iterations to push accuracy beyond 70%.
*   Experiment with different activation functions (e.g., ReLU).
*   Incorporate additional languages to assess scalability.
*   Explore convolutional networks (CNNs) to better capture word patterns.

* * *
