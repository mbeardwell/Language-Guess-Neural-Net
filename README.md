# Language Recognition Using Neural Networks

An application that uses a **feedforward neural network** to guess the language of an input word without any prior knowledge of what a language is. This project was developed throughout 2017 and submitted as part of my **EPQ Qualification**, for which I received an **A\***.

## Project Paper
For a detailed explanation of the methodology, implementation, and results, read the full project paper here:  
[4_Project_Outcome.pdf](./4_Project_Outcome.pdf)

## Overview
- **Neural Network Type:** Feedforward network with backpropagation
- **Languages Tested:** English, French, German, Latin, Italian, Greek
- **Accuracy Achieved:** **69.3%** (compared to a 16.7% random baseline)
- **Training Iterations:** 20 per network configuration
- **Optimisation Factors:** Number of hidden layers and neurons per layer

## Installation & Usage
### **Requirements**
- Python **3.0 or later**
- NumPy

### **Running the Program**
1. Install dependencies:
```bash
python3 -m pip install numpy
```

3. Clone the repository:
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

*   **Increase training iterations** to push accuracy beyond 70%.
*   **Experiment with different activation functions** (e.g., ReLU).
*   **Incorporate additional languages** to assess scalability.
*   **Explore convolutional networks (CNNs)** to better capture word patterns.

* * *
