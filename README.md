# ML-Task
-Data Structure and Logic
# Simple Permutations Generator
This Python script generates all possible permutations of a list of unique integers entered by the user. It employs backtracking to efficiently find and list these arrangements.
## Purpose
This code serves the purpose of showcasing how to create permutations of a list of distinct integers. Permutations are various sequences of elements, and this script finds all conceivable sequences based on the input provided by the user.

## How It Works
1. **Permutation Function**: The `permute(nums)` function is the core of this script. It accepts a list of unique integers as input and uses a backtracking approach to compute all permutations of these integers.
2. **Backtracking Algorithm**: The backtracking algorithm systematically explores all potential orders of the input integers. It starts with the first integer, exchanges it with others to explore different arrangements, and then reverts to consider other alternatives.
3. **User Input**: The script requests input from the user, expecting a list of unique integers separated by spaces. It then transforms this input into a Python list for further processing.
4. **Main Function**: The `main()` function manages user input and invokes the `permute()` function to generate permutations. It also displays the resulting list of permutations.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/permutations-generator.git
   cd permutations-generator
   ```

2. Run the script:

   ```bash
   python permutations.py
   ```

   You'll be prompted to input a list of unique integers separated by spaces. After providing the input, the script will calculate and show all permutations.

3. Customize the code or incorporate it into your projects as needed.

## Requirements
- Python 3.x

## Note

- To obtain meaningful permutations, make sure to input a list of unique integers separated by spaces.
- This code concentrates on efficiently generating permutations and can serve as a valuable resource for understanding backtracking algorithms.

Feel free to use this code as a reference for working with permutations or as a starting point for more complex projects involving combinatorial algorithms. If you have any questions or suggestions, don't hesitate to reach out or create an issue in the repository. Happy coding!
############################################################################################################################################################################
-Deep Learning CNN
# STL-10 Image Classification with RESNET-50

This repository contains code for training and evaluating a deep neural network, RESNET-50, on the STL-10 image classification dataset. The code is written in PyTorch and is designed to be easy to understand and use.

## About the Code

### Purpose
The purpose of this code is to demonstrate how to train a convolutional neural network (RESNET-50) for image classification tasks using the STL-10 dataset. The code covers data preprocessing, model definition, training, and evaluation.

### Key Components
1. **Data Loading**: The code loads the STL-10 dataset, which consists of labeled images for training and testing.

2. **Data Preprocessing**: Images are resized to match the input size expected by RESNET-50, converted to PyTorch tensors, and normalized.

3. **Model Definition**: RESNET-50 is used as the deep learning model. The final fully connected layer is replaced with a new one for classification into 10 classes (as in STL-10).

4. **Training**: The model is trained using the Adam optimizer and the cross-entropy loss function. Training runs for a specified number of epochs, and the loss is printed after each epoch.

5. **Evaluation**: After training, the model is evaluated on the test dataset, and the accuracy of the model on the test data is reported.

6. **GPU Support**: The code checks for the availability of a GPU and uses it for training and evaluation if available; otherwise, it uses the CPU.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/stl10-resnet50.git
   cd stl10-resnet50
   ```

2. Install the required libraries :

   ```bash
   pip install torch torchvision numpy
   ```

3. Run the code:

   ```bash
   python train_and_evaluate.py
   ```

   This will train the RESNET-50 model on the STL-10 dataset and print the training loss and test accuracy.

4. Customize the code as needed for your own projects or datasets.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy

## Note

- You may need to adjust hyperparameters (e.g., learning rate, batch size, number of epochs) for your specific problem.
- This code uses a pre-trained RESNET-50 model from torchvision and fine-tunes it for STL-10. You can explore other pre-trained models and customize them according to your requirements.

Feel free to use this code as a starting point for your own image classification projects or as a learning resource for working with PyTorch. If you have any questions or suggestions, please feel free to reach out or create an issue in the repository. Happy coding!
############################################################################################################################################################################
-Deep Learning - YOLO

