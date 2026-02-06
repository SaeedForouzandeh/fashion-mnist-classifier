# FashionMNIST Classification with PyTorch

## Overview
A convolutional neural network (CNN) implementation for classifying FashionMNIST images using PyTorch. This project demonstrates image classification with deep learning, achieving over 90% accuracy on the test set.

## Features
- Complete CNN implementation from scratch
- Training and evaluation pipeline
- Visualizations of predictions and misclassifications
- Class-wise accuracy analysis
- GPU support (if available)

## Dataset
FashionMNIST contains 70,000 grayscale images (28x28 pixels) of 10 clothing categories:
- 60,000 training images
- 10,000 test images

### Classes
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Model Architecture
```
CNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout): Dropout(p=0.25, inplace=False)
  (fc1): Linear(in_features=3136, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  (relu): ReLU()
)
```

## Requirements
- Python 3.7+
- PyTorch 1.9+
- torchvision
- matplotlib
- numpy

## Installation
```bash
# Clone the repository
git clone https://github.com/SaeedForouzandeh/fashion-mnist-classifier.git
cd fashion-mnist-classifier

# Install dependencies
pip install torch torchvision matplotlib numpy
```

## Usage
```python
# Run the main script
python fashion_mnist_classifier.py
```

The script will:
1. Download the FashionMNIST dataset
2. Train the CNN model for 10 epochs
3. Evaluate on the test set
4. Display prediction results
5. Show misclassified examples
6. Print class-wise accuracy

## Expected Results
- Training loss decreases from ~0.5 to ~0.2
- Test accuracy: 90-92%
- Best performing classes: Trouser, Sandal, Bag
- Most challenging classes: Shirt, Coat, Pullover

## Output Examples
After running, you'll see:
- 32 sample predictions (green=correct, red=incorrect)
- 10 misclassified examples
- Accuracy breakdown per class

## File Structure
```
fashion-mnist-classifier/
│
├── fashion_mnist_classifier.py    # Main implementation
├── README.md                      # This file
├── requirements.txt               # Dependencies
├── data/                          # Downloaded dataset (auto-created)
└── images/                        # Generated plots (auto-created)
```

## Customization
You can modify:
- `batch_size` in DataLoader
- Number of epochs
- Learning rate
- CNN architecture
- Dropout rate

## License
MIT License

## Author
Saeed Forouzandeh
- GitHub: [@SaeedForouzandeh](https://github.com/SaeedForouzandeh)

## Acknowledgments
- PyTorch team for the excellent framework
- FashionMNIST dataset creators
- Open source community

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

---
If you find this project useful, please give it a ⭐ on GitHub!
