# Roman Numeral Classifier (I–X)

This Flask-based web application, & has a pyqt app also, allows users to upload or draw images of Roman 
numerals ranging from I to X. The system classifies the input using a convolutional neural network (CNN) model. 
To improve accuracy, a binary subnet classifier is used to resolve frequent confusion between similar numerals.

---

## Overview

### Main Model:

A convolutional neural network trained to classify handwritten Roman numerals from I to X.

- **Input:** 28×28 grayscale images (centered and normalized)
- **Output:** 10-class softmax prediction:  
  `I, II, III, IV, V, VI, VII, VIII, IX, X`
- **Architecture:** Sequential CNN with Conv2D, MaxPooling, Dense layers
- **Loss Function:** Categorical crossentropy
- **Optimizer:** Adam

**Known Issue:**  
The model frequently misclassifies 'II' as 'V'.

---

### Subnet Model:

A binary CNN model trained specifically to distinguish between 'II' and 'V'.

- **Trigger Condition:** Activated only when the main model predicts 'V'
- **Output:**  
  - 1 → 'II'  
  - 0 → 'V'
- **Architecture:** Shallow CNN with Conv2D, MaxPooling, Dropout, Dense
- **Loss Function:** Binary crossentropy
- **Optimizer:** Adam

This auxiliary model enhances prediction accuracy for visually ambiguous cases.

---

## Preprocessing Pipeline

- Input is captured via file upload or canvas (base64)
- Converted to grayscale using PIL
- Cropped to remove surrounding white space
- Resized to 28×28 pixels
- Pixel values normalized to the range [0, 1]
- Optional data augmentation for the subnet (e.g., rotation, shear)

---

## Training Summary

### Main Model
- Trained using a combination of real and synthetic handwritten data
- Augmentations included:
  - Rotation
  - Width and height shifting
  - Zoom

### Subnet Model
- Trained on targeted examples of 'II' and 'V'
- Used to refine predictions when the main model is uncertain
- Augmentations included:
  - Rotation
  - Shear
  - Translation

---

## Deployment

The application runs as a Flask web server, where users can:

- Upload an image of a Roman numeral
- Draw directly on a canvas
- View the predicted numeral with visual feedback

---

## Future Improvements

- Expand classification range beyond 'X'
- Improve augmentation strategies
- Explore ensemble methods for better confidence estimation
