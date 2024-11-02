# Homework 3: Video Classification with 3D CNN

Welcome to the third homework assignment, where we'll tackle the task of video classification using 3D Convolutional Neural Networks (CNNs).

## Data and Labels

To access the data for this assignment, please download it from the following Kaggle link: [link](https://www.kaggle.com/competitions/hkustgz-aiaa-2205-hw-3-fall-2024/data).

## Dataset Structure

You will find a video folder named hw3_16fpv. Each subfolder contains 16 frames extracted from a single video. These frames are used to construct a 5D Tensor with dimensions (N C D H W). Additionally, the original MP4 videos are provided.

## Baseline Model

We have provided a baseline model using ResNet18-3D.

Navigate to the hw3 directory and execute the following command to train the ResNet18-3D model:

```
$ python train.py
```

To perform inference on the test set, run:

```
$ python test2csv.py
```


## Submitting to Kaggle

After obtaining your test outputs, submit them to the leaderboard via the following URL:

```
https://www.kaggle.com/competitions/hkustgz-aiaa-2205-hw-3-fall-2024/
```

The evaluation metric for this competition is accuracy. Please refer to test_for_student.csv for the correct submission format.


## Enhancing Model Performance

Now, let's dive into the exciting part—improving your model's performance. Here are some suggestions to get you started:

- Frame Extraction: Experiment with extracting different frames (other than 16) from each video using ffmpeg.
- Neural Network Architecture: Explore alternative neural network designs to see if they yield better results.
- Data Augmentation: Implement various data augmentation strategies to enhance the dataset's diversity.

Good luck, and happy experimenting! 😄 🎉
