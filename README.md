# Audio-Visual Speech Separation Using Deep Learning

This repository is the implementation for my thesis, **"Audio-Visual Speech Separation Using Deep Learning."** Welcome!

The goal of this project is to solve the "cocktail party problem"—how can we isolate a single person's voice from a noisy environment with multiple speakers? Our approach is to use both the audio from the microphone and the visual cues from the speaker's face to cleanly **separate** their speech.

The project is structured into four main stages, which you can follow in order to replicate the research.

## Step 1: Building the Visual Frontend

Before we can separate speech, our model needs to learn how to understand the visual information in a video. Processing raw video frames is very slow and expensive. So, our first step is to build a smart **visual frontend** that extracts a small but powerful set of features from the speaker's face.

This stage is broken into two parts:

#### Analyzing the Lip Region with PCA

To know what someone is saying, you need to look at their lips. But just the outer shape isn't enough; we need to know what's happening inside the mouth (like whether the teeth are visible).

The `1_frontend/1_1_pca/` directory handles this by:
1.  Carefully cropping the region inside the speaker's mouth from each video frame.
2.  Training a **Principal Component Analysis (PCA)** model to compress this image into a tiny, manageable feature vector. This gives us all the rich detail from inside the mouth without the heavy computational cost.

#### Training the Frontend Networks

Now, we train the two key networks that make up our frontend.

You can find the training notebooks in `1_frontend/1_2_frontend_training/`:
1.  **The Lip2Phone Network:** This network learns the **dynamics** of speech—the way a person's lips move as they talk. To do this, we use the audio to figure out the exact phonemes being spoken (using a tool called "Allosaurus"). The Lip2Phone network then learns to predict these same phonemes by *only* looking at the visual features of the lips. It learns to "hear with its eyes."
2.  **The FaceToSpeaker Network:** This network learns the **static** features of a person—the unique structure of their face that is linked to the sound of their voice. It learns to connect a person's face to their unique vocal signature (like pitch and tone).

The final output of this frontend is a single, powerful feature vector that describes both the lip movements and the speaker's identity.

## Step 2: Training the Main Speech Separation Model

Now that our frontend is ready, it's time to train the core of our system: the **Audio-Visual Speech Separation (AVSS)** model. This is handled in the `2_AVSS_main/` directory.

In this stage, we **freeze** the visual frontend we just trained. We then feed the main model:
* A mixed audio signal (the "cocktail party").
* The corresponding visual features from our frontend.

The model's job is to use the visual cues to figure out which voice belongs to the speaker on screen and cleanly separate it from the mix.

## Step 3: Fine-Tuning the Entire System

Now for better separation we do a fine tuning. In the `3_fine_tuning/` directory, we **unfreeze** all parts of the model and train the entire system together.

This **fine-tuning** step allows the visual and audio parts of the network to adjust to each other. The visual frontend learns to produce features that are more helpful for the specific task of speech separation. It's the final polish that makes the model work better.

## Step 4: Making the Model Robust with Curriculum Learning

Real-world environments are messy. To prepare our model for this, we use an new strategy called **Curriculum Learning**, which you can find in the `4_curriculum_learning/` directory.

Instead of just throwing random data at the model, we create a "curriculum" of progressively harder tasks:
1.  **The Basics:** The model starts with simple audio mixes (e.g., just two speakers and low noise).
2.  **Increasing Difficulty:** As it learns, we gradually give it more challenging problems (more speakers, louder background noise).
3.  **The Final Exam:** By the end, the model is tackling incredibly difficult scenarios with up to six speakers in a noisy room.

This "easy-to-hard" approach is how humans learn, and it makes our model far more robust and effective in unpredictable, real-world situations.
