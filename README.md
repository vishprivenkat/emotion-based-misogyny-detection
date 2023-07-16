# Emotion Based Misogyny Detection

Using multi-modal data from emotion speech vectors and text embeddings for an effective fusion mechanism to identify misogyny in YouTube videos 

## Modalities  

This project presents a detailed analysis of using just text and a fusion of text and speech emotion vectors on which works better, for identifying misogyny in videos/audios. We also present an exploration of different Machine Learning and Deep Learning approaches that have been used to answer this problem statement. 

## Repository Structure 

* **notebooks** - analysis of results, and exploration of different ML algorithms used for answering the problem statement along with extraction of videos from YouTube. 
* **scripts** - contain codes for preprocessing audio features and extracting emotion feature embeddings, as well as provide an insight into the deep learning techniques used for answering the problem statement. 


## Dataset Preparation 

* Videos from different platforms like YouTube, Twitter etc. were collected that were non-misogynistic as well as misogynistic in nature.
* Audio from the videos was extracted and preprocessed. Using a custom pre-trained Wav2Vec2 model from HuggingFace, speech-emotion vectors were synthesized.
* Video transcripts were extracted using the YouTube API and other audio transcription tools and manually verified by the contributors of this repository, and these are primarily used for the textual modality.
* The dataset associated with this project is private. 
