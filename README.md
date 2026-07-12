# NewsChecker

NewsChecker is an AI-powered fake news detection web application that analyzes online news articles using natural language processing and machine learning to predict whether an article is likely to be real or fake.

## Project Overview

### What is NewsChecker?

NewsChecker is a machine learning-powered web application that analyzes the content of news articles and predicts whether they are likely to be real or fake. Instead of basing its prediction on specific keywords or manually selected characteristics, the application converts each article into semantic embeddings that capture its overall meaning. Because articles can express the same ideas using different words, this allows the trained machine learning model to make predictions based on the article's content rather than its wording. The model then uses these embeddings to predict whether the article is likely to be real or fake.

### Why I Built It

I built NewsChecker to learn how to create a complete machine learning application, from preparing training data to building a web application that people can use. Instead of only training a machine learning model, I wanted to build the entire system. This included preparing news articles for training by scraping, cleaning, and converting them into embeddings, as well as creating a user interface where people can submit article text and receive a prediction in real time.

### How It Works

NewsChecker consists of two primary workflows:

#### Training Pipeline

The machine learning model was trained using a Kaggle dataset that contained news article URLs and their corresponding labels (real or fake), rather than the article text itself. To prepare this data for training, I built a preprocessing pipeline that automatically visited each URL, scraped the article content, cleaned the extracted text, and converted each article into a semantic embedding. These embeddings, paired with their labels, formed the final training dataset used to train the machine learning model.

#### Application Workflow

Once the model was trained, the preprocessing pipeline was no longer needed during normal application use. Instead, users provide the text of a news article directly through the web interface. Because the application already receives the article text, it simply converts the article text into a semantic embedding using the same embedding model employed during training and passes that embedding to the trained classifier. The model then predicts whether the article is likely to be real or fake and returns the prediction along with a confidence score.

## Features

### Training Pipeline
- Processes a Kaggle dataset that contains links to news articles.
- Extracts the text from each news article.
- Cleans the extracted text by removing advertisements, navigation menus, and other content that is not part of the main article text.
- Converts each article into a text embedding for machine learning.
- Uses six parallel Python processes to process multiple articles simultaneously, reducing the time required to generate the processed dataset.
- Splits the processed dataset into training, validation, and testing datasets.
- Saves the processed datasets as .npy files for efficient model training.

### Application
- Accepts the full text of a news article as user input.
- Converts submitted article text into a semantic embedding using the same embedding model used to train the machine learning model.
- Uses the trained machine learning model to classify the submitted article based on its generated embedding.
- Displays the model's prediction and confidence score.

## Tech Stack

## System Architecture

## Machine Learning Pipeline

## Dataset

## Results

## Project Structure

## Installation

## Future Improvements