# ResumeRank

A Resume Ranked App that ranks resumes according to their relevance to a job description using Natural Language Processing (NLP) techniques.

## Features

1 Text Extraction: Extracts content from PDF resumes.

2 Preprocessing: Processes text by:
- Tokenizing (splitting text into individual words/tokens).
- Removing stopwords (common words like "and," "the").
- Lemmatizing (converting words to their base forms, e.g., "running" → "run").
- Handling special cases like programming languages (e.g., C++, Python).

3 Vectorization: Uses CountVectorizer to convert text into numerical form for analysis.

4 Ranking: Computes cosine similarity to determine how closely resumes match the job description.

## Project Structure

- `main.py`: Contains the core logic for text extraction, preprocessing, vectorization, and similarity computation.
- `app.py` : Streamlit App
- `requirements.txt`: Lists all the dependencies required for the project.
- `README.md`: Project documentation.
