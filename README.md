<div style="text-align:center;">
    <h1>All About Einstein: An LLM-Powered Exploration of Albert Einstein</h1>
</div>

<div style="text-align:center;">
    <image src="docs/demo_vid.gif" controls title="Title"></image>
</div>

## Table of Contents
- [About the Project](#about-the-project)
- [Setup](#setup)
- [Usage](#usage)
- [Shoutout!](#shoutout)
- [Know more about me!](#know-more-about-me)

# About the Project
All About Einstein is an innovative question-answering application focused on Albert Einstein. Utilizing the knowledge base extracted from the [Britannica Albert Einstein encyclopedia](https://www.britannica.com/biography/Albert-Einstein), this project employs [instructor text embeddings](https://huggingface.co/hkunlp/instructor-xl) and the [FLAN-T5 LLM model](https://huggingface.co/google/flan-t5-xxl) from Hugging Face. The user-friendly interface, designed using [Streamlit](https://streamlit.io/), and the application's pipeline is constructed using the [Langchain](https://python.langchain.com/docs/get_started/introduction.html) framework, enabling users to pose a wide range of questions related to Albert Einstein's life and work.

<div style="text-align:center;">
    <image src="docs/how it work.jpg" controls title="How it works"></image>
</div>

# Setup
1. install the dependecies needed\
   ```pip install -r requirements.txt```
2. Add yout huggingface API key to the `.env` file in your directory \
   ```HUGGINGFACEHUB_API_TOKEN=your_secrit_api_key```

# Usage
1. Ensure that you have installed the required dependencies adn added the huggingface API key to the `.env` file
2. In the terminal, run this command:\
   ```streamlit run app.py```
3. The app will launch in you default web, displaying the UI
4. Ask a question about Albert Einstein!

# Shoutout!
Special thanks to Alejandro AO for being an invaluable source of knowledge and guidance in helping me learn more about LLM and its applications. Be sure to check out his informative content on his [YouTube channel](https://www.youtube.com/@alejandro_ao)

# Know more about me!
website: [mhamidasn.github.io](https://mhamidasn.github.io/)\
linkedin: [linkedin.com/in/mhamidasn](https://www.linkedin.com/in/mhamidasn/)\
github: [github.com/mhamidasn](https://github.com/mhamidasn)

<p align="center">
  <strong>🌌Here's to pushing boundaries, unraveling mysteries, and creating a future woven with the threads of innovation🌌</strong>
</p>