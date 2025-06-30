
# 📊 YouAnalyze – YouTube Comment Sentiment Analyzer

FeelTube is a machine learning-based web app that analyzes the sentiment of YouTube video comments using natural language processing (NLP). The tool classifies comments as **positive**, **neutral**, or **negative**, and provides visual insights to understand viewer engagement and mood.

---

## 🚀 Features

- 🔍 Extracts comments from any YouTube video via URL
- 🤖 Performs sentiment analysis using a trained ML model
- 📊 Visualizes sentiment distribution via charts and gauges
- ☁️ Generates word clouds for each sentiment category
- 📈 Highlights top commenters and common words
- 🖥️ Built with Python, Streamlit, and Plotly

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **Libraries:** 
  - `pytube` (YouTube API alternative)
  - `pandas`, `re`, `nltk`, `scikit-learn`
  - `plotly`, `matplotlib`, `wordcloud`

---

## 🔧 Installation

```bash
git clone https://github.com/yourusername/feeltube.git
cd feeltube
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📥 Input Format

- Paste any public YouTube video URL into the input box.
- The app will fetch the top 100 comments and perform sentiment analysis.

---

## 📊 Output

- Sentiment classification (`positive`, `neutral`, `negative`)
- Sentiment distribution (pie chart, bar chart, gauge)
- Word clouds
- Comment-level predictions in table view

---

## 🧠 Model Info

- Trained on YouTube comment datasets
- Uses vectorization (`TF-IDF`) + classifier (e.g., `Naive Bayes` or `Logistic Regression`)
- Preprocessing: tokenization, stopword removal, stemming

---

## 📂 Folder Structure

```
.
├── app.py
├── model/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
├── utils/
│   └── helpers.py
├── data/
│   └── example.csv
├── requirements.txt
└── README.md
```

---

## 💡 Future Improvements

- 🎯 Add support for comment timestamp-based sentiment trend
- 🧾 Export analysis as PDF/CSV
- 🌍 Support for multilingual comments
- 📽️ Analyze entire playlists or channels

---

## 🤝 Acknowledgements

- [NLTK](https://www.nltk.org/)
- [Pytube](https://pytube.io/en/latest/)
- [Plotly](https://plotly.com/python/)

---

## 📜 License

MIT License – feel free to use and modify for personal or commercial projects.

---

## 🙋‍♂️ Author

**Manan Sood**  
📧 [Email](mmanansood732004@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/mannan-sood-a38688253/) | [GitHub](https://github.com/Manan79)
