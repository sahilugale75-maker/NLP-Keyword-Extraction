
import streamlit as st
import pandas as pd
import nltk
import re
import io
import matplotlib.pyplot as plt

from collections import Counter
from pypdf import PdfReader
from docx import Document
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="NLP Keyword Extraction System",
    page_icon="🔍",
    layout="wide"
)

# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.main-title {
    font-size: 34px;
    font-weight: bold;
    color: #2563eb;
}
.subtitle {
    font-size: 16px;
    color: #64748b;
}
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}
div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# NLTK STOPWORDS
# ----------------------------------

@st.cache_resource
def load_stopwords():
    try:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))
    except Exception:
        return {
            "the", "is", "are", "was", "were", "a",
            "an", "and", "or", "of", "to", "in",
            "on", "for", "with", "this", "that",
            "it", "as", "by", "from", "be", "has",
            "have", "had", "at", "which", "we",
            "they", "you", "he", "she", "their"
        }

STOP_WORDS = load_stopwords()

# ----------------------------------
# TEXT EXTRACTION FROM FILES
# ----------------------------------

def read_uploaded_file(uploaded_file):

    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.getvalue().decode(
            "utf-8-sig", errors="replace"
        )

    elif filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    elif filename.endswith(".docx"):
        document = Document(uploaded_file)

        paragraphs = [
            p.text for p in document.paragraphs
            if p.text.strip()
        ]

        return "\n".join(paragraphs)

    return ""

# ----------------------------------
# TEXT PREPROCESSING
# ----------------------------------

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# ----------------------------------
# KEYWORD EXTRACTION USING TF-IDF
# ----------------------------------

def extract_keywords(text, top_n):

    clean_text = preprocess_text(text)

    if len(clean_text.split()) < 3:
        return pd.DataFrame()

    # Split the document into sentence-level units.
    sentences = re.split(
        r"[.!?;\n]+",
        clean_text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.split()) >= 2
    ]

    if not sentences:
        return pd.DataFrame()

    # Fit TF-IDF across sentences to obtain
    # more useful inverse document frequency values.
    vectorizer = TfidfVectorizer(
        stop_words=list(STOP_WORDS),
        ngram_range=(1, 3),
        max_features=300,
        sublinear_tf=True
    )

    try:
        matrix = vectorizer.fit_transform(sentences)

        terms = vectorizer.get_feature_names_out()

        # Average TF-IDF scores across sentences.
        scores = matrix.mean(axis=0).A1

        results = pd.DataFrame({
            "Keyword / Keyphrase": terms,
            "Relevance Score": scores
        })

        results = results[
            results["Relevance Score"] > 0
        ]

        results = results.sort_values(
            by="Relevance Score",
            ascending=False
        )

        return results.head(top_n).reset_index(drop=True)

    except ValueError:
        return pd.DataFrame()

# ----------------------------------
# WORD CLOUD
# ----------------------------------

def create_wordcloud(results):

    frequencies = dict(zip(
        results["Keyword / Keyphrase"],
        results["Relevance Score"]
    ))

    if not frequencies:
        return None

    wc = WordCloud(
        width=1000,
        height=450,
        background_color="white",
        colormap="viridis"
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    return fig

# ----------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------

st.sidebar.title("🔍 NLP Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Keyword Extraction",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Automatic Keyword and Keyphrase "
    "Extraction using TF-IDF"
)

# ----------------------------------
# HOME PAGE
# ----------------------------------

if page == "Home":

    st.markdown(
        '<p class="main-title">'
        'NLP-Based Automatic Keyword Extraction'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">'
        'An intelligent system for extracting '
        'important keywords and keyphrases '
        'from text documents.'
        '</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Welcome to the NLP Keyword Extraction System")

    st.write("""
    This application uses Natural Language Processing
    and TF-IDF techniques to identify important words
    and phrases from documents.

    Upload your document or enter text manually
    to discover its important topics.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Extraction Method", "TF-IDF")
    col2.metric("Phrase Range", "1–3 words")
    col3.metric("File Formats", "TXT, PDF, DOCX")

    st.info(
        "Go to Keyword Extraction from the sidebar "
        "to start using the application."
    )

# ----------------------------------
# KEYWORD EXTRACTION PAGE
# ----------------------------------

elif page == "Keyword Extraction":

    st.title("📄 Keyword and Keyphrase Extraction")

    st.write(
        "Enter your text or upload a document "
        "to extract important keywords."
    )

    st.subheader("1. Enter Text")

    user_text = st.text_area(
        "Paste your document text here:",
        height=220,
        placeholder="Enter your paragraph or document..."
    )

    st.markdown("### OR")

    st.subheader("2. Upload a Document")

    uploaded_file = st.file_uploader(
        "Upload TXT, PDF or DOCX file",
        type=["txt", "pdf", "docx"]
    )

    top_n = st.slider(
        "Number of keywords to extract",
        min_value=5,
        max_value=30,
        value=15
    )

    if st.button(
        "🔍 Extract Keywords",
        use_container_width=True
    ):

        text = user_text

        if uploaded_file is not None:

            try:
                text = read_uploaded_file(uploaded_file)

                if not text.strip():
                    st.error(
                        "No readable text found in this document."
                    )
                    st.stop()

            except Exception as e:
                st.error(f"File reading error: {e}")
                st.stop()

        if not text.strip():

            st.warning(
                "Please enter text or upload a document."
            )

        else:

            with st.spinner(
                "Processing document and extracting keywords..."
            ):

                results = extract_keywords(
                    text,
                    top_n
                )

            if results.empty:

                st.warning(
                    "No meaningful keywords found. "
                    "Please enter more text."
                )

            else:

                st.session_state["results"] = results
                st.session_state["source_text"] = text

    # Display stored results
    if "results" in st.session_state:

        results = st.session_state["results"]

        st.success(
            "Keywords extracted successfully!"
        )

        st.divider()

        st.subheader("3. Extracted Keywords")

        display_results = results.copy()
        display_results.index = display_results.index + 1
        display_results.index.name = "Rank"

        st.dataframe(
            display_results,
            use_container_width=True
        )

        # ----------------------------------
        # BAR CHART
        # ----------------------------------

        st.subheader("4. Keyword Relevance Chart")

        chart_data = results.set_index(
            "Keyword / Keyphrase"
        )["Relevance Score"]

        st.bar_chart(
            chart_data,
            horizontal=True
        )

        # ----------------------------------
        # WORD CLOUD
        # ----------------------------------

        st.subheader("5. Keyword Word Cloud")

        fig = create_wordcloud(results)

        if fig is not None:

            st.pyplot(fig)
            plt.close(fig)

        # ----------------------------------
        # DOWNLOAD CSV
        # ----------------------------------

        st.subheader("6. Download Results")

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Keywords as CSV",
            data=csv_data,
            file_name="extracted_keywords.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.download_button(
            label="⬇ Download Keywords as TXT",
            data=results.to_string(index=False),
            file_name="extracted_keywords.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.caption(
            "TF-IDF scores indicate relative term importance "
            "within the processed document."
        )

# ----------------------------------
# ABOUT PROJECT PAGE
# ----------------------------------

elif page == "About Project":

    st.title("About Our Project")

    st.subheader(
        "NLP-Based Automatic Keyword and Keyphrase "
        "Extraction from Text Documents Using TF-IDF"
    )

    st.write("""
    The objective of this project is to develop
    an automated system that identifies important
    keywords and keyphrases from text documents.

    The system uses Natural Language Processing
    techniques and TF-IDF to calculate term
    importance and rank the extracted terms.
    """)

    st.subheader("Technologies Used")

    technologies = [
        "Python",
        "Natural Language Toolkit (NLTK)",
        "Scikit-learn",
        "TF-IDF",
        "Streamlit",
        "Pandas",
        "Matplotlib",
        "WordCloud"
    ]

    for technology in technologies:
        st.write("✔ " + technology)

    st.subheader("Project Applications")

    st.write("""
    - Academic notes and study materials
    - Research paper analysis
    - News article analysis
    - Document indexing
    - Information retrieval
    """)

    st.subheader("Project Developed By")

    st.write("Department of Artificial Intelligence and Data Science")
    st.write("Academic Year: 2026–2027")

# ----------------------------------
# FOOTER
# ----------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "NLP Mini Project | AI & DS | 2026–2027"
)
