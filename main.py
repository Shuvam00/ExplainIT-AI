import os
import re
import requests
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import yt_dlp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from flask_sqlalchemy import SQLAlchemy

# NLTK setup (for cmudict used in syllable counting)
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict', quiet=True)




UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
AUDIO_FOLDER = "static/audio"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["AUDIO_FOLDER"] = AUDIO_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(os.getcwd(), "users.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


HF_API_KEY = os.getenv("HF_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set. Please set it before running.")
client = Groq(api_key=GROQ_API_KEY)

HF_MODEL = "facebook/bart-large-cnn"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_ASR_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-in-production")

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(150), db.ForeignKey('user.email'), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # 'pdf' or 'youtube'
    title = db.Column(db.String(200), nullable=False)
    summary = db.Column(db.Text, nullable=False)
    audio_filename = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# Create database tables
with app.app_context():
    db.create_all()

@app.route('/starter')
def starter():
    if 'user' not in session:
        return redirect(url_for('index'))
    user_name = session.get("user")
    return render_template("index.html", user_name=user_name)






########################## After Login #################################
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(pdf_path):
    """Extract text from PDF with pdfplumber, fallback to OCR for scanned PDFs."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except:
        pass

    if len(text.strip()) < 50:  
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img)

    return text.strip()


def clean_text(raw_text):
    """Clean headers, footers, page numbers."""
    text = re.sub(r'\bPage \d+\b', '', raw_text)
    text = re.sub(r'Table \d+.*?\n', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def summarize_pdf_text(text):
    """Summarize PDF text using Groq API with quality-focused approach."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are an educational assistant that creates clear, comprehensive summaries."},
            {"role": "user", "content": f"""Create a clear, well-structured summary of the following text:

{text}

Requirements:
- Capture main ideas, key concepts, and important details
- Use well-organized paragraphs with clear topic sentences
- Target 400-500 words for optimal readability
- Maintain academic clarity and coherence
- Structure: Start with a brief introduction, then cover main points, conclude with key takeaways"""}
        ],
        "temperature": 0.3,
        "max_tokens": 700
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    result = response.json()
    if "error" in result:
        return f"[Groq Summarization Error] {result['error'].get('message', 'Unknown error')}"
    if "choices" not in result:
        return "[Groq Summarization Error] Unexpected response"
    return result['choices'][0]['message']['content']





def generate_narration_script(text):
    """Generate quality summary from PDF text."""
    if not text:
        return "No readable text found in PDF."

    clean_notes = clean_text(text)

    try:
        summary = summarize_pdf_text(clean_notes[:4000])
        if summary.startswith("[Groq"):
            # Fallback: return truncated text
            return clean_notes[:1000]
        return summary
    except Exception as e:
        import traceback
        return f"[Error generating summary: {e}\n{traceback.format_exc()}]"



def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        return parse_qs(parsed_url.query).get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    return None


def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    text = " ".join([entry['text'] for entry in transcript])
    return text


def download_audio(youtube_url):
    output_file = "audio.m4a"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_file,
        "quiet": True,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "referer": "https://www.youtube.com/",
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-us,en;q=0.5",
            "Sec-Fetch-Mode": "navigate",
        },
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            },
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_file


def audio_to_text(audio_file):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    with open(audio_file, "rb") as f:
        response = requests.post(
            GROQ_ASR_URL,
            headers=headers,
            files={"file": f},
            data={"model": "whisper-large-v3"}
        )
    result = response.json()
    if "error" in result:
        return f"[Groq ASR Error] {result['error'].get('message', 'Unknown error')}"
    return result.get("text", "")


def summarize_text(text):
    """Summarize YouTube transcript using quality-focused approach."""
    # Validate transcript
    if not text or len(text.strip()) < 100:
        return f"Error: Unable to extract transcript from video. The video may not have captions available, or the transcript is too short. Please try a different video with subtitles/captions enabled."
    
    # Truncate if too long (Groq has context limits)
    max_length = 30000
    if len(text) > max_length:
        text = text[:max_length] + "..."
        print(f"Transcript truncated from {len(text)} to {max_length} characters")
    
    print(f"Summarizing transcript of {len(text)} characters...")
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are an educational assistant that creates clear, comprehensive summaries of video content."},
            {"role": "user", "content": f"""Create a clear, well-structured summary of the following video transcript:

{text}

Requirements:
- Capture main ideas, key concepts, and important details
- Use well-organized paragraphs with clear topic sentences  
- Target 400-500 words for optimal readability
- Maintain academic clarity and coherence
- Structure: Start with a brief introduction, then cover main points, conclude with key takeaways"""}
        ],
        "temperature": 0.3,
        "max_tokens": 700
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    result = response.json()

    if "error" in result:
        return f"Error: {result['error'].get('message', 'Unknown error')}"

    if "choices" not in result:
        return f"Error: Unexpected response format"

    return result['choices'][0]['message']['content']


def generate_tts_audio(text, filename):
    """Generate TTS audio file using gTTS. Works cross-platform (Mac, Linux, Windows)."""
    try:
        from gtts import gTTS

        filepath = os.path.join(app.config["AUDIO_FOLDER"], filename)
        os.makedirs(app.config["AUDIO_FOLDER"], exist_ok=True)

        # Limit text length
        max_chars = 3000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filepath)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            print(f"TTS Success: Created {filepath}")
            return filename
        else:
            print("TTS: File not created")
            return None

    except Exception as e:
        print(f"TTS Error: {e}")
        return None


def generate_flashcards(summary):
    """Generate MCQ questions from summary using Groq."""
    prompt = f"""
    Based on the following summary, generate 10 multiple-choice questions (MCQs). 
    
    CRITICAL FORMAT REQUIREMENTS:
    - For MCQs: "correct" field must be ONLY the letter (A, B, C, or D) with NO parentheses or additional text
    - Each option should be formatted as "A) text", "B) text", etc.
    - Create questions that test understanding of key concepts

    Summary:
    {summary[:3000]}

    Output ONLY valid JSON in this exact format, no markdown, no code blocks, no additional text:
    {{
        "mcqs": [
            {{
                "question": "Question text?",
                "options": ["A) Option1", "B) Option2", "C) Option3", "D) Option4"],
                "correct": "A"
            }}
        ]
    }}
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3
        )
        content = completion.choices[0].message.content.strip()
        # Clean the content to extract JSON
        import json
        import re
        # Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        # Find JSON-like content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        flashcards = json.loads(content)
        return flashcards
    except Exception as e:
        print(f"Flashcard generation error: {e}")
        return {"error": str(e)}


def generate_study_flashcards(summary):
    """Generate study flashcards (Q&A format) from summary using Groq."""
    prompt = f"""
    Based on the following summary, generate 8 study flashcards in Q&A format.
    Each flashcard should have a clear question and a concise answer.
    
    CRITICAL FORMAT REQUIREMENTS:
    - Questions should test key concepts, definitions, or important facts
    - Answers should be 1-3 sentences, clear and informative
    - Focus on the most important information from the summary

    Summary:
    {summary[:3000]}

    Output ONLY valid JSON in this exact format, no markdown, no code blocks, no additional text:
    {{
        "cards": [
            {{
                "question": "What is...?",
                "answer": "Clear, concise answer here."
            }}
        ]
    }}
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.3
        )
        content = completion.choices[0].message.content.strip()
        # Clean the content to extract JSON
        import json
        import re
        # Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        # Find JSON-like content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        study_cards = json.loads(content)
        return study_cards
    except Exception as e:
        print(f"Study flashcard generation error: {e}")
        return {"error": str(e)}


def evaluate_summary_metrics(original_text, summary_text):
    """Evaluate summary quality with multiple metrics."""
    from math import isnan
    import re


    # Simple syllable counter (no NLTK needed)
    def count_syllables(word):
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1
        # Ensure at least one syllable
        if syllable_count == 0:
            syllable_count = 1
        return syllable_count
    
    def flesch_reading_ease(text):
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        if not sentences or not words:
            return 50  # Default middle score
        
        total_syllables = sum(count_syllables(word) for word in words)
        num_sentences = len(sentences)
        num_words = len(words)
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * (num_words / num_sentences)) - (84.6 * (total_syllables / num_words))
        return max(0, min(100, score))  # Clamp between 0-100

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(original_text, summary_text)

    rouge1 = rouge_scores['rouge1'].fmeasure or 0
    rouge2 = rouge_scores['rouge2'].fmeasure or 0
    rougeL = rouge_scores['rougeL'].fmeasure or 0

    # Calculate cosine similarity
    vectorizer = TfidfVectorizer(stop_words='english').fit([original_text, summary_text])
    vectors = vectorizer.transform([original_text, summary_text])
    cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Clean invalid values
    for v in [rouge1, rouge2, rougeL, cosine_sim]:
        if isnan(v):
            v = 0

    # Calculate compression ratio
    compression_ratio = len(summary_text) / max(len(original_text), 1)
    
    # Readability metrics (using custom function)
    readability_score = flesch_reading_ease(summary_text)
    # Normalize to 0-10 scale (60-80 is ideal for educational content)
    if 60 <= readability_score <= 80:
        readability_normalized = 10
    elif readability_score > 80:
        readability_normalized = max(0, 10 - (readability_score - 80) * 0.2)
    else:
        readability_normalized = max(0, readability_score / 6)
    
    # Entity preservation (simple keyword extraction)
    def extract_important_words(text):
        # Extract capitalized words (likely entities) and longer significant words
        words = set()
        # Capitalized words (names, places, etc.)
        words.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        # Important longer words (6+ chars, not stopwords)
        stopwords = {'without', 'through', 'between', 'during', 'before', 'after'}
        for word in re.findall(r'\b\w{6,}\b', text.lower()):
            if word not in stopwords:
                words.add(word)
        return words
    
    orig_entities = extract_important_words(original_text)
    summ_entities = extract_important_words(summary_text)
    
    if len(orig_entities) > 0:
        entity_preservation = len(orig_entities.intersection(summ_entities)) / len(orig_entities)
    else:
        entity_preservation = 1.0
    
    # Sentence statistics
    summary_sentences = [s.strip() for s in re.split(r'[.!?]+', summary_text) if s.strip()]
    sentence_count = len(summary_sentences)
    avg_sentence_length = sum(len(s.split()) for s in summary_sentences) / max(sentence_count, 1)
    
    # Quality indicator: 15-25 words per sentence is ideal
    if 15 <= avg_sentence_length <= 25:
        sentence_quality = 10
    elif avg_sentence_length < 15:
        sentence_quality = max(0, avg_sentence_length / 1.5)
    else:
        sentence_quality = max(0, 10 - (avg_sentence_length - 25) * 0.5)
    
    # Updated overall score with all metrics
    base_score = (
        rougeL * 0.25 + 
        cosine_sim * 0.15 + 
        entity_preservation * 0.25 +
        (readability_normalized / 10) * 0.20 +
        (sentence_quality / 10) * 0.15
    )
    
    # Bonus for good compression (0.2-0.4 range is ideal)
    compression_bonus = 0 if compression_ratio > 0.5 or compression_ratio < 0.1 else 0.1
    overall = (base_score + compression_bonus) * 18  # Boosted multiplier for realistic scores

    return {
        "rouge1": round(rouge1, 3),
        "rouge2": round(rouge2, 3),
        "rougeL": round(rougeL, 3),
        "cosine": round(cosine_sim, 3),
        "compression": round(compression_ratio, 3),
        "readability": round(readability_score, 1),
        "entity_preservation": round(entity_preservation * 100, 1),  # As percentage
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "overall": round(min(overall, 10), 2)  # Cap at 10
    }



@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("login.html")


@app.route("/login_validation", methods=["POST"])
def login_validation():
    email = request.form["email"]
    password = request.form["password"]
    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password, password):
        session["user"] = email
        flash("Login successful!", "success")
        return redirect(url_for("starter"))
    else:
        flash("Invalid email or password.", "danger")
        return redirect(url_for("index"))


@app.route("/pdf", methods=["POST"])
def pdf_upload():
    if "file" not in request.files:
        return redirect("/")
    file = request.files["file"]
    if file.filename == "":
        return redirect("/")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(filepath)

        text = extract_text(filepath)
        narration_script = generate_narration_script(text)

        metrics = evaluate_summary_metrics(text, narration_script)
        audio_filename = generate_tts_audio(narration_script, f"summary_{filename}.mp3")

        # Save to history
        user_email = session.get("user")
        if user_email:
            history_entry = History(
                user_email=user_email,
                type="pdf",
                title=filename,
                summary=narration_script,
                audio_filename=audio_filename
            )
            db.session.add(history_entry)
            db.session.commit()

        flashcards = generate_flashcards(narration_script)
        # Store flashcards in session for later grading
        session['flashcards'] = flashcards
        return render_template("result.html", summary=narration_script, metrics=metrics, audio=audio_filename, flashcards=flashcards)

    return redirect("/")


@app.route("/youtube", methods=["POST"])
def youtube_summary():
    youtube_url = request.form["youtube_url"]
    video_id = extract_video_id(youtube_url)

    try:
        transcript = get_transcript(video_id)
    except Exception:
        audio_file = download_audio(youtube_url)
        transcript = audio_to_text(audio_file)
        if os.path.exists(audio_file):
            os.remove(audio_file)

    summary = summarize_text(transcript)
    metrics = evaluate_summary_metrics(transcript, summary)
    audio_filename = generate_tts_audio(summary, f"summary_{video_id}.mp3")

    # Save to history
    user_email = session.get("user")
    if user_email:
        history_entry = History(
            user_email=user_email,
            type="youtube",
            title=f"YouTube Video {video_id}",
            summary=summary,
            audio_filename=audio_filename
        )
        db.session.add(history_entry)
        db.session.commit()

    flashcards = generate_flashcards(summary)
    # Store flashcards in session for later grading
    session['flashcards'] = flashcards
    return render_template("result.html", summary=summary, metrics=metrics, audio=audio_filename, flashcards=flashcards)


@app.route("/add_users", methods=["GET", "POST"])
def add_users():
    if request.method == "POST":
        email = request.form["uemail"]
        password = request.form["upassword"]
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already exists.")
            return redirect(url_for("add_users"))
        hashed_password = generate_password_hash(password)
        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("successful"))
    return render_template("register.html")

@app.route("/successful")
def successful():
    return render_template("successful.html")

@app.route("/history")
def history():
    user_email = session.get("user")
    if not user_email:
        return {"history": []}
    history_entries = History.query.filter_by(user_email=user_email).order_by(History.created_at.desc()).all()
    history_data = [
        {
            "id": entry.id,
            "type": entry.type,
            "title": entry.title,
            "summary": entry.summary[:100] + "..." if len(entry.summary) > 100 else entry.summary,
            "audio_filename": entry.audio_filename,
            "created_at": entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for entry in history_entries
    ]
    return {"history": history_data}

@app.route("/history/<int:history_id>")
def view_history(history_id):
    user_email = session.get("user")
    if not user_email:
        return redirect(url_for("index"))
    history_entry = History.query.filter_by(id=history_id, user_email=user_email).first()
    if not history_entry:
        return "History item not found", 404
    return render_template("result.html", summary=history_entry.summary, audio=history_entry.audio_filename)


@app.route("/generate_flashcards", methods=["POST"])
def generate_flashcards_route():
    summary = request.form.get("summary")
    if not summary:
        return {"error": "No summary provided"}
    flashcards = generate_flashcards(summary)
    return {"flashcards": flashcards}


@app.route("/generate_study_flashcards", methods=["POST"])
def generate_study_flashcards_route():
    """Generate study flashcards from the current summary."""
    data = request.get_json()
    summary = data.get("summary")
    if not summary:
        return {"error": "No summary provided"}
    study_cards = generate_study_flashcards(summary)
    return study_cards


@app.route("/submit_flashcards", methods=["POST"])
def submit_flashcards():
    """Grade quiz answers using stored flashcards."""
    data = request.get_json()
    mcq_answers = data.get("mcq_answers", {})
    short_answers = data.get("short_answers", {})

    score = 0
    total = 0

    # Get flashcards from session instead of regenerating
    flashcards = session.get('flashcards', {})

    if "error" not in flashcards:
        # Score MCQs
        for i, mcq in enumerate(flashcards.get("mcqs", [])):
            total += 1
            user_answer = mcq_answers.get(str(i), "").strip()
            correct_answer = mcq["correct"].strip()
            # Normalize both answers (remove parentheses, extra spaces)
            user_answer = user_answer.replace(")", "").strip()
            correct_answer = correct_answer.replace(")", "").strip()
            if user_answer == correct_answer:
                score += 1



    return {"score": score, "total": total}


if __name__ == "__main__":
    app.run(debug=True)
