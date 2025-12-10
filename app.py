import streamlit as st
import numpy as np
from PIL import Image
import cv2
import base64
import io
import json
import os
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user

# Page configuration
st.set_page_config(
    page_title="Celebrity Image Classifier",
    page_icon="🌟",
    layout="centered"
)

# Celebrity database for reference
CELEBRITY_DATABASE = {
    "Leonardo DiCaprio": "Titanic, Inception, The Revenant",
    "Brad Pitt": "Fight Club, Troy, Once Upon a Time in Hollywood",
    "Angelina Jolie": "Tomb Raider, Maleficent, Mr. & Mrs. Smith",
    "Jennifer Aniston": "Friends, The Morning Show, Marley & Me",
    "Tom Cruise": "Mission Impossible, Top Gun, Jerry Maguire",
    "Scarlett Johansson": "Black Widow, Lost in Translation, Marriage Story",
    "Robert Downey Jr.": "Iron Man, Sherlock Holmes, Oppenheimer",
    "Chris Hemsworth": "Thor, Extraction, Rush",
    "Emma Watson": "Harry Potter, Beauty and the Beast, Little Women",
    "Johnny Depp": "Pirates of the Caribbean, Edward Scissorhands",
    "Margot Robbie": "Barbie, Wolf of Wall Street, I Tonya",
    "Ryan Gosling": "La La Land, Drive, The Notebook",
    "Natalie Portman": "Black Swan, Star Wars, V for Vendetta",
    "Morgan Freeman": "Shawshank Redemption, Bruce Almighty, Se7en",
    "Meryl Streep": "The Devil Wears Prada, Mamma Mia, Sophie's Choice",
    "George Clooney": "Ocean's Eleven, ER, Gravity",
    "Sandra Bullock": "Speed, Gravity, The Blind Side",
    "Dwayne Johnson": "Jumanji, Fast & Furious, Moana",
    "Will Smith": "Men in Black, The Pursuit of Happyness, I Am Legend",
    "Julia Roberts": "Pretty Woman, Erin Brockovich, Notting Hill",
    "Chris Evans": "Captain America, Knives Out, Gifted",
    "Anne Hathaway": "Les Miserables, The Dark Knight Rises, The Intern",
    "Hugh Jackman": "X-Men, The Greatest Showman, Les Miserables",
    "Kate Winslet": "Titanic, Mare of Easttown, The Reader",
    "Keanu Reeves": "John Wick, The Matrix, Speed",
    "Nicole Kidman": "Moulin Rouge, Big Little Lies, The Hours",
    "Matt Damon": "The Bourne Identity, Good Will Hunting, The Martian",
    "Charlize Theron": "Mad Max: Fury Road, Monster, Atomic Blonde",
    "Tom Hanks": "Forrest Gump, Cast Away, Saving Private Ryan",
    "Emma Stone": "La La Land, Easy A, The Amazing Spider-Man",
    "Zendaya": "Euphoria, Dune, Spider-Man",
    "Timothee Chalamet": "Dune, Call Me By Your Name, Little Women",
    "Jennifer Lawrence": "The Hunger Games, Silver Linings Playbook",
    "Chris Pratt": "Guardians of the Galaxy, Jurassic World",
    "Gal Gadot": "Wonder Woman, Fast & Furious",
    "Jason Momoa": "Aquaman, Game of Thrones",
    "Cate Blanchett": "Lord of the Rings, Carol, Thor: Ragnarok",
    "Benedict Cumberbatch": "Doctor Strange, Sherlock, The Imitation Game",
    "Joaquin Phoenix": "Joker, Walk the Line, Gladiator",
    "Viola Davis": "The Help, Fences, How to Get Away with Murder"
}

@st.cache_resource
def load_face_detector():
    """Load OpenCV's pre-trained face detector"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception:
        return None

def get_openai_client():
    """Get OpenAI client if API key is available"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)
    return None

def detect_faces(image, face_cascade):
    """Detect faces in the image"""
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces, img_array

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_celebrity_match(client, base64_image):
    """Use OpenAI Vision to analyze celebrity resemblance"""
    celebrity_list = ", ".join(list(CELEBRITY_DATABASE.keys())[:20])
    
    prompt = f"""Analyze this photo and identify which famous celebrities the person in the image most closely resembles.

Consider facial features like:
- Face shape and structure
- Eye shape, color, and placement
- Nose shape and size
- Lip shape and fullness
- Jawline and chin
- Overall facial proportions
- Hair style and color (if visible)
- Expression and demeanor

Provide your top 5 celebrity matches with confidence percentages. The celebrities can be from this list or any other well-known celebrities: {celebrity_list}

Respond in JSON format exactly like this:
{{
    "matches": [
        {{"name": "Celebrity Name", "confidence": 85, "reason": "Brief explanation of similarity"}},
        {{"name": "Celebrity Name 2", "confidence": 72, "reason": "Brief explanation"}},
        {{"name": "Celebrity Name 3", "confidence": 65, "reason": "Brief explanation"}},
        {{"name": "Celebrity Name 4", "confidence": 58, "reason": "Brief explanation"}},
        {{"name": "Celebrity Name 5", "confidence": 52, "reason": "Brief explanation"}}
    ],
    "face_detected": true,
    "analysis_notes": "Brief overall analysis of the person's distinctive features"
}}

If no clear face is detected, set face_detected to false and provide your best guesses based on what's visible."""

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=1024
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {"error": str(e)}

def draw_faces_on_image(img_array, faces):
    """Draw rectangles around detected faces"""
    img_with_faces = img_array.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 3)
    return img_with_faces

# Main app
st.title("🌟 Celebrity Image Classifier")
st.markdown("Upload a photo to discover which famous celebrities you resemble using AI vision analysis!")

# Check for API key
client = get_openai_client()

if not client:
    st.error("""
    **OpenAI API Key Required**
    
    This app uses AI vision to analyze faces and match them to celebrities. 
    Please add your OpenAI API key to use this feature.
    """)
    st.info("Add your OPENAI_API_KEY in the Secrets tab to enable celebrity matching.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("How It Works")
    st.markdown("""
    This app uses advanced AI vision to analyze your photo:
    
    1. **Upload Photo** - Provide a clear face image
    2. **Face Detection** - OpenCV locates faces
    3. **AI Analysis** - GPT-5 Vision analyzes facial features
    4. **Celebrity Matching** - AI identifies lookalikes
    
    **Best Results:**
    - Clear, well-lit photos
    - Face the camera directly
    - Avoid heavy filters or makeup
    """)
    
    st.divider()
    st.header("Celebrity Database")
    st.markdown(f"**{len(CELEBRITY_DATABASE)}** celebrities in reference database")
    
    with st.expander("View celebrities"):
        for name, films in sorted(CELEBRITY_DATABASE.items()):
            st.markdown(f"• **{name}**")
            st.caption(f"  {films[:50]}...")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear photo with a visible face for best results"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Your Photo")
        st.image(image, use_container_width=True)
    
    face_cascade = load_face_detector()
    
    if face_cascade is not None:
        faces, img_array = detect_faces(image, face_cascade)
        
        with col2:
            st.subheader("🔍 Face Detection")
            if len(faces) > 0:
                img_with_faces = draw_faces_on_image(img_array, faces)
                st.image(img_with_faces, use_container_width=True)
                st.success(f"✓ Detected {len(faces)} face(s)")
            else:
                st.image(image, use_container_width=True)
                st.warning("No face clearly detected - AI will analyze the full image")
    else:
        with col2:
            st.subheader("🔍 Preview")
            st.image(image, use_container_width=True)
    
    st.divider()
    st.subheader("🎯 Celebrity Match Analysis")
    
    with st.spinner("🧠 AI is analyzing your photo..."):
        base64_img = image_to_base64(image)
        result = analyze_celebrity_match(client, base64_img)
    
    if "error" in result:
        st.error(f"Analysis failed: {result['error']}")
    else:
        matches = result.get("matches", [])
        analysis_notes = result.get("analysis_notes", "")
        
        if matches:
            st.markdown("**Your Top Celebrity Lookalikes:**")
            
            for i, match in enumerate(matches, 1):
                with st.container():
                    col_rank, col_info, col_bar = st.columns([1, 4, 5])
                    
                    with col_rank:
                        if i == 1:
                            st.markdown("### 🥇")
                        elif i == 2:
                            st.markdown("### 🥈")
                        elif i == 3:
                            st.markdown("### 🥉")
                        else:
                            st.markdown(f"### #{i}")
                    
                    with col_info:
                        st.markdown(f"**{match['name']}**")
                        if match['name'] in CELEBRITY_DATABASE:
                            st.caption(f"Known for: {CELEBRITY_DATABASE[match['name']][:40]}...")
                    
                    with col_bar:
                        confidence = min(100, max(0, match.get('confidence', 50)))
                        st.progress(confidence / 100)
                        st.caption(f"{confidence}% match")
                    
                    if match.get('reason'):
                        st.caption(f"💡 {match['reason']}")
                    
                    st.markdown("---")
            
            st.divider()
            best_match = matches[0]
            
            st.success(f"""
            ### 🏆 Best Match: **{best_match['name']}**
            
            You most closely resemble **{best_match['name']}** with a **{best_match.get('confidence', 0)}%** similarity score!
            
            **Why this match:** {best_match.get('reason', 'Similar facial features detected.')}
            """)
            
            if analysis_notes:
                with st.expander("📊 Detailed Analysis"):
                    st.markdown(f"**AI Analysis Notes:**\n\n{analysis_notes}")
        else:
            st.warning("Could not determine celebrity matches. Please try with a clearer photo.")

else:
    st.markdown("### 👆 Upload a photo to find your celebrity lookalike!")
    
    st.markdown("""
    **What this app does:**
    - Uses AI vision to analyze your facial features
    - Compares your features to famous celebrities
    - Provides detailed matching with explanations
    - Shows your top 5 celebrity lookalikes
    """)
    
    st.markdown("---")
    st.subheader("🎬 Featured Celebrities")
    
    cols = st.columns(3)
    featured = list(CELEBRITY_DATABASE.items())[:12]
    
    for i, (name, films) in enumerate(featured):
        with cols[i % 3]:
            st.markdown(f"⭐ **{name}**")
            st.caption(films[:35] + "...")

# Footer
st.markdown("---")
st.caption("Celebrity Image Classifier | Powered by AI Vision Analysis | For entertainment purposes")
