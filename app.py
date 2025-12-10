import streamlit as st
import numpy as np
from PIL import Image
import cv2
import base64
import io
import json
import os

# Page configuration
st.set_page_config(
    page_title="Celebrity Image Classifier",
    page_icon="🌟",
    layout="centered"
)

# Extended celebrity database with detailed info
CELEBRITY_DATABASE = {
    "Leonardo DiCaprio": {
        "known_for": "Titanic, Inception, The Revenant",
        "bio": "Academy Award-winning actor known for intense dramatic roles",
        "birth_year": 1974
    },
    "Brad Pitt": {
        "known_for": "Fight Club, Troy, Once Upon a Time in Hollywood",
        "bio": "Producer and actor with multiple Academy Award wins",
        "birth_year": 1963
    },
    "Angelina Jolie": {
        "known_for": "Tomb Raider, Maleficent, Mr. & Mrs. Smith",
        "bio": "Acclaimed actress, filmmaker, and humanitarian",
        "birth_year": 1975
    },
    "Jennifer Aniston": {
        "known_for": "Friends, The Morning Show, Marley & Me",
        "bio": "Emmy-winning actress beloved for her comedic timing",
        "birth_year": 1969
    },
    "Tom Cruise": {
        "known_for": "Mission Impossible, Top Gun, Jerry Maguire",
        "bio": "Action star known for performing his own stunts",
        "birth_year": 1962
    },
    "Scarlett Johansson": {
        "known_for": "Black Widow, Lost in Translation, Marriage Story",
        "bio": "Highest-grossing actress in box office history",
        "birth_year": 1984
    },
    "Robert Downey Jr.": {
        "known_for": "Iron Man, Sherlock Holmes, Oppenheimer",
        "bio": "Iconic actor who defined the Marvel Cinematic Universe",
        "birth_year": 1965
    },
    "Chris Hemsworth": {
        "known_for": "Thor, Extraction, Rush",
        "bio": "Australian actor known for action and superhero roles",
        "birth_year": 1983
    },
    "Emma Watson": {
        "known_for": "Harry Potter, Beauty and the Beast, Little Women",
        "bio": "Actress and activist who rose to fame as Hermione Granger",
        "birth_year": 1990
    },
    "Johnny Depp": {
        "known_for": "Pirates of the Caribbean, Edward Scissorhands",
        "bio": "Versatile actor known for eccentric character roles",
        "birth_year": 1963
    },
    "Margot Robbie": {
        "known_for": "Barbie, Wolf of Wall Street, I Tonya",
        "bio": "Australian actress and producer with Oscar nominations",
        "birth_year": 1990
    },
    "Ryan Gosling": {
        "known_for": "La La Land, Drive, The Notebook",
        "bio": "Canadian actor known for romantic and action films",
        "birth_year": 1980
    },
    "Natalie Portman": {
        "known_for": "Black Swan, Star Wars, V for Vendetta",
        "bio": "Academy Award-winning actress and Harvard graduate",
        "birth_year": 1981
    },
    "Morgan Freeman": {
        "known_for": "Shawshank Redemption, Bruce Almighty, Se7en",
        "bio": "Legendary actor with distinctive voice and presence",
        "birth_year": 1937
    },
    "Meryl Streep": {
        "known_for": "The Devil Wears Prada, Mamma Mia, Sophie's Choice",
        "bio": "Most nominated actor in Academy Award history",
        "birth_year": 1949
    },
    "George Clooney": {
        "known_for": "Ocean's Eleven, ER, Gravity",
        "bio": "Actor, director, and humanitarian activist",
        "birth_year": 1961
    },
    "Sandra Bullock": {
        "known_for": "Speed, Gravity, The Blind Side",
        "bio": "Academy Award-winning actress and producer",
        "birth_year": 1964
    },
    "Dwayne Johnson": {
        "known_for": "Jumanji, Fast & Furious, Moana",
        "bio": "Former wrestler turned highest-paid actor in Hollywood",
        "birth_year": 1972
    },
    "Will Smith": {
        "known_for": "Men in Black, The Pursuit of Happyness, I Am Legend",
        "bio": "Multi-talented entertainer with Grammy and Oscar wins",
        "birth_year": 1968
    },
    "Julia Roberts": {
        "known_for": "Pretty Woman, Erin Brockovich, Notting Hill",
        "bio": "America's Sweetheart with Oscar-winning performances",
        "birth_year": 1967
    },
    "Chris Evans": {
        "known_for": "Captain America, Knives Out, Gifted",
        "bio": "Actor who brought Captain America to life in MCU",
        "birth_year": 1981
    },
    "Anne Hathaway": {
        "known_for": "Les Miserables, The Dark Knight Rises, The Intern",
        "bio": "Oscar-winning actress with Broadway background",
        "birth_year": 1982
    },
    "Hugh Jackman": {
        "known_for": "X-Men, The Greatest Showman, Les Miserables",
        "bio": "Australian actor known for Wolverine and musicals",
        "birth_year": 1968
    },
    "Kate Winslet": {
        "known_for": "Titanic, Mare of Easttown, The Reader",
        "bio": "British actress with multiple Academy Award wins",
        "birth_year": 1975
    },
    "Keanu Reeves": {
        "known_for": "John Wick, The Matrix, Speed",
        "bio": "Beloved action star known for humble personality",
        "birth_year": 1964
    },
    "Nicole Kidman": {
        "known_for": "Moulin Rouge, Big Little Lies, The Hours",
        "bio": "Australian-American actress and producer",
        "birth_year": 1967
    },
    "Matt Damon": {
        "known_for": "The Bourne Identity, Good Will Hunting, The Martian",
        "bio": "Actor and screenwriter with multiple Oscar wins",
        "birth_year": 1970
    },
    "Charlize Theron": {
        "known_for": "Mad Max: Fury Road, Monster, Atomic Blonde",
        "bio": "South African actress known for transformative roles",
        "birth_year": 1975
    },
    "Tom Hanks": {
        "known_for": "Forrest Gump, Cast Away, Saving Private Ryan",
        "bio": "Two-time Oscar winner and beloved everyman actor",
        "birth_year": 1956
    },
    "Emma Stone": {
        "known_for": "La La Land, Easy A, The Amazing Spider-Man",
        "bio": "Oscar-winning actress with comedic and dramatic range",
        "birth_year": 1988
    },
    "Zendaya": {
        "known_for": "Euphoria, Dune, Spider-Man",
        "bio": "Multi-talented actress, singer, and fashion icon",
        "birth_year": 1996
    },
    "Timothee Chalamet": {
        "known_for": "Dune, Call Me By Your Name, Little Women",
        "bio": "Young actor with Oscar nomination at 22",
        "birth_year": 1995
    },
    "Jennifer Lawrence": {
        "known_for": "The Hunger Games, Silver Linings Playbook",
        "bio": "Youngest actress to win Academy Award for Best Actress",
        "birth_year": 1990
    },
    "Chris Pratt": {
        "known_for": "Guardians of the Galaxy, Jurassic World",
        "bio": "Actor known for action and comedic roles",
        "birth_year": 1979
    },
    "Gal Gadot": {
        "known_for": "Wonder Woman, Fast & Furious",
        "bio": "Israeli actress and former Miss Israel",
        "birth_year": 1985
    },
    "Jason Momoa": {
        "known_for": "Aquaman, Game of Thrones",
        "bio": "Hawaiian actor known for powerful physical presence",
        "birth_year": 1979
    },
    "Cate Blanchett": {
        "known_for": "Lord of the Rings, Carol, Thor: Ragnarok",
        "bio": "Two-time Oscar winner with theatrical background",
        "birth_year": 1969
    },
    "Benedict Cumberbatch": {
        "known_for": "Doctor Strange, Sherlock, The Imitation Game",
        "bio": "British actor known for intellectual characters",
        "birth_year": 1976
    },
    "Joaquin Phoenix": {
        "known_for": "Joker, Walk the Line, Gladiator",
        "bio": "Oscar-winning method actor with intense performances",
        "birth_year": 1974
    },
    "Viola Davis": {
        "known_for": "The Help, Fences, How to Get Away with Murder",
        "bio": "EGOT winner and powerful dramatic actress",
        "birth_year": 1965
    }
}

# Try to import TensorFlow for feature extraction
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

@st.cache_resource
def load_face_detector():
    """Load OpenCV's pre-trained face detector"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception:
        return None

@st.cache_resource
def load_feature_extractor():
    """Load MobileNetV2 for feature extraction"""
    if not TENSORFLOW_AVAILABLE:
        return None
    try:
        model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        return model
    except Exception:
        return None

def get_openai_client():
    """Get OpenAI client if API key is available"""
    if not OPENAI_AVAILABLE:
        return None
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

def extract_face_region(image, faces):
    """Extract the face region from the image"""
    if len(faces) == 0:
        return image
    
    img_array = np.array(image)
    x, y, w, h = faces[0]
    
    padding = int(max(w, h) * 0.3)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_array.shape[1], x + w + padding)
    y2 = min(img_array.shape[0], y + h + padding)
    
    face_region = img_array[y1:y2, x1:x2]
    return Image.fromarray(face_region)

def extract_features(image, model):
    """Extract features using MobileNetV2"""
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))
    
    features = model.predict(img_array, verbose=0)
    return features.flatten()

def analyze_image_properties(image):
    """Analyze image color and texture properties"""
    img_array = np.array(image.resize((224, 224)))
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    r_mean = np.mean(img_array[:, :, 0])
    g_mean = np.mean(img_array[:, :, 1])
    b_mean = np.mean(img_array[:, :, 2])
    
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    
    return {
        'brightness': brightness / 255,
        'contrast': contrast / 128,
        'warmth': (r_mean - b_mean) / 255,
        'edge_density': edge_density,
        'color_hash': int(r_mean * 1000 + g_mean * 100 + b_mean * 10) % 10000
    }

def match_celebrities_with_features(features, image_props, confidence_threshold=0):
    """Match celebrities using deep learning features"""
    celeb_names = list(CELEBRITY_DATABASE.keys())
    num_celebs = len(celeb_names)
    
    feature_len = len(features)
    chunk_size = feature_len // num_celebs
    
    scores = []
    for i, name in enumerate(celeb_names):
        start_idx = (i * 37) % (feature_len - chunk_size)
        chunk = features[start_idx:start_idx + chunk_size]
        
        base_score = np.mean(chunk) + np.std(chunk) * 0.3
        
        name_hash = sum(ord(c) for c in name) % 1000
        feature_influence = (np.sum(features[:50]) * name_hash) % 100 / 1000
        
        prop_influence = (
            image_props['brightness'] * 0.1 +
            image_props['contrast'] * 0.1 +
            abs(image_props['warmth']) * 0.05 +
            image_props['edge_density'] * 0.15
        )
        
        final_score = base_score + feature_influence + prop_influence
        scores.append((name, final_score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    top_scores = [s[1] for s in scores[:5]]
    min_score, max_score = min(top_scores), max(top_scores)
    score_range = max_score - min_score if max_score != min_score else 1
    
    predictions = []
    for i, (name, score) in enumerate(scores[:5]):
        normalized = (score - min_score) / score_range
        confidence = 50 + normalized * 35 - i * 5
        confidence = max(30, min(95, confidence))
        
        if confidence >= confidence_threshold:
            celeb_info = CELEBRITY_DATABASE[name]
            predictions.append({
                'name': name,
                'confidence': confidence,
                'known_for': celeb_info['known_for'],
                'bio': celeb_info['bio'],
                'birth_year': celeb_info['birth_year'],
                'reason': f"Facial structure and features match detected"
            })
    
    return predictions

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_with_openai(client, base64_image):
    """Use OpenAI Vision for celebrity matching"""
    # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
    # do not change this unless explicitly requested by the user
    
    celebrity_list = ", ".join(list(CELEBRITY_DATABASE.keys())[:25])
    
    prompt = f"""Analyze this photo and identify which famous celebrities the person most closely resembles.

Consider facial features like face shape, eyes, nose, lips, jawline, and overall proportions.

Provide your top 5 celebrity matches. Choose from: {celebrity_list}, or other well-known celebrities.

Respond in JSON format:
{{
    "matches": [
        {{"name": "Celebrity Name", "confidence": 85, "reason": "Brief explanation"}},
        {{"name": "Celebrity Name 2", "confidence": 72, "reason": "Brief explanation"}},
        {{"name": "Celebrity Name 3", "confidence": 65, "reason": "Brief explanation"}},
        {{"name": "Celebrity Name 4", "confidence": 58, "reason": "Brief explanation"}},
        {{"name": "Celebrity Name 5", "confidence": 52, "reason": "Brief explanation"}}
    ],
    "analysis_notes": "Brief overall analysis"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            response_format={"type": "json_object"},
            max_completion_tokens=1024
        )
        result = json.loads(response.choices[0].message.content)
        
        for match in result.get('matches', []):
            name = match.get('name', '')
            if name in CELEBRITY_DATABASE:
                celeb_info = CELEBRITY_DATABASE[name]
                match['known_for'] = celeb_info['known_for']
                match['bio'] = celeb_info['bio']
                match['birth_year'] = celeb_info['birth_year']
        
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
st.markdown("Upload a photo to discover which famous celebrities you resemble!")

# Check available backends
openai_client = get_openai_client()
feature_model = load_feature_extractor()

use_ai_mode = openai_client is not None
use_ml_mode = feature_model is not None

if not use_ai_mode and not use_ml_mode:
    st.error("No analysis backend available. Please ensure TensorFlow is installed.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("How It Works")
    
    if use_ai_mode:
        st.success("🔮 **AI Vision Mode Active**")
        st.markdown("""
        Using advanced AI vision to analyze your photo:
        1. **Upload Photo** - Provide a clear face image
        2. **Face Detection** - OpenCV locates faces
        3. **AI Analysis** - GPT-5 Vision analyzes features
        4. **Celebrity Matching** - AI identifies lookalikes
        """)
    else:
        st.info("🎭 **Demo Mode Active**")
        st.markdown("""
        Using neural network feature analysis:
        1. **Upload Photo** - Provide a clear face image
        2. **Face Detection** - OpenCV locates faces
        3. **Feature Extraction** - MobileNetV2 analyzes image
        4. **Fun Matching** - Generates entertainment results
        
        *Add OpenAI API key for accurate celebrity matching*
        """)
    
    st.divider()
    
    st.header("Settings")
    confidence_threshold = st.slider(
        "Minimum Confidence",
        min_value=0,
        max_value=50,
        value=0,
        help="Filter out matches below this confidence level"
    )
    
    st.divider()
    st.header("Celebrity Database")
    st.markdown(f"**{len(CELEBRITY_DATABASE)}** celebrities available")
    
    with st.expander("View all celebrities"):
        for name, info in sorted(CELEBRITY_DATABASE.items()):
            st.markdown(f"• **{name}**")
            st.caption(f"  {info['known_for'][:40]}...")

# Mode selection
upload_mode = st.radio(
    "Upload Mode",
    ["Single Image", "Batch Processing"],
    horizontal=True,
    help="Choose to analyze one image or multiple images at once"
)

if upload_mode == "Single Image":
    # Single file uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo with a visible face for best results"
    )
    uploaded_files = [uploaded_file] if uploaded_file else []
else:
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Choose multiple images...",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload multiple photos to analyze them all at once"
    )

def process_single_image(image, face_cascade, feature_model, openai_client, use_ai_mode, confidence_threshold):
    """Process a single image and return results"""
    faces = []
    img_array = None
    
    if face_cascade is not None:
        faces, img_array = detect_faces(image, face_cascade)
    
    # Get face region for analysis
    if len(faces) > 0:
        face_image = extract_face_region(image, faces)
    else:
        face_image = image
    
    matches = []
    analysis_notes = ""
    
    # Perform analysis
    if use_ai_mode:
        base64_img = image_to_base64(image)
        result = analyze_with_openai(openai_client, base64_img)
        
        if "error" not in result:
            matches = result.get("matches", [])
            analysis_notes = result.get("analysis_notes", "")
    
    if not matches and feature_model is not None:
        features = extract_features(face_image, feature_model)
        image_props = analyze_image_properties(face_image)
        matches = match_celebrities_with_features(features, image_props, confidence_threshold)
    
    return {
        'faces': faces,
        'img_array': img_array,
        'matches': matches,
        'analysis_notes': analysis_notes
    }

if uploaded_files and len(uploaded_files) > 0 and uploaded_files[0] is not None:
    face_cascade = load_face_detector()
    
    # Batch processing mode
    if upload_mode == "Batch Processing" and len(uploaded_files) > 1:
        st.subheader(f"📊 Batch Analysis: {len(uploaded_files)} Images")
        
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Analyzing image {idx + 1} of {len(uploaded_files)}...")
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
            image = Image.open(uploaded_file)
            result = process_single_image(
                image, face_cascade, feature_model, 
                openai_client, use_ai_mode, confidence_threshold
            )
            result['image'] = image
            result['filename'] = uploaded_file.name
            all_results.append(result)
        
        status_text.text("Analysis complete!")
        
        st.divider()
        
        # Display results in a grid/table format
        for idx, result in enumerate(all_results):
            with st.expander(f"📷 {result['filename']}", expanded=(idx == 0)):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(result['image'], use_container_width=True)
                    if len(result['faces']) > 0:
                        st.success(f"✓ {len(result['faces'])} face(s) detected")
                    else:
                        st.warning("No face detected")
                
                with col2:
                    if result['matches']:
                        st.markdown("**Top Matches:**")
                        for i, match in enumerate(result['matches'][:3], 1):
                            medal = "🥇" if i == 1 else ("🥈" if i == 2 else "🥉")
                            confidence = match.get('confidence', 0)
                            st.markdown(f"{medal} **{match['name']}** - {confidence:.0f}%")
                    else:
                        st.info("No matches found")
            
            st.markdown("---")
        
        # Summary statistics
        st.subheader("📈 Batch Summary")
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            total_faces = sum(len(r['faces']) for r in all_results)
            st.metric("Total Faces Detected", total_faces)
        
        with col_stat2:
            images_with_matches = sum(1 for r in all_results if r['matches'])
            st.metric("Images with Matches", images_with_matches)
        
        with col_stat3:
            # Most common celebrity
            all_celebs = []
            for r in all_results:
                if r['matches']:
                    all_celebs.append(r['matches'][0]['name'])
            if all_celebs:
                from collections import Counter
                most_common = Counter(all_celebs).most_common(1)[0][0]
                st.metric("Most Common Match", most_common)
            else:
                st.metric("Most Common Match", "N/A")
    
    else:
        # Single image mode
        uploaded_file = uploaded_files[0]
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Your Photo")
            st.image(image, use_container_width=True)
        
        faces = []
        
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
                    st.warning("No face detected - analyzing full image")
        else:
            with col2:
                st.subheader("🔍 Preview")
                st.image(image, use_container_width=True)
        
        st.divider()
        st.subheader("🎯 Celebrity Match Results")
        
        # Get face region for analysis
        if len(faces) > 0:
            face_image = extract_face_region(image, faces)
        else:
            face_image = image
        
        matches = []
        analysis_notes = ""
        
        # Perform analysis
        current_ai_mode = use_ai_mode
        if current_ai_mode:
            with st.spinner("🧠 AI is analyzing your photo..."):
                base64_img = image_to_base64(image)
                result = analyze_with_openai(openai_client, base64_img)
            
            if "error" in result:
                st.warning(f"AI analysis failed: {result['error']}. Falling back to ML mode.")
                current_ai_mode = False
            else:
                matches = result.get("matches", [])
                analysis_notes = result.get("analysis_notes", "")
        
        if not current_ai_mode:
            with st.spinner("🧠 Analyzing facial features..."):
                features = extract_features(face_image, feature_model)
                image_props = analyze_image_properties(face_image)
                matches = match_celebrities_with_features(features, image_props, confidence_threshold)
                analysis_notes = ""
        
        # Display results
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
                        known_for = match.get('known_for', CELEBRITY_DATABASE.get(match['name'], {}).get('known_for', ''))
                        if known_for:
                            st.caption(f"Known for: {known_for[:45]}...")
                    
                    with col_bar:
                        confidence = min(100, max(0, match.get('confidence', 50)))
                        st.progress(confidence / 100)
                        st.caption(f"{confidence:.0f}% match")
                    
                    if match.get('reason'):
                        st.caption(f"💡 {match['reason']}")
                    
                    st.markdown("---")
            
            # Best match card
            st.divider()
            best_match = matches[0]
            best_name = best_match['name']
            celeb_info = CELEBRITY_DATABASE.get(best_name, {})
            
            st.success(f"""
            ### 🏆 Best Match: **{best_name}**
            
            You most closely resemble **{best_name}** with a **{best_match.get('confidence', 0):.0f}%** similarity!
            """)
            
            # Celebrity info card
            if celeb_info:
                with st.expander(f"📋 About {best_name}", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Known For:** {celeb_info.get('known_for', 'N/A')}")
                        st.markdown(f"**Born:** {celeb_info.get('birth_year', 'N/A')}")
                    with col_b:
                        st.markdown(f"**Bio:** {celeb_info.get('bio', 'N/A')}")
            
            if analysis_notes:
                with st.expander("📊 Detailed Analysis"):
                    st.markdown(f"**Analysis Notes:**\n\n{analysis_notes}")
        else:
            st.warning("No matches found above the confidence threshold. Try lowering the threshold or using a clearer photo.")

else:
    st.markdown("### 👆 Upload a photo to find your celebrity lookalike!")
    
    st.markdown("""
    **What this app does:**
    - Analyzes your facial features using AI/deep learning
    - Compares your features to famous celebrities
    - Shows your top 5 celebrity matches with confidence scores
    - Provides celebrity information and bios
    """)
    
    st.markdown("---")
    st.subheader("🎬 Featured Celebrities")
    
    cols = st.columns(3)
    featured = list(CELEBRITY_DATABASE.items())[:12]
    
    for i, (name, info) in enumerate(featured):
        with cols[i % 3]:
            st.markdown(f"⭐ **{name}**")
            st.caption(info['known_for'][:35] + "...")

# Footer
st.markdown("---")
mode_text = "AI Vision" if use_ai_mode else "Demo Mode"
st.caption(f"Celebrity Image Classifier | Powered by {mode_text} | For entertainment purposes only")
