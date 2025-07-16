import os
import openai
from flask import Flask, render_template, request, jsonify, session
import uuid
import json
from datetime import datetime
import random
from transformers.pipelines import pipeline
from openai import OpenAI
import threading
import time


from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Configuration ---
LOG_FILE = os.path.join(os.path.dirname(__file__), 'chat_logs.jsonl')
MODAL_PERSONA_FILE = os.path.join(os.path.dirname(__file__), 'modal-persona.json')
EMOTIONS = ['Happy', 'Sad', 'Angry', 'Surprised', 'Afraid', 'Disgusted', 'Neutral', 'Other']
CHARACTERS = ['Detective', 'Alien', 'Teacher', 'Robot', 'Pirate', 'Doctor', 'Wizard']

# Global variables for caching and performance
_emotion_classifier = None
_emotion_classifier_lock = threading.Lock()
_modal_personas = None
_emotion_directives = None
_memory_store = {}
_memory_cleanup_interval = 300  # 5 minutes
_last_cleanup = time.time()

# Load modal persona data
def load_modal_personas():
    global _modal_personas
    if _modal_personas is None:
        try:
            with open(MODAL_PERSONA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _modal_personas = {modality['key']: modality['description'] for modality in data['modalities']}
        except Exception as e:
            print(f"Error loading modal personas: {e}")
            _modal_personas = {}
    return _modal_personas

def load_emotion_directives():
    global _emotion_directives
    if _emotion_directives is None:
        try:
            with open("C:/Users/TRISHA/eq2ai-project/chatbot_ui/emotion_director.json", "r") as f:
                data = json.load(f)
                _emotion_directives = data["full_emotion_opposite_directives"]
        except Exception as e:
            print(f"Error loading emotion directives: {e}")
            _emotion_directives = {}
    return _emotion_directives

def get_random_persona_key():
    """Get a random persona key from available modalities"""
    personas = load_modal_personas()
    if personas:
        return random.choice(list(personas.keys()))
    return "CBT"  # Fallback if no personas loaded

def get_emotion_based_persona_key(detected_emotion):
    """Select persona key based on detected emotion"""
    # Positive emotions - use anti-modalities
    positive_emotions = [
        'admiration', 'amusement', 'approval', 'caring', 'curiosity', 
        'desire', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 
        'pride', 'relief'
    ]
    
    # Negative emotions - use normal modalities
    negative_emotions = [
        'anger', 'annoyance', 'confusion', 'disappointment', 'disapproval',
        'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'
    ]
    
    detected_emotion_lower = detected_emotion.lower()
    personas = load_modal_personas()
    
    if detected_emotion_lower in positive_emotions:
        # Use anti-modalities for positive emotions
        anti_keys = [key for key in personas.keys() if key.startswith('Anti-')]
        if anti_keys:
            return random.choice(anti_keys)
        else:
            return get_random_persona_key()
    
    elif detected_emotion_lower in negative_emotions:
        # Use normal modalities for negative emotions
        normal_keys = [key for key in personas.keys() if not key.startswith('Anti-')]
        if normal_keys:
            return random.choice(normal_keys)
        else:
            return get_random_persona_key()
    
    else:
        # For neutral/other emotions, use any modality randomly
        return get_random_persona_key()

# ---- Emotion Opposites ---- #
emotion_opposites = {
    "admiration": "humility", "amusement": "seriousness", "anger": "calm",
    "annoyance": "ease", "approval": "detachment", "caring": "indifference",
    "confusion": "clarity", "curiosity": "certainty", "desire": "satisfaction",
    "disappointment": "hope", "disapproval": "acceptance", "disgust": "admiration" ,"embarrassment": "confidence",
    "excitement": "composure", "fear": "reassurance", "gratitude": "entitlement",
    "grief": "comfort", "joy": "reflection", "love": "detachment",
    "nervousness": "confidence", "neutral": "engagement", "optimism": "skepticism",
    "pride": "modesty", "realization": "uncertainty", "relief": "tension",
    "remorse": "forgiveness", "sadness": "hope", "surprise": "predictability"
}

# ---- Conversation Memory Tracker ---- #
class ConversationMemory:
    def __init__(self, session_id, max_turns=5):
        self.session_id = session_id
        self.history = []
        self.max_turns = max_turns
        self.persona_key = None  # Will be set based on first emotion detected
        self.last_activity = time.time()

    def _update_history(self, user_input, assistant_response):
        self.history.append({"user": user_input, "assistant": assistant_response})
        if len(self.history) > self.max_turns:
            self.history.pop(0)
        self.last_activity = time.time()

    def get_history_prompt(self):
        prompt = ""
        for turn in self.history:
            prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        return prompt

    def add_turn(self, user_input, assistant_response):
        self._update_history(user_input, assistant_response)

    def get_persona_key(self):
        return self.persona_key
    
    def set_persona_key(self, persona_key):
        self.persona_key = persona_key

def get_opposite_emotion(emotion):
    return emotion_opposites.get(emotion.lower(), "neutral")

# --- Emotion Classifier with Lazy Loading and Caching ---
def load_emotion_classifier():
    global _emotion_classifier
    with _emotion_classifier_lock:
        if _emotion_classifier is None:
            try:
                print("Loading emotion classifier...")
                _emotion_classifier = pipeline(
                    "text-classification",
                    model="monologg/bert-base-cased-goemotions-original",
                    return_all_scores=True,
                    truncation=True
                )
                print("Emotion classifier loaded successfully")
            except Exception as e:
                print(f"Error loading emotion classifier: {e}")
                return None
    return _emotion_classifier

def classify_emotion(text):
    classifier = load_emotion_classifier()
    if classifier is None:
        return "neutral", []
    try:
        result = classifier(text)
        if result and isinstance(result, list) and len(result) > 0:
            scores = result[0]
            if isinstance(scores, list):
                score_list = [(score['label'], float(score['score'])) for score in scores]
                sorted_scores = sorted(score_list, key=lambda x: x[1], reverse=True)
                top_emotion = sorted_scores[0][0]
                return top_emotion, scores
        return "neutral", []
    except Exception as e:
        print(f"Error classifying emotion: {e}")
        return "neutral", []

# --- OpenAI Chat Completion with Improved Configuration ---
def openai_generate_response(prompt, system_prompt=None):
    api_key = os.getenv("API_KEY")
    client = OpenAI(api_key=api_key)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,  # Increased from 100 for better responses
            temperature=0.7,
            timeout=30  # Add timeout to prevent hanging
        )
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content.strip()
        return "I'm having trouble generating a response right now."
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "[OpenAI API error: {}]".format(e)

# --- Memory Cleanup Function ---
def cleanup_old_memories():
    global _memory_store, _last_cleanup
    current_time = time.time()
    if current_time - _last_cleanup > _memory_cleanup_interval:
        cutoff_time = current_time - 3600  # 1 hour
        to_remove = []
        for session_id, memory in _memory_store.items():
            if memory.last_activity < cutoff_time:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del _memory_store[session_id]
        
        if to_remove:
            print(f"Cleaned up {len(to_remove)} old memory sessions")
        _last_cleanup = current_time

# --- Helper Functions ---
def log_message(role, message, detected_emotion=None, self_reported_emotion=None, target_emotion=None, emotion_scores=None, persona_key=None):
    try:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session.get('session_id'),
            'character': session.get('character'),
            'role': role,
            'message': message,
            'detected_emotion': detected_emotion,
            'self_reported_emotion': self_reported_emotion,
            'target_emotion': target_emotion,
            'emotion_scores': emotion_scores,
            'persona_key': persona_key
        }
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"ERROR: Failed to log message: {e}")

def get_fallback_response(character, target_emotion, user_message):
    fallback_responses = {
        "Detective": {
            "calm": "I see what's happening here. Let's approach this systematically.",
            "hope": "Every case has a solution. We'll figure this out together.",
            "confidence": "I've seen cases like this before. We can handle this.",
            "engagement": "Tell me more about what's going on. I'm listening."
        },
        "Teacher": {
            "calm": "I understand this is challenging. Let's take a deep breath and work through it.",
            "hope": "Every problem has a solution. We'll find the right approach.",
            "confidence": "You have the skills to handle this. Let's break it down.",
            "engagement": "I'm here to help you learn and grow through this."
        },
        "Robot": {
            "calm": "Processing your input. Analyzing optimal response patterns.",
            "hope": "Calculating positive outcomes. Probability of success: high.",
            "confidence": "My circuits are designed to help. Let's solve this together.",
            "engagement": "I am fully operational and ready to assist you."
        },
        "Pirate": {
            "calm": "Aye, matey! Let's navigate these rough waters together.",
            "hope": "There's treasure in every challenge, me hearty!",
            "confidence": "With the right crew, we can weather any storm!",
            "engagement": "Tell me your tale, and I'll help you find your way."
        },
        "Doctor": {
            "calm": "I understand this is stressful. Let's address this step by step.",
            "hope": "There's always a path to improvement. We'll find it together.",
            "confidence": "You're stronger than you think. Let's work through this.",
            "engagement": "I'm here to help you heal and grow through this."
        },
        "Wizard": {
            "calm": "The ancient spells of wisdom can guide us through this.",
            "hope": "Magic exists in every challenge. We'll find the right spell.",
            "confidence": "Your inner magic is powerful. Let's channel it together.",
            "engagement": "Share your story, and I'll help you find your magical path."
        },
        "Alien": {
            "calm": "From my observations, this situation requires careful analysis.",
            "hope": "My species believes in infinite possibilities. Solutions exist.",
            "confidence": "Your human resilience is remarkable. We can overcome this.",
            "engagement": "I am fascinated by your experience. Please continue."
        }
    }
    char_responses = fallback_responses.get(character, fallback_responses["Detective"])
    response = char_responses.get(target_emotion, char_responses["engagement"])
    return response

def make_prompt(opposite_emotion, user_msg, memory_prompt="", character="Assistant", persona_key="CBT"):
    # Get the persona description from modal-persona.json
    personas = load_modal_personas()
    persona_description = personas.get(persona_key, personas.get("CBT", ""))
    
    # Get emotion directives
    directives = load_emotion_directives()
    
    # Pick a random directive for the target emotion
    directives_list = directives.get(opposite_emotion, [])
    if directives_list:
        random_directive = random.choice(directives_list)
    else:
        random_directive = f"Respond in a way that helps the user feel more {opposite_emotion}."

    # Build the full prompt
    prompt = f"""{persona_description}

    {memory_prompt}

    User ({character}): {user_msg}
    Assistant : {random_directive}
    """
    return prompt

def agent_response(user_message, detected_emotion, self_reported_emotion, character, persona_key=None):
    global _memory_store
    target_emotion = get_opposite_emotion(detected_emotion)
    session_id = session.get('session_id')
    memory = _memory_store.get(session_id)
    if memory is None:
        memory = ConversationMemory(session_id)
        _memory_store[session_id] = memory
    
    # Use the persona key from memory if not provided
    if persona_key is None:
        persona_key = memory.get_persona_key() or "CBT"  # Ensure persona_key is always a string

    memory_prompt = memory.get_history_prompt()
    prompt = make_prompt(target_emotion, user_message, memory_prompt, character, persona_key)
    response = openai_generate_response(prompt)
    if not response or response.startswith("["):
        response = get_fallback_response(character, target_emotion, user_message)
    memory.add_turn(user_message, response)
    _memory_store[session_id] = memory
    return response, target_emotion

# --- Routes ---
@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['history'] = []
    session_id = session['session_id']
    if session_id not in _memory_store:
        _memory_store[session_id] = ConversationMemory(session_id)
    return render_template('index.html', characters=CHARACTERS + ['Random'])

@app.route('/start', methods=['POST'])
def start_chat():
    data = request.json
    if data is None:
        return jsonify({'error': 'Invalid request data'}), 400
    selected_character = data.get('character')
    if selected_character == 'Random':
        session['character'] = random.choice(CHARACTERS)
    else:
        session['character'] = selected_character
    session_id = session.get('session_id')
    _memory_store[session_id] = ConversationMemory(session_id)
    return jsonify({
        'status': 'success',
        'character': session['character'],
        'message': 'Chat started.',
        'persona_key': 'Not set yet - will be determined by first emotion detected'
    })

@app.route('/send', methods=['POST'])
def send_message():
    # Clean up old memories periodically
    cleanup_old_memories()
    
    data = request.json
    if data is None:
        return jsonify({'error': 'Invalid request data'}), 400
    user_message = data.get('message')
    self_reported_emotion = data.get('emotion') or "Neutral"
    if not user_message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    # Get persona key from memory (emotion-based assignment)
    session_id = session.get('session_id')
    memory = _memory_store.get(session_id)
    if memory is None:
        memory = ConversationMemory(session_id)
        _memory_store[session_id] = memory
    
    # Detect emotion first
    detected_emotion, emotion_scores = classify_emotion(user_message)
    
    # Set persona key based on emotion if not already set
    if memory.get_persona_key() is None:
        persona_key = get_emotion_based_persona_key(detected_emotion)
        memory.set_persona_key(persona_key)
    else:
        persona_key = memory.get_persona_key()
    
    log_message('user', user_message, detected_emotion, self_reported_emotion, emotion_scores=emotion_scores, persona_key=persona_key)
    if 'history' not in session:
        session['history'] = []
    session['history'].append({
        'role': 'user',
        'message': user_message,
        'detected_emotion': detected_emotion,
        'self_reported_emotion': self_reported_emotion
    })
    character = session.get('character', 'Assistant')
    agent_msg, target_emotion = agent_response(user_message, detected_emotion, self_reported_emotion, character, persona_key)
    log_message('agent', agent_msg, target_emotion=target_emotion, persona_key=persona_key)
    session['history'].append({'role': 'agent', 'message': agent_msg})
    session.modified = True
    return jsonify({
        'agent_message': agent_msg,
        'detected_emotion': detected_emotion,
        'target_emotion': target_emotion,
        'emotion_scores': emotion_scores,
        'persona_key': persona_key
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002) 