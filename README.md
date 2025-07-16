# Emotion-Driven Chatbot UI

This is an enhanced chatbot interface that integrates emotion detection, conversation memory, and emotion-driven response generation using LLaMA models.

## Features

### 🧠 Emotion Detection
- **Automatic Emotion Classification**: Uses the `j-hartmann/emotion-english-distilroberta-base` model to automatically detect emotions in user messages
- **Self-Reported Emotions**: Users can manually report their emotional state
- **Dual Emotion Tracking**: Both detected and self-reported emotions are logged and displayed

### 💬 Emotion-Driven Responses
- **Opposite Emotion Targeting**: The system identifies the opposite of the detected emotion and generates responses to guide users toward that emotional state
- **Character-Based Responses**: Different character personas (Detective, Teacher, Robot, etc.) respond in character while targeting specific emotions
- **Conversation Memory**: Maintains context of recent conversation turns to generate more coherent responses

### 📊 Enhanced Logging
- **Comprehensive Data Collection**: Logs both detected and self-reported emotions, target emotions, and emotion confidence scores
- **Session Management**: Tracks conversation sessions with unique IDs
- **Research-Ready Format**: Data is stored in JSONL format for easy analysis

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the Emotion Detection System**:
   ```bash
   python test_emotion_detection.py
   ```

3. **Run the Chatbot**:
   ```bash
   python app.py
   ```

4. **Access the Interface**:
   Open your browser and go to `http://localhost:5001`

## System Architecture

### Core Components

1. **Emotion Classifier** (`load_emotion_classifier()`)
   - Uses Hugging Face transformers pipeline
   - Model: `j-hartmann/emotion-english-distilroberta-base`
   - Returns emotion labels and confidence scores

2. **LLaMA Generator** (`load_llama()`)
   - Uses Meta's LLaMA-2-7b-chat-hf model
   - Generates contextually appropriate responses
   - Targets specific emotional states

3. **Conversation Memory** (`ConversationMemory`)
   - Maintains recent conversation history
   - Provides context for response generation
   - Configurable memory length (default: 5 turns)

4. **Emotion Opposites Mapping**
   - Maps detected emotions to their opposites
   - Used to determine target emotional states for responses

### Data Flow

1. **User Input** → Emotion Detection → Self-Reported Emotion
2. **Detected Emotion** → Opposite Emotion Calculation → Target Emotion
3. **User Message + Target Emotion + Conversation History** → LLaMA Prompt
4. **LLaMA Response** → Logging → UI Display

## API Endpoints

### `POST /start`
Starts a new chat session with character selection.

**Request**:
```json
{
  "character": "Detective"
}
```

**Response**:
```json
{
  "status": "success",
  "character": "Detective",
  "message": "Chat started."
}
```

### `POST /send`
Sends a message and receives an emotion-driven response.

**Request**:
```json
{
  "message": "I'm feeling really angry today!",
  "emotion": "Angry"
}
```

**Response**:
```json
{
  "agent_message": "I understand you're feeling frustrated...",
  "detected_emotion": "anger",
  "target_emotion": "calm",
  "emotion_scores": [...]
}
```

## Data Logging

Messages are logged to `chat_logs.jsonl` with the following structure:

```json
{
  "timestamp": "2025-01-21T14:30:00.000000",
  "session_id": "uuid-string",
  "character": "Detective",
  "role": "user",
  "message": "User message",
  "detected_emotion": "anger",
  "self_reported_emotion": "Angry",
  "target_emotion": "calm",
  "emotion_scores": [...]
}
```

## Emotion Mapping

The system maps emotions to their opposites for response targeting:

| Detected Emotion | Target Emotion |
|------------------|----------------|
| anger | calm |
| joy | reflection |
| fear | reassurance |
| sadness | hope |
| surprise | predictability |
| neutral | engagement |

## UI Features

### Emotion Tags
- **Blue**: Self-reported emotions ("You felt: Angry")
- **Yellow**: AI-detected emotions ("AI detected: anger")
- **Green**: Target emotions for responses ("Targeting: calm")

### Character Selection
- Detective, Alien, Teacher, Robot, Pirate, Doctor, Wizard
- Random character selection option
- Character-specific response generation

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure you have sufficient RAM/VRAM
   - Check internet connection for model downloads
   - Verify transformers library installation

2. **CUDA/GPU Issues**:
   - The system automatically falls back to CPU if CUDA is unavailable
   - Check torch installation: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Memory Issues**:
   - Reduce `max_turns` in ConversationMemory class
   - Use smaller LLaMA model variants

### Performance Optimization

- **Model Caching**: Models are loaded once and cached in memory
- **Lazy Loading**: Models are only loaded when first needed
- **Error Handling**: Graceful fallbacks for model loading failures

## Research Applications

This system is designed for:
- **Emotion Regulation Studies**: Analyzing how AI responses affect user emotional states
- **Conversation Analysis**: Studying emotion patterns in human-AI interactions
- **Therapeutic Applications**: Exploring AI-assisted emotional support
- **Character Development**: Researching personality-driven AI responses

## License

This project is for research purposes. Please ensure compliance with model licenses and ethical guidelines when using this system. 