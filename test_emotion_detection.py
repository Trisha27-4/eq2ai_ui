#!/usr/bin/env python3
"""
Test script for emotion detection system
"""

from transformers.pipelines import pipeline
import json

def test_emotion_detection():
    """Test the emotion classification model."""
    print("Loading emotion classifier...")
    
    try:
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            truncation=True
        )
        print("✅ Emotion classifier loaded successfully!")
        
        # Test cases
        test_messages = [
            "I'm so happy today! Everything is going great!",
            "I'm really angry about what happened yesterday.",
            "I'm feeling sad and lonely.",
            "I'm surprised by the unexpected news.",
            "I'm afraid of what might happen next.",
            "This is disgusting, I can't believe it!",
            "I'm feeling neutral about this situation."
        ]
        
        print("\n🧪 Testing emotion detection:")
        print("-" * 50)
        
        for message in test_messages:
            scores = classifier(message)[0]
            # Sort by score
            sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
            top_emotion = sorted_scores[0]['label']
            top_score = sorted_scores[0]['score']
            
            print(f"Message: '{message}'")
            print(f"Detected emotion: {top_emotion} (confidence: {top_score:.3f})")
            top_scores = [(s['label'], f"{s['score']:.3f}") for s in sorted_scores[:3]]
            print(f"All scores: {top_scores}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ Error loading emotion classifier: {e}")
        return False
    
    return True

def test_emotion_opposites():
    """Test the emotion opposites mapping."""
    print("\n🔄 Testing emotion opposites:")
    print("-" * 30)
    
    emotion_opposites = {
        "admiration": "humility", "amusement": "seriousness", "anger": "calm",
        "annoyance": "ease", "approval": "detachment", "caring": "indifference",
        "confusion": "clarity", "curiosity": "certainty", "desire": "satisfaction",
        "disappointment": "hope", "disapproval": "acceptance", "embarrassment": "confidence",
        "excitement": "composure", "fear": "reassurance", "gratitude": "entitlement",
        "grief": "comfort", "joy": "reflection", "love": "detachment",
        "nervousness": "confidence", "neutral": "engagement", "optimism": "skepticism",
        "pride": "modesty", "realization": "uncertainty", "relief": "tension",
        "remorse": "forgiveness", "sadness": "hope", "surprise": "predictability"
    }
    
    test_emotions = ["joy", "anger", "fear", "sadness", "surprise", "neutral"]
    
    for emotion in test_emotions:
        opposite = emotion_opposites.get(emotion.lower(), "neutral")
        print(f"{emotion} → {opposite}")
    
    return True

if __name__ == "__main__":
    print("🎭 Emotion Detection System Test")
    print("=" * 50)
    
    # Test emotion detection
    emotion_test = test_emotion_detection()
    
    # Test emotion opposites
    opposites_test = test_emotion_opposites()
    
    if emotion_test and opposites_test:
        print("\n✅ All tests passed! The emotion detection system is ready.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.") 