from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import os
from pathlib import Path
import re

# Add utils to path
sys.path.append(str(Path(__file__).parent))

# Import your existing G2PPredictor
from utils.g2p_inference import G2PPredictor

# Initialize FastAPI
app = FastAPI(
    title="Ga G2P API",
    description="Grapheme-to-Phoneme conversion for Ga language",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - allow all origins (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None


@app.on_event("startup")
async def load_model():
    """Load G2P model on startup"""
    global predictor
    
    model_path = Path(__file__).parent / "models" / "final" / "G2P_best_model.pt"
    
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}")
    
    try:
        predictor = G2PPredictor(str(model_path), device=None, verbose=True)
        print(f"✓ G2P model loaded successfully from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


# Request/Response Models
class G2PRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="Ga text to convert")
    preserve_punctuation: bool = Field(default=True, description="Preserve punctuation in output")


class WordBreakdown(BaseModel):
    word: str
    phonemes: str


class G2PResponse(BaseModel):
    success: bool
    input_sentence: str
    word_count: int
    total_phonemes: int
    sentence_phonemes: str
    word_breakdown: List[WordBreakdown]
    error: Optional[str] = None


# Helper functions
def is_punctuation(text: str) -> bool:
    """Check if text is punctuation or whitespace"""
    punctuation_set = set('.,;:!?\'"()[]{}…—–-<>/\\|@#$%^&*+=~` \t\n')
    return all(c in punctuation_set for c in text)


def tokenize_with_punctuation(text: str) -> List[dict]:
    """
    Tokenize text preserving punctuation as separate tokens
    
    Returns:
        List of dicts with 'text' and 'type' ('word' or 'punctuation')
    """
    tokens = []
    current_word = []
    
    for char in text:
        if char in '.,;:!?\'"()[]{}…—–-' or char.isspace():
            # Save accumulated word
            if current_word:
                tokens.append({
                    'text': ''.join(current_word),
                    'type': 'word'
                })
                current_word = []
            
            # Add punctuation/whitespace
            tokens.append({
                'text': char,
                'type': 'punctuation'
            })
        else:
            current_word.append(char)
    
    # Add final word if exists
    if current_word:
        tokens.append({
            'text': ''.join(current_word),
            'type': 'word'
        })
    
    return tokens


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Ga G2P API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "convert": "POST /api/g2p",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.post("/api/g2p", response_model=G2PResponse)
async def convert_g2p(request: G2PRequest):
    """
    Convert Ga text to phonemes with punctuation preservation
    
    Args:
        request: G2PRequest with text and options
        
    Returns:
        G2PResponse with phonemes and word breakdown
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input preserving punctuation
        tokens = tokenize_with_punctuation(request.text)
        
        # Process each token
        word_breakdown = []
        all_phonemes_parts = []
        word_count = 0
        total_phonemes = 0
        
        for token in tokens:
            if token['type'] == 'punctuation':
                # Preserve punctuation as-is
                word_breakdown.append(WordBreakdown(
                    word=token['text'],
                    phonemes=token['text']
                ))
                all_phonemes_parts.append(token['text'])
            else:
                # This is a word - get phonemes
                word = token['text']
                
                try:
                    word_phonemes = predictor.predict(
                        word,
                        preserve_punctuation=False  # Single word processing
                    )
                    
                    word_breakdown.append(WordBreakdown(
                        word=word,
                        phonemes=word_phonemes
                    ))
                    
                    if word_phonemes:
                        all_phonemes_parts.append(word_phonemes)
                        # Count phonemes in this word
                        total_phonemes += len([p for p in word_phonemes.split() if p.strip()])
                    
                    word_count += 1
                    
                except Exception as e:
                    # On error, keep original word
                    word_breakdown.append(WordBreakdown(
                        word=word,
                        phonemes=word
                    ))
                    all_phonemes_parts.append(word)
        
        # Join all parts to create sentence phonemes
        sentence_phonemes = ''.join(all_phonemes_parts)
        
        return G2PResponse(
            success=True,
            input_sentence=request.text,
            word_count=word_count,
            total_phonemes=total_phonemes,
            sentence_phonemes=sentence_phonemes,
            word_breakdown=word_breakdown,
            error=None
        )
        
    except Exception as e:
        return G2PResponse(
            success=False,
            input_sentence=request.text,
            word_count=0,
            total_phonemes=0,
            sentence_phonemes="",
            word_breakdown=[],
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)