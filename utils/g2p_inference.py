import torch
import torch.nn as nn
import json
import os
import argparse
import re
from typing import List, Union, Dict, Optional
import warnings


class G2PModel(nn.Module):
    """G2P Model for inference"""
    
    def __init__(self, char_vocab_size, phoneme_vocab_size, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.char_vocab_size = char_vocab_size
        self.phoneme_vocab_size = phoneme_vocab_size
        
        # Embeddings with proper scaling
        self.char_embedding = nn.Embedding(char_vocab_size, d_model)
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, d_model)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.char_embedding.weight)
        nn.init.xavier_uniform_(self.phoneme_embedding.weight)
        
        # Positional encoding - registered as buffer
        self.register_buffer('pos_encoding', self.create_positional_encoding(1000, d_model))
        
        # Scale factor - registered as buffer
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([d_model])))
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, phoneme_vocab_size)
        nn.init.xavier_uniform_(self.output_projection.weight)
        
        self.dropout = nn.Dropout(dropout)
    
    def create_positional_encoding(self, max_len, d_model):
        """Create positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, src, tgt):
        """Forward pass"""
        # Source (character) encoding
        src_embedded = self.char_embedding(src) * self.scale
        src_embedded = src_embedded + self.pos_encoding[:src_embedded.size(1)]
        src_embedded = self.dropout(src_embedded)
        
        # Target (phoneme) encoding
        tgt_embedded = self.phoneme_embedding(tgt) * self.scale
        tgt_embedded = tgt_embedded + self.pos_encoding[:tgt_embedded.size(1)]
        tgt_embedded = self.dropout(tgt_embedded)
        
        # Create target mask (causal mask)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Create padding masks
        src_key_padding_mask = (src == 0)  # PAD = 0
        tgt_key_padding_mask = (tgt == 0)
        
        # Transformer forward
        output = self.transformer(
            src_embedded, 
            tgt_embedded, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def generate(self, src, max_len=50):
        """Generate phoneme sequence from character sequence (greedy decoding)"""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Start with SOS token
        tgt = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)  # 2 = <SOS>
        
        with torch.no_grad():
            for _ in range(max_len):
                # Get predictions
                logits = self.forward(src, tgt)
                
                # Get next token (greedy)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
                # Append to sequence
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Stop if all sequences have generated EOS (3 = <EOS>)
                if (next_token == 3).all():
                    break
        
        return tgt


class G2PPredictor:
    """Production-ready G2P inference system with punctuation handling"""
    
    # Define punctuation and special characters that should be preserved
    PUNCTUATION = set('.,;:!?\'"-()[]{}…—–<>/\\|@#$%^&*+=~`')
    
    def __init__(self, model_path: str, device: Optional[str] = None, verbose: bool = True):
        """
        Initialize G2P predictor
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            verbose: Whether to print loading information
        """
        
        # Device selection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if verbose:
            print(f"Initializing G2P predictor on {self.device}")
            if torch.cuda.is_available() and self.device.type == 'cuda':
                print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Suppress warnings for cleaner inference
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Load model
        self._load_model(model_path, verbose)
        
        if verbose:
            print("G2P predictor ready for inference (punctuation-aware)")
    
    def _load_model(self, model_path: str, verbose: bool):
        """Load model and vocabularies"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load vocabularies
        self.char_vocab = checkpoint['char_vocab']
        self.phoneme_vocab = checkpoint['phoneme_vocab']
        
        # Create reverse vocabularies for decoding
        self.idx_to_char = {v: k for k, v in self.char_vocab.items()}
        self.idx_to_phoneme = {v: k for k, v in self.phoneme_vocab.items()}
        
        # Initialize model
        model_config = checkpoint['model_config']
        self.model = G2PModel(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Store model info
        self.model_info = {
            'epoch': checkpoint['epoch'],
            'val_loss': checkpoint['val_loss'],
            'config': model_config
        }
        
        if verbose:
            print(f"Loaded model from epoch {self.model_info['epoch']} (val_loss: {self.model_info['val_loss']:.4f})")
    
    def _is_punctuation(self, char: str) -> bool:
        """Check if a character is punctuation or whitespace"""
        return char in self.PUNCTUATION or char.isspace()
    
    def _tokenize_with_punctuation(self, text: str) -> List[Dict[str, str]]:
        """
        Tokenize text into words and punctuation
        
        Args:
            text: Input text
            
        Returns:
            List of tokens with type ('word' or 'punctuation')
        """
        tokens = []
        current_word = []
        
        for char in text:
            if self._is_punctuation(char):
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
    
    def encode_word(self, word: str) -> torch.Tensor:
        """Encode a word to character indices"""
        word = word.strip().lower()
        
        if not word:
            raise ValueError("Empty word provided")
        
        tokens = list(word)
        encoded = [self.char_vocab['<SOS>']]
        
        for char in tokens:
            encoded.append(self.char_vocab.get(char, self.char_vocab['<UNK>']))
        
        encoded.append(self.char_vocab['<EOS>'])
        return torch.tensor([encoded], dtype=torch.long, device=self.device)
    
    def decode_phonemes(self, phoneme_indices: torch.Tensor) -> str:
        """Decode phoneme indices to phoneme string"""
        phonemes = []
        
        for idx in phoneme_indices:
            idx_val = idx.item()
            
            if idx_val == self.phoneme_vocab['<SOS>']:
                continue
            elif idx_val == self.phoneme_vocab['<EOS>']:
                break
            elif idx_val == self.phoneme_vocab['<PAD>']:
                continue
            else:
                phoneme = self.idx_to_phoneme.get(idx_val, '<UNK>')
                if phoneme != '<UNK>':  # Skip unknown tokens
                    phonemes.append(phoneme)
        
        return ' '.join(phonemes)
    
    def predict(self, text: str, max_len: int = 50, preserve_punctuation: bool = True) -> str:
        """
        Predict phonemes for text (word or sentence)
        
        Args:
            text: Input text to convert to phonemes
            max_len: Maximum length of generated phoneme sequence per word
            preserve_punctuation: If True, preserve punctuation in output
            
        Returns:
            Predicted phoneme sequence with preserved punctuation
        """
        
        if not isinstance(text, str):
            raise TypeError(f"Text must be string, got {type(text)}")
        
        text = text.strip()
        if not text:
            raise ValueError("Empty text provided")
        
        # If preserve_punctuation is False, treat as single word
        if not preserve_punctuation:
            return self._predict_word(text, max_len)
        
        # Tokenize text into words and punctuation
        tokens = self._tokenize_with_punctuation(text)
        
        result_parts = []
        
        with torch.no_grad():
            for token in tokens:
                if token['type'] == 'punctuation':
                    # Preserve punctuation as-is
                    result_parts.append(token['text'])
                else:
                    # Predict phonemes for word
                    word = token['text']
                    if word:  # Only process non-empty words
                        try:
                            phonemes = self._predict_word(word, max_len)
                            result_parts.append(phonemes)
                        except Exception as e:
                            # On error, keep original word
                            result_parts.append(word)
        
        return ''.join(result_parts)
    
    def _predict_word(self, word: str, max_len: int = 50) -> str:
        """
        Predict phonemes for a single word (internal method)
        
        Args:
            word: Input word to convert to phonemes
            max_len: Maximum length of generated phoneme sequence
            
        Returns:
            Predicted phoneme sequence as space-separated string
        """
        
        word = word.strip()
        if not word:
            return ''
        
        with torch.no_grad():
            # Encode input
            try:
                input_ids = self.encode_word(word)
            except Exception as e:
                raise ValueError(f"Failed to encode word '{word}': {e}")
            
            # Generate phonemes
            output_ids = self.model.generate(input_ids, max_len=max_len)
            
            # Decode output
            predicted_phonemes = self.decode_phonemes(output_ids[0])
            
            return predicted_phonemes
    
    def predict_batch(self, texts: List[str], max_len: int = 50, preserve_punctuation: bool = True) -> List[Dict[str, str]]:
        """
        Predict phonemes for multiple texts
        
        Args:
            texts: List of texts to convert
            max_len: Maximum length of generated phoneme sequences
            preserve_punctuation: If True, preserve punctuation in output
            
        Returns:
            List of dictionaries with 'text' and 'phonemes' keys
        """
        
        results = []
        
        for text in texts:
            try:
                phonemes = self.predict(text, max_len, preserve_punctuation)
                results.append({
                    'text': text,
                    'phonemes': phonemes,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'text': text,
                    'phonemes': '',
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_info': self.model_info,
            'vocabularies': {
                'char_vocab_size': len(self.char_vocab),
                'phoneme_vocab_size': len(self.phoneme_vocab)
            },
            'device': str(self.device)
        }


if __name__ == "__main__":
    # Standalone CLI functionality
    parser = argparse.ArgumentParser(description='G2P Inference Script')
    parser.add_argument('--model_path', type=str, default='models/final/G2P_best_model.pt')
    parser.add_argument('--text', type=str, help='Text to convert')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    predictor = G2PPredictor(args.model_path, device=args.device)
    if args.text:
        result = predictor.predict(args.text)
        print(result)