"""
Flask web application for watermark detection.

Run with: python web_app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detector_utils import WatermarkDetector

app = Flask(__name__, 
            template_folder='web',
            static_folder='web/static')
CORS(app)

# Initialize detector once at startup (more efficient)
print("Loading watermark detector...")
try:
    detector = WatermarkDetector(ngram_len=5)
    print("✅ Detector loaded and ready!")
except Exception as e:
    print("⚠️ Failed to load real watermark detector:", e)
    print("➡️ Falling back to stub detector (scores are pseudo-random, NOT real).")
    import hashlib, random
    class WatermarkDetectorStub:
        def __init__(self, ngram_len=5):
            self.ngram_len = ngram_len
            self.device = 'cpu'
        def detect(self, code: str, threshold: float = 0.5) -> dict:
            if not code or not code.strip():
                score = 0.0
            else:
                h = hashlib.sha256(code.encode()).hexdigest()
                # Deterministic pseudo-score based on hash
                score = (int(h[:8], 16) % 1000) / 1000.0
            is_watermarked = score >= threshold
            if score < 0.3:
                confidence = 'low'
            elif score < 0.7:
                confidence = 'medium'
            else:
                confidence = 'high'
            return {
                'is_watermarked': is_watermarked,
                'score': score,
                'confidence': confidence,
                'threshold': threshold,
                'ngram_len': self.ngram_len,
                'stub': True
            }
    detector = WatermarkDetectorStub(ngram_len=5)


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect_watermark():
    """
    API endpoint to detect watermarks in code.
    
    Request JSON:
    {
        "code": "def hello(): print('hi')",
        "threshold": 0.5  // optional, defaults to 0.5
    }
    
    Response JSON:
    {
        "success": true,
        "result": {
            "is_watermarked": false,
            "score": 0.1234,
            "confidence": "low",
            "threshold": 0.5,
            "ngram_len": 5
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'code' not in data:
            return jsonify({
                'success': False,
                'error': 'No code provided'
            }), 400
        
        code = data['code']
        threshold = data.get('threshold', 0.5)
        
        # Validate threshold
        if not (0 <= threshold <= 1):
            return jsonify({
                'success': False,
                'error': 'Threshold must be between 0 and 1'
            }), 400
        
        # Detect watermark
        result = detector.detect(code, threshold=threshold)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check if the detector is ready."""
    return jsonify({
        'status': 'ready',
        'detector': {
            'ngram_len': detector.ngram_len,
            'device': str(detector.device)
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Watermark Detection Web App")
    print("="*60)
    print(f"📍 Open in browser: http://localhost:5001")
    print("="*60 + "\n")
    
    # Disable debug auto-reload to keep single stable process for stub fallback
    app.run(debug=False, host='0.0.0.0', port=5001)
