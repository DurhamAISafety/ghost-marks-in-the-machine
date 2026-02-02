// DOM Elements
const codeInput = document.getElementById('code-input');
const thresholdSlider = document.getElementById('threshold');
const thresholdValue = document.getElementById('threshold-value');
const detectBtn = document.getElementById('detect-btn');
const clearBtn = document.getElementById('clear-btn');
const exampleBtn = document.getElementById('example-btn');

const resultContainer = document.getElementById('result-container');
const loadingContainer = document.getElementById('loading');
const errorContainer = document.getElementById('error-container');

// Result elements
const statusBadge = document.getElementById('status-badge');
const isWatermarked = document.getElementById('is-watermarked');
const scoreElement = document.getElementById('score');
const confidenceElement = document.getElementById('confidence');
const ngramElement = document.getElementById('ngram');
const progressFill = document.getElementById('progress-fill');
const thresholdMarker = document.getElementById('threshold-marker');
const errorMessage = document.getElementById('error-message');

// Example codes
const examples = [
    `def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)`,

    `def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)`,

    `class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(root.value)
        inorder_traversal(root.right)`
];

let currentExampleIndex = 0;

// Update threshold display
thresholdSlider.addEventListener('input', (e) => {
    const value = (e.target.value / 100).toFixed(2);
    thresholdValue.textContent = value;

    // Update marker position
    thresholdMarker.style.left = `${e.target.value}%`;
});

// Clear button
clearBtn.addEventListener('click', () => {
    codeInput.value = '';
    hideAll();
});

// Example button
exampleBtn.addEventListener('click', () => {
    codeInput.value = examples[currentExampleIndex];
    currentExampleIndex = (currentExampleIndex + 1) % examples.length;
    hideAll();
});

// Detect button
detectBtn.addEventListener('click', async () => {
    const code = codeInput.value.trim();

    if (!code) {
        showError('Please enter some code to analyze');
        return;
    }

    const threshold = parseFloat(thresholdValue.textContent);

    // Show loading
    hideAll();
    loadingContainer.classList.remove('hidden');
    detectBtn.disabled = true;

    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ code, threshold })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Detection failed');
        }

        displayResult(data.result);

    } catch (error) {
        showError(error.message);
    } finally {
        loadingContainer.classList.add('hidden');
        detectBtn.disabled = false;
    }
});

// Display result
function displayResult(result) {
    hideAll();

    // Status badge
    if (result.is_watermarked) {
        statusBadge.textContent = '⚠️ Watermarked';
        statusBadge.className = 'status-badge watermarked';
    } else {
        statusBadge.textContent = '✓ Not Watermarked';
        statusBadge.className = 'status-badge not-watermarked';
    }

    // Values
    isWatermarked.textContent = result.is_watermarked ? 'Yes' : 'No';
    isWatermarked.style.color = result.is_watermarked ? 'var(--danger)' : 'var(--success)';

    scoreElement.textContent = result.score.toFixed(4);

    confidenceElement.textContent = result.confidence.charAt(0).toUpperCase() + result.confidence.slice(1);
    confidenceElement.style.color = getConfidenceColor(result.confidence);

    ngramElement.textContent = result.ngram_len;

    // Progress bar
    const percentage = (result.score * 100).toFixed(1);
    progressFill.style.width = `${percentage}%`;

    resultContainer.classList.remove('hidden');
}

// Get confidence color
function getConfidenceColor(confidence) {
    switch (confidence.toLowerCase()) {
        case 'low': return 'var(--success)';
        case 'medium': return 'var(--warning)';
        case 'high': return 'var(--danger)';
        default: return 'var(--text)';
    }
}

// Show error
function showError(message) {
    hideAll();
    errorMessage.textContent = message;
    errorContainer.classList.remove('hidden');
}

// Hide all results/loading/errors
function hideAll() {
    resultContainer.classList.add('hidden');
    loadingContainer.classList.add('hidden');
    errorContainer.classList.add('hidden');
}

// Initialize threshold marker position
thresholdMarker.style.left = '50%';

// Check API status on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        console.log('Detector status:', data);
    } catch (error) {
        console.error('Failed to check detector status:', error);
    }
});
