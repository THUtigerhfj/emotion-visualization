// DOM Elements
const imageUpload = document.getElementById('imageUpload');
const uploadLabel = document.getElementById('uploadLabel');
const fileName = document.getElementById('fileName');
const runBtn = document.getElementById('runBtn');
const btnText = runBtn.querySelector('.btn-text');
const btnLoader = runBtn.querySelector('.btn-loader');
const imageContainer = document.getElementById('imageContainer');
const faceCount = document.getElementById('faceCount');
const facesScroll = document.getElementById('facesScroll');
const summaryContent = document.getElementById('summaryContent');

// State
let uploadedImageData = null;

// Initialize radio button styling
document.querySelectorAll('.radio-option').forEach(option => {
    const input = option.querySelector('input');
    input.addEventListener('change', () => {
        document.querySelectorAll('.radio-option').forEach(opt => {
            opt.classList.remove('selected');
        });
        option.classList.add('selected');
    });
    if (input.checked) {
        option.classList.add('selected');
    }
});

// File Upload Handler
imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    fileName.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (event) => {
        uploadedImageData = event.target.result;
        runBtn.disabled = false;
    };
    reader.readAsDataURL(file);
});

// Drag and Drop Support
uploadLabel.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadLabel.style.background = 'var(--primary-hover)';
});

uploadLabel.addEventListener('dragleave', () => {
    uploadLabel.style.background = 'var(--primary)';
});

uploadLabel.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadLabel.style.background = 'var(--primary)';

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        fileName.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (event) => {
            uploadedImageData = event.target.result;
            runBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
});

// Run Inference
runBtn.addEventListener('click', async () => {
    if (!uploadedImageData) return;

    // Show loading state
    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');
    runBtn.disabled = true;

    try {
        const mode = document.querySelector('input[name="mode"]:checked').value;

        const response = await fetch('/api/inference', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: uploadedImageData,
                mode: mode
            })
        });

        const result = await response.json();

        if (result.success) {
            displayResults(result);
        } else {
            showError(result.error || 'An error occurred during inference');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        // Reset button state
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
        runBtn.disabled = false;
    }
});

// Display Results
function displayResults(result) {
    // Update image container with boxed image
    imageContainer.innerHTML = `
        <img src="data:image/png;base64,${result.boxed_image}" alt="Detected faces">
    `;

    // Update face count
    faceCount.textContent = `${result.faces_detected} face${result.faces_detected !== 1 ? 's' : ''}`;

    // Display faces
    facesScroll.innerHTML = '';
    if (result.faces.length > 0) {
        result.faces.forEach(face => {
            const card = createFaceCard(face);
            facesScroll.appendChild(card);
        });
    } else {
        facesScroll.innerHTML = `
            <div class="placeholder">
                <p>No faces detected</p>
            </div>
        `;
    }

    // Display summary
    if (result.summary) {
        summaryContent.textContent = result.summary;
    } else {
        summaryContent.innerHTML = `
            <div class="placeholder">
                <p>No results</p>
            </div>
        `;
    }
}

// Create Face Card
function createFaceCard(face) {
    const card = document.createElement('div');
    card.className = 'face-card';

    const emotionClass = face.emotion.toLowerCase();

    card.innerHTML = `
        <div class="face-card-image">
            <img src="data:image/png;base64,${face.crop}" alt="Face ${face.index}">
        </div>
        <div class="face-card-attention">
            <img src="data:image/png;base64,${face.attention}" alt="Attention visualization">
        </div>
        <div class="face-card-content">
            <div class="face-card-header">
                <span class="face-badge">Face ${face.index}</span>
                <span class="emotion-badge ${emotionClass}">${face.emotion}</span>
            </div>
            <span class="confidence">Confidence: ${face.confidence}</span>
        </div>
    `;

    return card;
}

// Show Error
function showError(message) {
    const existingError = document.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }

    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;

    const controls = document.querySelector('.controls');
    controls.parentNode.insertBefore(errorDiv, controls.nextSibling);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}
