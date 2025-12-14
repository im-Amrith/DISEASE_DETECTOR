document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const fileNameDisplay = document.getElementById('file-name');
    const uploadForm = document.getElementById('upload-form');
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const predictionText = document.getElementById('prediction-text');
    const imagePreviewDiv = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');

    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            fileNameDisplay.textContent = `SELECTED: ${file.name.toUpperCase()}`;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImg.src = e.target.result;
                imagePreviewDiv.classList.remove('hidden');
            }
            reader.readAsDataURL(file);
            
            // Reset results
            resultDiv.classList.add('hidden');
        } else {
            fileNameDisplay.textContent = 'NO FILE SELECTED';
            imagePreviewDiv.classList.add('hidden');
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (fileInput.files.length === 0) {
            alert('PLEASE SELECT A FILE FIRST');
            return;
        }

        const formData = new FormData(uploadForm);
        
        // Show loading
        loadingDiv.classList.remove('hidden');
        resultDiv.classList.add('hidden');
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                predictionText.textContent = data.prediction.toUpperCase();
                resultDiv.classList.remove('hidden');
            } else {
                predictionText.textContent = `ERROR: ${data.error}`;
                resultDiv.classList.remove('hidden');
            }
        } catch (error) {
            console.error('Error:', error);
            predictionText.textContent = 'SYSTEM ERROR: CONNECTION FAILED';
            resultDiv.classList.remove('hidden');
        } finally {
            loadingDiv.classList.add('hidden');
        }
    });
});
