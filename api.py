from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import shutil
import os
from main import main1

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get_test_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Clothing Classification API</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 600px; 
                margin: 50px auto; 
                padding: 20px;
            }
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
            }
            .result {
                background: #f5f5f5;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }
            button {
                background: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background: #0056b3;
            }
            #loading {
                display: none;
                text-align: center;
                color: #666;
            }
        </style>
    </head>
    <body>
        <h1> Clothing Classification API Demo</h1>
        <p>Upload a clothing image and see what AI thinks it is!</p>
        
        <div class="upload-area">
            <input type="file" id="imageInput" accept="image/*" style="display: none;">
            <button onclick="document.getElementById('imageInput').click()">
                Choose Image
            </button>
            <p>Or drag and drop an image here</p>
        </div>
        
        <div id="loading">
            <p> AI is analyzing your image...</p>
        </div>
        
        <div id="result" class="result" style="display: none;">
            <h3>Results:</h3>
            <div id="resultContent"></div>
        </div>

        <script>
            const API_URL = window.location.origin + '/predict/';
            
            document.getElementById('imageInput').addEventListener('change', function(event) {
                const file = event.target.files[0];
                if (file) {
                    uploadImage(file);
                }
            });

            // Drag and drop functionality
            const uploadArea = document.querySelector('.upload-area');
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#007bff';
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.style.borderColor = '#ccc';
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#ccc';
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    uploadImage(files[0]);
                }
            });

            async function uploadImage(file) {
                const formData = new FormData();
                formData.append('file', file);
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                try {
                    const response = await fetch(API_URL, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    displayResult(result);
                    
                } catch (error) {
                    document.getElementById('resultContent').innerHTML = 
                        `<p style="color: red;">Error: ${error.message}</p>`;
                    document.getElementById('result').style.display = 'block';
                }
                
                document.getElementById('loading').style.display = 'none';
            }

            function displayResult(data) {
                const resultDiv = document.getElementById('resultContent');
                const result = data.result;
                
                resultDiv.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div style="background: #e8f4f8; padding: 15px; border-radius: 8px;">
                            <h4> Clothing Type</h4>
                            <p><strong>${result.clothing.best_match}</strong></p>
                            <p>Confidence: ${(result.clothing.confidence * 100).toFixed(1)}%</p>
                        </div>
                        <div style="background: #f8e8f4; padding: 15px; border-radius: 8px;">
                            <h4> Color</h4>
                            <p><strong>${result.color.best_match}</strong></p>
                            <p>Confidence: ${(result.color.confidence * 100).toFixed(1)}%</p>
                        </div>
                    </div>
                    <div style="background: #e7f3ff; padding: 10px; border-radius: 5px; margin-top: 15px; text-align: center;">
                        <small>Powered by CLIP AI Model </small>
                    </div>
                `;
                
                document.getElementById('result').style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        results = main1(temp_file_path)  # This now returns both clothing and color
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)