from flask import Flask
import glob
import os
from forecasting_script import run_forecasting_pipeline

app = Flask(__name__)

@app.route('/')
def index():
    run_forecasting_pipeline()
    files = glob.glob("static/plots/*.png")
    links = ''.join([f'<li><a href="/{file}">{file}</a></li>' for file in files])
    return f"<h1>Forecast Completed</h1><ul>{links}</ul>"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
