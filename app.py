from flask import Flask, render_template, send_from_directory
import os
from forecasting_script import run_forecasting_pipeline  # This should save plot to static/plots/forecast.png

app = Flask(__name__)

@app.route('/')
def index():
    run_forecasting_pipeline()  # Generate the forecast and save the graph
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render assigns a port in PORT
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
