"""
This script sets up the Flask backend for the application.
It handles:
1. User interactions via HTTP routes for messages and file uploads.
2. Integration with the worker module for processing user messages and documents.
3. Serving the chatbot's front-end interface.

Run this file to start the server and enable the chatbot functionality.
"""

import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import worker  # Import the worker module for document and message processing

# Initialize Flask application and enable Cross-Origin Resource Sharing (CORS)
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from any origin
app.logger.setLevel(logging.ERROR)  # Suppress detailed error logs

# Route: Render the index page
@app.route('/', methods=['GET'])
def index():
    """
    Load the front-end interface for the chatbot.
    Returns:
        HTML template for the chatbot interface.
    """
    return render_template('index.html')  # Ensure `index.html` exists in the `templates/` directory

# Route: Process user messages
@app.route('/process-message', methods=['POST'])
def process_message_route():
    """
    Handle user messages and return responses from the chatbot.
    Expects:
        JSON payload with "userMessage" key containing the user's message.
    Returns:
        JSON response with the chatbot's reply.
    """
    user_message = request.json['userMessage']  # Extract the user's message
    print('user_message:', user_message)  # Debugging log (can be removed or adjusted)

    bot_response = worker.process_prompt(user_message)  # Use `worker` module to process the message

    return jsonify({"botResponse": bot_response}), 200  # Return chatbot's response

# Route: Process uploaded PDF documents
@app.route('/process-document', methods=['POST'])
def process_document_route():
    """
    Handle PDF file uploads and process the document for future chatbot queries.
    Returns:
        Success message or error if the file upload fails.
    """
    if 'file' not in request.files:
        return jsonify({
            "botResponse": "It seems like the file was not uploaded correctly. Please try again with a valid file."
        }), 400

    file = request.files['file']  # Extract the uploaded file
    file_path = file.filename  # Save the file using its original name (can be updated for security)
    file.save(file_path)  # Save the file locally

    worker.process_document(file_path)  # Process the document for embeddings and query preparation

    return jsonify({
        "botResponse": "Thank you for providing your PDF document. You can now ask me any questions regarding its content!"
    }), 200

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')  # Debugging enabled for development
