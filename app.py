from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Face Mask Detection AI Project Running Successfully"
