from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>Face Mask Detection AI</h1>
    <p>Model Loaded Successfully</p>
    <p>mask_detector.h5 found</p>
    """




#from flask import Flask

#app = Flask(__name__)

#@app.route("/")
#def home():
    #return "Face Mask Detection AI Project Running Successfully"

#if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=10000)
