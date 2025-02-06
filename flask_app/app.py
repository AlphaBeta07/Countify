from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

class ThreadCounter:
    def __init__(self, cloth_width_inch, cloth_height_inch, dpi=300):
        self.cloth_width_inch = cloth_width_inch
        self.cloth_height_inch = cloth_height_inch
        self.dpi = dpi
        self.min_contour_area = 50  # Minimum area to consider a contour as a thread

    def count_threads_in_cloth(self, image):
        processed_image = self.preprocess_image(image)
        contours = self.detect_contours(processed_image)
        
        lines = self.detect_lines(processed_image)
        thread_count, vertical_count, horizontal_count = self.analyze_contours(contours, lines)

        threads_per_sq_inch = self.calculate_threads_per_sq_inch(thread_count)

        return thread_count, vertical_count, horizontal_count, threads_per_sq_inch

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
        return edges

    def detect_contours(self, edges):
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def detect_lines(self, edges):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        return lines

    def analyze_contours(self, contours, lines):
        thread_count = 0
        vertical_count = 0
        horizontal_count = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 45:  # Horizontal threads
                    horizontal_count += 1
                else:  # Vertical threads
                    vertical_count += 1
                thread_count += 1

        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                thread_count += 1
                if len(contour) >= 5:  
                    ellipse = cv2.fitEllipse(contour)
                    angle = ellipse[2]
                    if 45 < angle < 135:  # Vertical threads
                        vertical_count += 1
                    else:  # Horizontal threads
                        horizontal_count += 1

        return thread_count, vertical_count, horizontal_count

    def calculate_threads_per_sq_inch(self, thread_count):
        cloth_area_sq_inch = self.cloth_width_inch * self.cloth_height_inch
        return thread_count / cloth_area_sq_inch if cloth_area_sq_inch > 0 else 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process image
        image = cv2.imread(file_path)
        cloth_width_inch = 1  # Assume appropriate cloth dimensions
        cloth_height_inch = 1
        
        thread_counter = ThreadCounter(cloth_width_inch, cloth_height_inch)
        thread_count, vertical_count, horizontal_count, threads_per_sq_inch = thread_counter.count_threads_in_cloth(image)

        return jsonify({
            "thread_count": thread_count,
            "vertical_count": vertical_count,
            "horizontal_count": horizontal_count,
            "threads_per_sq_inch": threads_per_sq_inch
        })

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True)