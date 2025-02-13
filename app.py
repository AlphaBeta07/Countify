from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

#Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreadCounter:
    def __init__(self, cloth_width_inch, cloth_height_inch, dpi=300):
        self.cloth_width_inch = cloth_width_inch
        self.cloth_height_inch = cloth_height_inch
        self.dpi = dpi
        self.min_contour_area = 50  #minimum area to consider a contour as a thread
        self.line_length_threshold = 50  #minimum length to consider a line as a thread

    def count_threads_in_cloth(self, image):
        try:
            processed_image = self.preprocess_image(image)
            contours = self.detect_contours(processed_image)
            lines = self.detect_lines(processed_image)
            thread_count, vertical_count, horizontal_count = self.analyze_contours_and_lines(contours, lines)
            threads_per_sq_inch = self.calculate_threads_per_sq_inch(thread_count)

            return thread_count, vertical_count, horizontal_count, threads_per_sq_inch
        except Exception as e:
            logger.error(f"Error in counting threads: {e}")
            raise

    def preprocess_image(self, image):
        #Convert to grayscale, apply histogram equalization, blur, and detect edges 
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
            return edges
        except Exception as e:
            logger.error(f"Error in preprocessing image: {e}")
            raise

    def detect_contours(self, edges):
        #Detect contours in the image
        try:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        except Exception as e:
            logger.error(f"Error in detecting contours: {e}")
            raise

    def detect_lines(self, edges):
        #Detect straight lines using Hough Transform
        try:
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)
            return lines
        except Exception as e:
            logger.error(f"Error in detecting lines: {e}")
            raise

    def analyze_contours_and_lines(self, contours, lines):
        #Analyze detected contours and lines to count vertical and horizontal threads
        try:
            thread_count = 0
            vertical_count = 0
            horizontal_count = 0

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                    if 80 <= angle <= 100: 
                        vertical_count += 1
                    else:
                        horizontal_count += 1
                    thread_count += 1  

            for contour in contours:
                if cv2.contourArea(contour) > self.min_contour_area:
                    thread_count += 1
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        angle = ellipse[2]

                        if 80 <= angle <= 100:
                            vertical_count += 1
                        else:
                            horizontal_count += 1

            # Debugging Prints
            print(f"Total Lines Detected: {len(lines) if lines is not None else 0}")
            print(f"Total Contours Detected: {len(contours)}")
            print(f"Vertical Count: {vertical_count}, Horizontal Count: {horizontal_count}")

            return thread_count, vertical_count, horizontal_count
        except Exception as e:
            logger.error(f"Error in analyzing contours and lines: {e}")
            raise

    def calculate_threads_per_sq_inch(self, thread_count):
        #Calculate threads per square inch based on cloth dimensions
        try:
            cloth_area_sq_inch = self.cloth_width_inch * self.cloth_height_inch
            return thread_count / cloth_area_sq_inch if cloth_area_sq_inch > 0 else 0
        except Exception as e:
            logger.error(f"Error in calculating threads per square inch: {e}")
            raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            image = cv2.imread(file_path)
            if image is None:
                return jsonify({"error": "Could not read the image"}), 400

            cloth_width_inch = 1  #cloth dimensions
            cloth_height_inch = 1 #cloth dimensions

            thread_counter = ThreadCounter(cloth_width_inch, cloth_height_inch)
            thread_count, vertical_count, horizontal_count, threads_per_sq_inch = thread_counter.count_threads_in_cloth(image)

            os.remove(file_path)

            return jsonify({
                "thread_count": thread_count,
                "vertical_count": vertical_count,
                "horizontal_count": horizontal_count,
                "threads_per_sq_inch": threads_per_sq_inch
            })
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({"error": "Internal server error"}), 500

def allowed_file(filename):
    """ Check if the file has an allowed extension. """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True)
