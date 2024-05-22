# from flask import Flask, request, jsonify
# from flask_restx import Api, Resource, reqparse
# from werkzeug.datastructures import FileStorage
# import barmi
# app = Flask(__name__) 
# api = Api(app)

# @api.route('/hello')  # for test
# class HelloWorld(Resource):
#     def get(self):  # return JSON File
#         return {"hello": "world!"}

# parser = reqparse.RequestParser()
# parser.add_argument('text', type=str, required=True, help='Text cannot be blank')
# parser.add_argument('font_photo', type=FileStorage, location='files', required=True, help='Font photo is required')
# parser.add_argument('handwriting_photo', type=FileStorage, location='files', required=True, help='Handwriting photo is required')

# @api.route('/feedback')
# class Feedback(Resource):
#     def post(self):
#         args = parser.parse_args()
        
#         text = args['text']
#         font_photo = args['font_photo']
#         handwriting_photo = args['handwriting_photo']
#         if not text or not font_photo or not handwriting_photo:
#             return jsonify({"message": "Missing data"}), 400
#         # For test (example feedback)
#         response = barmi.feedback(text, font_photo, handwriting_photo)
#         if response == None:
#             return jsonify({"message": "Internal server error"}), 500
#         return response, 200



# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=6258)

# # curl -X POST -F "text=대한 사람 대한으로 길이 보전" -F "font_photo=@/Users/ihanseong/Desktop/testforwrite/test_for_ans.png" -F "handwriting_photo=@/Users/ihanseong/Desktop/testforwrite/test_for_user.png" http://175.196.97.78:6258//feedback

from flask import Flask, request, jsonify
from flask_restx import Api, Resource, reqparse
from werkzeug.datastructures import FileStorage
import os
import barmi

app = Flask(__name__) 
api = Api(app)

UPLOAD_FOLDER = 'uploads'
# automatic_increae = 1
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@api.route('/hello')  # for test
class HelloWorld(Resource):
    def get(self):  # return JSON File
        return {"hello": "world!"}

parser = reqparse.RequestParser()
parser.add_argument('text', type=str, required=True, help='Text cannot be blank')
parser.add_argument('font_photo', type=FileStorage, location='files', required=True, help='Font photo is required')
parser.add_argument('handwriting_photo', type=FileStorage, location='files', required=True, help='Handwriting photo is required')

@api.route('/feedback')
class Feedback(Resource):
    def post(self):
        args = parser.parse_args()
        
        text = args['text']
        font_photo = args['font_photo']
        handwriting_photo = args['handwriting_photo']
        
        if not text or not font_photo or not handwriting_photo:
            return jsonify({"message": "Missing data"}), 400
        
        # Save handwriting photo
        handwriting_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], handwriting_photo.filename)
        handwriting_photo.save(handwriting_photo_path)
        
        # Log the saved file path
        print(f"Saved handwriting photo to: {handwriting_photo_path}")
        response = barmi.feedback(text, font_photo,handwriting_photo, handwriting_photo_path)
        try:
            response = barmi.feedback(text, font_photo,handwriting_photo, handwriting_photo_path)
        finally:
            # Ensure the file is removed after processing
            os.remove(handwriting_photo_path)
            print(f"Removed handwriting photo from: {handwriting_photo_path}")
        
        if response is None:
            return jsonify({"message": "Internal server error"}), 500
        return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6258)
