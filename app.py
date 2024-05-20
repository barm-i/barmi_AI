from flask import Flask, request, jsonify
from flask_restx import Api, Resource, reqparse
import service
from werkzeug.datastructures import FileStorage
app = Flask(__name__) 
api = Api(app)

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
        # For test (example feedback)
        feedbacks = [
            {
                "feedback": "글씨가 너무 좁습니다..",
                "coordinates": {"x": 150, "y": 300}
            },
            {
                "feedback": "폰트와 유사도가 낮습니다. 다시 한 번 작성해 보세요.",
                "coordinates": {"x": 200, "y": 350}
            }
        ]
        
        deduction_score = -5
        
        # 응답 생성
        response = {
            "message" : "success",
            "feedbacks": feedbacks,
            "deductionScore": deduction_score
        }

        return response, 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6258)

