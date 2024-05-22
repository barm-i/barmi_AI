import requests
import uuid
import time
import json
from werkzeug.datastructures import FileStorage

API_URL = 'https://of8iaoehqq.apigw.ntruss.com/custom/v1/29771/02aafeb3a323d948ebf247e54886df1bb978f76a6d74aec46d57bd4721ca83c5/general'
SECRET_KEY = 'aktWZmRuTHdYSmVDVllVc0VJRXNkSkVaQkZnWG5YdVk='

image_file = 'test_OCR.png'
output_file = 'output/output1.json'

# request_json = {
#     'images': [
#         {
#             'format': 'png',
#             'name': 'demo'
#         }
#     ],
#     'requestId': str(uuid.uuid4()),
#     'version': 'V2',
#     'timestamp': int(round(time.time() * 1000))
# }

# payload = {'message': json.dumps(request_json).encode('UTF-8')}
# files = [
#   ('file', open(image_file,'rb'))
# ]
# headers = {
#   'X-OCR-SECRET': SECRET_KEY
# }

# response = requests.request("POST", API_URL, headers=headers, data = payload, files = files)

# res = json.loads(response.text.encode('utf8'))
# print(res)

# with open(output_file, 'w', encoding='utf-8') as outfile:
#     json.dump(res, outfile, indent=4, ensure_ascii=False)


def ocr_api(input_image:str):
  """
    Perform OCR on the input image using the OCR API.
    input : input_image (str) : path to the input image (example: 'test.png')
    output: concatenated_text (str) : concatenated text extracted from the image
  """
  
  request_json = {
    'images': [
        {
            'format': 'png',
            'name': 'demo'
        }
    ],
    'requestId': str(uuid.uuid4()),
    'version': 'V2',
    'timestamp': int(round(time.time() * 1000))
  }
  payload = {'message': json.dumps(request_json).encode('UTF-8')}

  files = [
    ('file', open(input_image,'rb'))
  ]
  headers = {
    'X-OCR-SECRET': SECRET_KEY
  }
  response = requests.request("POST", API_URL, headers=headers, data = payload, files = files)
  # Parse the JSON response
  response_json = response.json()
  
  # Extract inferText values
  infer_texts = [field['inferText'].replace(' ', '') for image in response_json['images'] for field in image['fields']]
  
  # Concatenate inferText values without spaces
  concatenated_text = ''.join(infer_texts)
  
  return concatenated_text

# def ocr_api(input_file):
#     """
#     Perform OCR on the input file using the OCR API.
#     input : input_file (FileStorage) : FileStorage object representing the input image
#     output: concatenated_text (str) : concatenated text extracted from the image
#     """
#     request_json = {
#         'images': [
#             {
#                 'format': 'png',
#                 'name': 'demo'
#             }
#         ],
#         'requestId': str(uuid.uuid4()),
#         'version': 'V2',
#         'timestamp': int(round(time.time() * 1000))
#     }
#     payload = {'message': json.dumps(request_json).encode('UTF-8')}
#     files = [
#         ('file', (input_file.filename, input_file.stream, input_file.mimetype))
#     ]
#     headers = {
#         'X-OCR-SECRET': SECRET_KEY
#     }
#     response = requests.request("POST", API_URL, headers=headers, data=payload, files=files)
#     # Parse the JSON response
#     response_json = response.json()
    
#     # Extract inferText values
#     infer_texts = [field['inferText'].replace(' ', '') for image in response_json['images'] for field in image['fields']]
    
#     # Concatenate inferText values without spaces
#     concatenated_text = ''.join(infer_texts)
    
#     return concatenated_text

# if __name__ == '__main__':
#   print(ocr_api(image_file))




string = "대한 사람 대한으로 길이 보전"
li_string = list(string)




# def ocr_api_for_one_letter(input_image):
#   """
#     Perform OCR on the input image using the OCR API.
#     input : input_image (str) : path to the input image (example: 'test.png')
#     output: concatenated_text (str) : concatenated text extracted from the image
#   """
#   request_json = {
#     'images': [
#         {
#             'format': 'png',
#             'name': 'demo'
#         }
#     ],
#     'requestId': str(uuid.uuid4()),
#     'version': 'V2',
#     'timestamp': int(round(time.time() * 1000))
#   }
#   payload = {'message': json.dumps(request_json).encode('UTF-8')}
#   files = [
#     ('file', open(input_image,'rb'))
#   ]
#   headers = {
#     'X-OCR-SECRET': SECRET_KEY
#   }
#   response = requests.request("POST", API_URL, headers=headers, data = payload, files = files)
#   # Parse the JSON response
#   response_json = response.json()
  
#   # Extract inferText values
#   infer_texts = [field['inferText'].replace(' ', '') for image in response_json['images'] for field in image['fields']]
  
#   # Concatenate inferText values without spaces
#   concatenated_text = ''.join(infer_texts)
  
#   return concatenated_text

