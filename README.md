## This repo contains codes and files for bank transaction analysis.

* controller: This folder contains the services and saved models.
* model: This folder contains files and notebooks used to create models and save them
* test: It contains all sample notebooks and test data

### Running the application by creating a virtualenv

```
#install virutal env
python3 -m venv dev-env

#install all the required libs
pip install -r requirements.txt

#run the controller
python ClassificationController.py 

```
This will expose one /classify endpoint. Pass a json structure like { "key" : "test data"} through postman or curl.  
Example:
```
curl -X GET \
  -H "Content-type: application/json" \
  -H "Accept: application/json" \
  -d '{"key":"AMAZON"}' \
  http://127.0.0.1:12345/classify
```