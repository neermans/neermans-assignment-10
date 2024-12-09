FLASK_APP = app.py

run:
	FLASK_APP=$(FLASK_APP) flask run --port 3000