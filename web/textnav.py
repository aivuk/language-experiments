from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///books.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(120), unique=True)

    def __init__(self, username, email):
        self.username = username
        self.email = email
     
    def __repr__(self):
        return '<User %r>' % self.username

class Book(db.Model):
     id = db.Column(db.Integer, primary_key=True) 
     name = db.Column(db.String())
     filename = db.Column(db.String())
     # language = pass
     # number_of_words = pass
     # number_of_chars =

class Picture(db.Model):
    id = db.Column(db.Integer, primary_key=True)  
    # date
    # method
    # file

class PictureParams(db.Model):
    id = db.Column(db.Integer, primary_key=True)  
    # picture_id
    # param
 

