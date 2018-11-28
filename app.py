from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps

app = Flask(__name__)

app.config["SQLALCHEMY_ECHO"] = False
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
app.config["SQLALCHEMY_DATABASE_URI"] = "postgres://clrjmonqrginde:284171df33b7d58673e9b9f3b572523ce59e2560dcf462efdb8f3052bc2f07c5@ec2-54-246-86-167.eu-west-1.compute.amazonaws.com:5432/damcpa0auiprid"
app.config["SECRET_KEY"] = "this is my secret key 1245215"

db = SQLAlchemy(app)


# ------------------DATABASE --------------------------------------


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(150), unique=True)
    uname = db.Column(db.String(50), unique=True)
    name = db.Column(db.String(50))
    lname = db.Column(db.String(50))
    quote = db.Column(db.String(300))
    sex = db.Column(db.String(1))
    country = db.Column(db.String(150))
    password = db.Column(db.String(80))


class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(200))
    complete = db.Column(db.Boolean)
    user_id = db.Column(db.Integer)


class Slinks(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    link = db.Column(db.String(300))
    site = db.Column(db.String(100))


class Fdata(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(150))
    weight = db.Column(db.Integer)
    mood = db.Column(db.String(200))
    hbeat = db.Column(db.Integer)
    todos = db.Column(db.Integer)


class Stats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    data_bar = db.Column(db.ARRAY(db.Float))

# ---------------------- HELPER FUNCTIONS ----------------------------


def token_checker(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
            print(request.headers['x-access-token'])

        if not token:
            print(request.headers)
            return jsonify({"ok": False, "message": "Token is missing!"})

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
            current_user = User.query.filter_by(
                user_id=data['user_id']).first()
        except:
            return jsonify({"ok": False, "message": "Token is invalid"})

        return f(current_user, *args, **kwargs)

    return decorated


# ---------------------- ROUTES --------------------------------------
@app.route('/login', methods=['POST'])
def login():

    auth = request.authorization
    if not auth or not auth.username or not auth.password:
        return make_response('Verification error', 401, {"WWW-Authentificate": "Login required"})
    user = User.query.filter_by(uname=auth.username).first()

    if not user:
        return make_response('Verification error', 401, {"WWW-Authentificate": "Login required"})

    if check_password_hash(user.password, auth.password):
        token = jwt.encode({'user_id': user.user_id, 'exp': datetime.datetime.utcnow(
        ) + datetime.timedelta(hours=168)}, app.config['SECRET_KEY'])
        return jsonify({"ok": True, "token": token.decode('UTF-8')})

    return make_response('Verification error', 401, {"WWW-Authentificate": "Login required"})


@app.route('/user', methods=['POST'])
def signup():
    data = request.get_json()
    hashed_password = generate_password_hash(data['password'], method='sha256')

    new_user = User(user_id=str(uuid.uuid4(
    )), uname=data['uname'], name=data['name'], lname=data['lname'], sex=data['sex'], password=hashed_password)

    if(data['quote']):
        new_user.quote = data['quote']

    if(data['country']):
        new_user.country = data['country']

    db.session.add(new_user)
    db.session.commit()
    return jsonify({"ok": True, "message": "New user created!"})


@app.route('/user', methods=['GET'])
@token_checker
def get_user_data(current_user):
    user = User.query.filter_by(uname=current_user.uname).first()
    user_data = {}
    user_data["id"] = user.id
    user_data["user_id"] = user.user_id
    user_data["uname"] = user.uname
    user_data["name"] = user.name
    user_data["lname"] = user.lname
    user_data["quote"] = user.quote
    user_data["sex"] = user.sex
    user_data["country"] = user.country

    return jsonify({"ok": "true", "user": user_data})


@app.route('/fdata', methods=['GET'])
@token_checker
def get_user_fdata(current_user):
    print(current_user.user_id)
    udata = Fdata.query.filter_by(user_id=current_user.user_id).first()

    if not udata:
        return jsonify({"ok": "false", "message": "No data to display"})

    user_fdata = {}
    user_fdata["mood"] = udata.mood
    user_fdata["weight"] = udata.weight
    user_fdata["hbeat"] = udata.hbeat

    return jsonify({"ok": "true", "fdata": user_fdata})


@app.route('/fdata', methods=['POST'])
@token_checker
def update_user_fdata(current_user):

    data = request.form
    new_fdata = Fdata(user_id=current_user.user_id,
                      mood=data["mood"], hbeat=data["hbeat"], weight=data["weight"])

    db.session.add(new_fdata)
    db.session.commit()
    return jsonify({"ok": "true", "message": "Fdata updated"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
