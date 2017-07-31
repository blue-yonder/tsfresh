from flask import Flask

from tsfresh.big_fresh.app import add_views

app = Flask(__name__)
add_views(app)