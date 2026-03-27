import os
import threading
import webbrowser

from flask import Flask

from config import BASE_DIR, UPLOAD_FOLDER, MPL_CACHE_DIR
from db import init_auth_db
from data_utils import get_train_metrics_context

from routes.auth import auth_bp
from routes.runs import runs_bp
from routes.folders import folders_bp
from routes.main import main_bp
from routes.plots import plots_bp
from routes.halftimes import halftimes_bp
from routes.sigmoid import sigmoid_bp
from routes.smart_summary import smart_summary_bp
from routes.aggregation import aggregation_bp
from routes.event_ai import event_ai_bp

os.environ["MPLCONFIGDIR"] = MPL_CACHE_DIR

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "Koder", "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.register_blueprint(auth_bp)
app.register_blueprint(runs_bp)
app.register_blueprint(folders_bp)
app.register_blueprint(main_bp)
app.register_blueprint(plots_bp)
app.register_blueprint(halftimes_bp)
app.register_blueprint(sigmoid_bp)
app.register_blueprint(smart_summary_bp)
app.register_blueprint(aggregation_bp)
app.register_blueprint(event_ai_bp)


@app.context_processor
def inject_train_metrics():
    return get_train_metrics_context()


init_auth_db()


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5050
    url = f"http://localhost:{port}/"

    threading.Timer(1.0, lambda: webbrowser.open_new_tab(url)).start()

    # use_reloader=False keeps the server alive without restarting on file changes
    app.run(host=host, port=port, debug=False, use_reloader=False)
