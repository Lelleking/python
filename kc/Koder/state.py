# Shared mutable in-memory state for all modules.
# Import specific names from here rather than the whole module to keep access explicit.

# ML models (lazy-loaded on first use)
_clf_model = None
_reg_model = None
_baseline_reg_model = None
_plateau_reg_model = None
_rep_curve_model = None

# In-memory session stores (keyed by upload_set_id or session UUID)
_stored_upload_sets = {}   # raw uploaded data sets
_plot_datasets = {}        # datasets prepared for plot/select
_thalf_sessions = {}       # /analyze halftime sessions
_control_sessions = {}     # /control_halftimes sessions
_sigmoid_sessions = {}     # /control_sigmoid sessions
_group_analysis_sessions = {}  # /aggregation_analysis sessions
_event_ai_sessions = {}    # /aggregation_event_ai sessions
_plot_images = {}          # rendered PNG blobs keyed by plot_id
_gvc_sessions = {}         # /plot/group_vs_control sessions
