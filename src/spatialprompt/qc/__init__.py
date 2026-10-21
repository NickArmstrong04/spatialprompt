"""QC: detectors (doublet / split / empty / missed), repairs (box re-prompt, merge, drop) and the agent loop."""
from spatialprompt.qc.agent import QCAgent, RulesPlanner  # noqa: F401
from spatialprompt.qc.detectors import dapi_foreground, detect_all, summarize  # noqa: F401
