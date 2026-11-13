"""The dashboard still renders every tab (existing six plus the two advisor tabs)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_dashboard_renders_all_tabs():
    at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [t.label for t in at.tabs]
    for name in ("📈 Company", "⚖️ Compare", "🔎 Screener", "🗺️ Market map", "🕘 History",
                 "🗄️ Database", "💡 Advisor", "🩺 Portfolio"):
        assert name in labels
