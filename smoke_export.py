import os
import sys
import tempfile

import plotly.graph_objects as go

fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "probe.png")
    try:
        fig.write_image(path, format="png", width=1200, height=700, scale=2)
    except Exception as e:  # noqa: BLE001
        print(f"SMOKE TEST FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
    size = os.path.getsize(path)
    if size < 10_000:
        print(f"SMOKE TEST FAILED: PNG written but only {size} bytes")
        sys.exit(1)
    print(f"SMOKE TEST OK: frozen PNG export wrote {size} bytes")
