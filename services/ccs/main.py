from kive.shared.app_factory import build_app
from .detector import CCSDetector

app = build_app(
    service_name="CCS",
    service_version="1.0.0",
    weight=0.10,
    detector_class=CCSDetector,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
