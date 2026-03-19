from kive.shared.app_factory import build_app
from .detector import SVPDetector

app = build_app(
    service_name="SVP",
    service_version="1.0.0",
    weight=0.24,
    detector_class=SVPDetector,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
