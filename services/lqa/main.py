from kive.shared.app_factory import build_app
from .detector import LQADetector

app = build_app(
    service_name="LQA",
    service_version="1.0.0",
    weight=0.14,
    detector_class=LQADetector,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
