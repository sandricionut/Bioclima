from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class AIAnalytics:
    """Machine-learning analytics interface (D1.1 §8.1.6)."""

    def __init__(self) -> None:
        self.model: RandomForestClassifier | None = None
        self.feature_names: list[str] | None = None

    def train_rf_classifier(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: list[str] | None = None,
        n_estimators: int = 300,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ) -> RandomForestClassifier:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.model.fit(np.asarray(x), np.asarray(y))
        self.feature_names = feature_names
        return self.model

    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        self._require_model()
        return self.model.predict(np.asarray(x))

    def predict_probabilities(self, x: np.ndarray) -> np.ndarray:
        self._require_model()
        return self.model.predict_proba(np.asarray(x))

    def feature_importances(self) -> dict[str, float]:
        self._require_model()
        names = self.feature_names or [f"feature_{i}" for i in range(self.model.n_features_in_)]
        return dict(zip(names, self.model.feature_importances_.astype(float), strict=True))

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("No model has been trained")
