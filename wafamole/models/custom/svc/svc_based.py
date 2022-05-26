import numpy as np
from wafamole.tokenizer import Tokenizer
from wafamole.models import SklearnModelWrapper
from wafamole.utils.check import type_check
from wafamole.exceptions.models_exceptions import (
    NotSklearnModelError,
    SklearnInternalError,
    ModelNotLoadedError,
)


class SVCClassifierWrapper(SklearnModelWrapper):
    def extract_features(self, value: str):
        type_check(value, str, "value")
        #feature_vector = np.fromstring(value, np.int8) - 48
        return value

    def classify(self, value):
        """Produce probability of being sql injection.
        
        Arguments:
            value (str) : input query
        
        Raises:
        TypeError: value is not string
        ModelNotLoaderError: no model is loaded
        SklearnInternalError: generic exception

        Returns:
           float : probability of being a sql injection
        """
        if(self._sklearn_classifier == None):
            raise ModelNotLoadedError()
        feature_vector = self.extract_features(value)
        try:
            y_pred = self._sklearn_classifier.predict_proba([feature_vector])
            return y_pred[0,0]
        except Exception as e:
            raise SklearnInternalError("Internal sklearn error.") from e
