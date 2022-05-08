import cv2
import joblib
import numpy as np
from math import copysign, log10
import imageio
import joblib
import os.path
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier

PATH_TO_RESOURCES = "resources/train"

class RGBSymbolExtractor():
    def fit(self, X, y=None, sample_weight=None):
        return self
    
    def transform(self, X, copy=None):
        return [symbol[np.sum(symbol, axis=2) < 255 * 3] for symbol in X]

class ColorHistogram():
    def __init__(self):
        self.kmeans = MiniBatchKMeans(8)
        
    def fit(self, X, y=None, sample_weight=None):
        sample_pixels = X[0]
        for i in range(1, len(X)):
            sample_pixels = np.concatenate((sample_pixels, X[i]), axis=0)

        self.kmeans.fit(sample_pixels)
        
        return self
    
    def transform(self, X, copy=None):
        histograms = []
        for i in range(len(X)):
            histo = np.bincount(np.uint8(self.kmeans.predict(X[i])),
                        minlength = self.kmeans.n_clusters) / len(X[i])
            histograms.append(histo)

        return histograms

class HuMoments():
    def __init__(self):
        pass
        
    def fit(self, X, y=None, sample_weight=None):
        return self
    
    def transform(self, X, copy=None):
        X_gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in X]
        huMoments = []

        for i in range(len(X)):
            # Calculate Moments
            moments = cv2.moments(X_gray[i])
            # Calculate Hu Moments
            theseHuMoments = cv2.HuMoments(moments)

            # Log scale hu moments
            for i in range(0,7):
                theseHuMoments[i] = -1* copysign(1.0, theseHuMoments[i]) * log10(abs(theseHuMoments[i]))

            huMoments.append(theseHuMoments.reshape((7,)))
            
        return huMoments

if __name__ == "__main__":
    RANDOM_STATE = 42

    X = []
    y = []
    for root, dirs, files in os.walk(PATH_TO_RESOURCES):
        for name in files:
            if name.endswith((".png")):
                file_path = os.path.join(root, name)
                dir_name = root.split(os.path.sep)[-1]
                im = imageio.imread(file_path)
                X.append(im)
                y.append(int(dir_name))
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=RANDOM_STATE)

    color_histo_step = ("color histogram", Pipeline([("extraction", RGBSymbolExtractor()), ("histogram", ColorHistogram())]))
    pipeline = Pipeline([("extractors", FeatureUnion([color_histo_step, ("hu moments", HuMoments())])), ("clf", RandomForestClassifier(random_state=RANDOM_STATE))])
    pipeline.fit(X, y)

    joblib.dump(pipeline, "trained_pipeline.pkl")