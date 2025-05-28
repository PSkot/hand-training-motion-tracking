import src.landmarks as lm
import numpy as np
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
from collections import Counter


def elbow_plot(
    input, exercise, exercise_state, max_clusters=10, start_cluster=2, save_plot=True
):
    errors = []
    for i in range(max_clusters - 1):
        model = train_kmeans(input, i + start_cluster, exercise, exercise_state, False)
        errors.append(model.inertia_)

    plt.plot(errors)
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE (inertia)")
    plt.xticks(
        list(range(max_clusters - 1)),
        [i + start_cluster for i in range(max_clusters - 1)],
    )

    plt.title("KMeans Elbow Plot for Hand Clustering")

    if save_plot:
        plt.savefig("plots/elbow.png")
    plt.show()


def plot_landmark_cluster_means(model, X, n_clusters):
    clusters = {
        cluster: np.array([X[i] for i, c in enumerate(model.labels_) if c == cluster])
        for cluster in range(n_clusters)
    }

    for cluster in clusters:
        fig = lm.get_landmark_plot(
            np.mean(clusters[cluster], axis=0).reshape(21, 3),
            title=f"Cluster {cluster}, cluster size: {len(clusters[cluster])}",
        )
        fig.show()


def prepare_landmark_data(exercise, exercise_state):
    s_path = f"./inference/landmark_predictions/student/{exercise}/{exercise_state}/combined.json"

    with open(s_path, "r") as f:
        landmarks = np.array(json.load(f))

    aligned_landmarks = np.array(
        [lm.align_landmarks_stable(s)[0] for s in landmarks]
    ).reshape(-1, 63)

    return aligned_landmarks, landmarks


def train_kmeans(X, n_clusters, exercise, exercise_state, save_model=True, suffix=""):
    model = KMeans(n_clusters=n_clusters)
    model.fit(X)

    if save_model:
        with open(
            f"saved_models/kmeans_{exercise}_{exercise_state}_{suffix}.pkl", "wb"
        ) as f:
            pickle.dump(model, f)
            print(f"Model saved to: saved_models/kmeans_{exercise}_{exercise_state}_{suffix}.pkl")

    return model


if __name__ == "__main__":
    exercise = "fstretch"
    exercise_state = "0000"
    n_clusters = 3
    X, _ = prepare_landmark_data(exercise, exercise_state)

    model_landmarks = train_kmeans(
        X,
        n_clusters,
        exercise,
        exercise_state,
        save_model=True,
        suffix="aligned_landmarks",
    )