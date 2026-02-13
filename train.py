import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=666
    )

    C = 1.0
    max_iter = 200

    with mlflow.start_run():
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)

        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"accuracy : {acc : 4f}")

        run_id = mlflow.active_run().info.run_id

        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="IrisClassifier"
        )

        print("registered:", result.name, result.version)


if __name__ == "__main__":
    main()




