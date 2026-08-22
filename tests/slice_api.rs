use ferrouslearn::{
    DistanceMetric, KMeans, KNearestNeighboursRegressor, LinearRegression, LogisticRegression,
    PrincipalComponentAnalysis, Verbosity, WeightingFunction,
};

#[test]
fn public_learning_apis_accept_slices() {
    let x_train: &[Vec<f64>] = &[vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    let y_train: &[f64] = &[2.0, 3.0, 4.0];
    let prediction_matrix: &[Vec<f64>] = &[vec![2.0, 3.0]];
    let data: &[Vec<f64>] = &[vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    let labels: &[f64] = &[3.0, 5.0, 7.0];

    let mut knn =
        KNearestNeighboursRegressor::new(3, WeightingFunction::Uniform, DistanceMetric::Euclidean);
    knn.fit(x_train, y_train, Verbosity::Silent);
    let knn_predictions = knn.predict(prediction_matrix);
    assert_eq!(knn_predictions.len(), 1);

    let mut linear = LinearRegression::new(0.01, 10);
    linear.fit(data, labels, false);
    let linear_predictions = linear.predict(data);
    assert_eq!(linear_predictions.len(), data.len());

    let mut logistic = LogisticRegression::new(0.01, 10);
    logistic.fit(data, &[0.0, 0.0, 1.0], false);
    let logistic_predictions = logistic.predict(data);
    assert_eq!(logistic_predictions.len(), data.len());

    let pca = PrincipalComponentAnalysis::new(1, 0.01);
    let transformed = pca.transform(data.to_vec());
    assert_eq!(transformed.len(), data.len());

    let mut kmeans = KMeans::new(1, 10, 0.0001);
    kmeans.fit(data, 42);
    let kmeans_predictions = kmeans.predict(data);
    assert_eq!(kmeans_predictions.len(), data.len());
}
