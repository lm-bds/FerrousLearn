use ferrouslearn::{
    DistanceMetric, FerrousError, KMeans, KNearestNeighboursRegressor, LinearRegression,
    LogisticRegression, PrincipalComponentAnalysis, Verbosity, WeightingFunction,
};

#[test]
fn kmeans_rejects_invalid_shapes_and_values() {
    let mut model = KMeans::new(2, 10, 1e-4);

    assert_eq!(model.fit(&[], 7), Err(FerrousError::EmptyDataset));

    let ragged = vec![vec![1.0, 2.0], vec![3.0]];
    assert!(matches!(
        model.fit(&ragged, 7),
        Err(FerrousError::RaggedMatrix {
            row: 2,
            expected: 2,
            actual: 1,
        })
    ));

    let non_finite = vec![vec![1.0, f64::NAN]];
    assert!(matches!(
        model.fit(&non_finite, 7),
        Err(FerrousError::NonFiniteInput { row: 1, column: 2 })
    ));
}

#[test]
fn kmeans_rejects_invalid_cluster_counts_and_prediction_mismatch() {
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

    let mut zero_clusters = KMeans::new(0, 10, 1e-4);
    assert!(matches!(
        zero_clusters.fit(&data, 7),
        Err(FerrousError::InvalidClusterCount {
            clusters: 0,
            sample_count: 2,
        })
    ));

    let mut too_many_clusters = KMeans::new(3, 10, 1e-4);
    assert!(matches!(
        too_many_clusters.fit(&data, 7),
        Err(FerrousError::InvalidClusterCount {
            clusters: 3,
            sample_count: 2,
        })
    ));

    let model = KMeans::new(1, 10, 1e-4);
    assert_eq!(model.predict(&data), Err(FerrousError::PredictionBeforeFit));

    let mut fitted = KMeans::new(1, 10, 1e-4);
    fitted.fit(&data, 7).unwrap();
    let wrong_shape = vec![vec![1.0, 2.0, 3.0]];
    assert!(matches!(
        fitted.predict(&wrong_shape),
        Err(FerrousError::FeatureCountMismatch {
            expected: 2,
            actual: 3,
        })
    ));
}

#[test]
fn pca_rejects_invalid_components_and_values() {
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

    let pca = PrincipalComponentAnalysis::new(0, 1e-4);
    assert!(matches!(
        pca.transform(&data),
        Err(FerrousError::InvalidPcaComponentCount {
            components: 0,
            feature_count: 2,
        })
    ));

    let pca = PrincipalComponentAnalysis::new(3, 1e-4);
    assert!(matches!(
        pca.transform(&data),
        Err(FerrousError::InvalidPcaComponentCount {
            components: 3,
            feature_count: 2,
        })
    ));

    let non_finite = vec![vec![1.0, f64::INFINITY], vec![3.0, 4.0]];
    let pca = PrincipalComponentAnalysis::new(1, 1e-4);
    assert!(matches!(
        pca.transform(&non_finite),
        Err(FerrousError::NonFiniteInput { row: 1, column: 2 })
    ));

    let single_sample = vec![vec![1.0, 2.0]];
    let pca = PrincipalComponentAnalysis::new(1, 1e-4);
    assert!(matches!(
        pca.transform(&single_sample),
        Err(FerrousError::InsufficientSamples { samples: 1 })
    ));

    let zero_variance = vec![vec![1.0, 5.0], vec![3.0, 5.0], vec![4.0, 5.0]];
    let pca = PrincipalComponentAnalysis::new(1, 1e-4);
    assert!(matches!(
        pca.transform(&zero_variance),
        Err(FerrousError::ZeroVarianceFeature { column: 2 })
    ));

    let empty = PrincipalComponentAnalysis::new(1, 1e-4);
    assert_eq!(empty.transform(&[]), Err(FerrousError::EmptyDataset));
    assert!(matches!(
        empty.transform(&[vec![]]),
        Err(FerrousError::EmptyRow { row: 1 })
    ));
}

#[test]
fn knn_rejects_invalid_training_inputs_and_prediction_mismatch() {
    let mut model =
        KNearestNeighboursRegressor::new(1, WeightingFunction::Uniform, DistanceMetric::Euclidean);
    let clean_x_train = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

    assert_eq!(
        model.fit(&clean_x_train, &[], Verbosity::Silent),
        Err(FerrousError::EmptyTargets)
    );

    let y_train = vec![1.0];
    assert!(matches!(
        model.fit(&clean_x_train, &y_train, Verbosity::Silent),
        Err(FerrousError::TargetLengthMismatch {
            samples: 2,
            targets: 1,
        })
    ));

    let y_train = vec![1.0, f64::NAN];
    assert!(matches!(
        model.fit(&clean_x_train, &y_train, Verbosity::Silent),
        Err(FerrousError::NonFiniteTarget { index: 2 })
    ));

    let x_train = vec![vec![1.0, f64::NAN], vec![3.0, 4.0]];
    let y_train = vec![1.0, 2.0];
    assert!(matches!(
        model.fit(&x_train, &y_train, Verbosity::Silent),
        Err(FerrousError::NonFiniteInput { row: 1, column: 2 })
    ));

    let y_train = vec![1.0, 2.0];

    let mut model =
        KNearestNeighboursRegressor::new(0, WeightingFunction::Uniform, DistanceMetric::Euclidean);
    assert!(matches!(
        model.fit(&clean_x_train, &y_train, Verbosity::Silent),
        Err(FerrousError::InvalidK {
            k: 0,
            sample_count: 2,
        })
    ));

    let mut model =
        KNearestNeighboursRegressor::new(3, WeightingFunction::Uniform, DistanceMetric::Euclidean);
    assert!(matches!(
        model.fit(&clean_x_train, &y_train, Verbosity::Silent),
        Err(FerrousError::InvalidK {
            k: 3,
            sample_count: 2,
        })
    ));

    let mut fitted =
        KNearestNeighboursRegressor::new(1, WeightingFunction::Uniform, DistanceMetric::Euclidean);
    fitted
        .fit(&clean_x_train, &y_train, Verbosity::Silent)
        .unwrap();
    assert_eq!(fitted.predict(&[]), Err(FerrousError::EmptyDataset));
    assert!(matches!(
        fitted.predict(&[vec![]]),
        Err(FerrousError::EmptyRow { row: 1 })
    ));
    let prediction = vec![vec![1.0, 2.0, 3.0]];
    assert!(matches!(
        fitted.predict(&prediction),
        Err(FerrousError::FeatureCountMismatch {
            expected: 2,
            actual: 3,
        })
    ));
}

#[test]
fn knn_distance_weighting_handles_exact_matches_without_infinities() {
    let mut model =
        KNearestNeighboursRegressor::new(3, WeightingFunction::Distance, DistanceMetric::Euclidean);

    let x_train = vec![vec![0.0], vec![0.0], vec![1.0]];
    let y_train = vec![2.0, 4.0, 100.0];
    model.fit(&x_train, &y_train, Verbosity::Silent).unwrap();

    let predictions = model.predict(&[vec![0.0]]).unwrap();
    assert_eq!(predictions, vec![3.0]);
    assert!(predictions.iter().all(|value| value.is_finite()));
}

#[test]
fn knn_distance_weighting_returns_single_exact_match_target() {
    let mut model =
        KNearestNeighboursRegressor::new(2, WeightingFunction::Distance, DistanceMetric::Euclidean);

    let x_train = vec![vec![0.0], vec![2.0]];
    let y_train = vec![7.5, 99.0];
    model.fit(&x_train, &y_train, Verbosity::Silent).unwrap();

    let predictions = model.predict(&[vec![0.0]]).unwrap();
    assert_eq!(predictions, vec![7.5]);
    assert!(predictions.iter().all(|value| value.is_finite()));
}

#[test]
fn knn_uniform_weighting_keeps_all_k_neighbours_for_exact_queries() {
    let mut model =
        KNearestNeighboursRegressor::new(2, WeightingFunction::Uniform, DistanceMetric::Euclidean);

    let x_train = vec![vec![0.0], vec![2.0]];
    let y_train = vec![10.0, 20.0];
    model.fit(&x_train, &y_train, Verbosity::Silent).unwrap();

    assert_eq!(model.predict(&[vec![0.0]]).unwrap(), vec![15.0]);
}

#[test]
fn kmeans_handles_empty_cluster_regression_without_panicking() {
    let data = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let mut model = KMeans::new(2, 10, 1e-4);

    model.fit(&data, 7).unwrap();
    assert_eq!(model.predict(&[]), Err(FerrousError::EmptyDataset));
    assert!(matches!(
        model.predict(&[vec![]]),
        Err(FerrousError::EmptyRow { row: 1 })
    ));
    let predictions = model.predict(&data).unwrap();

    assert_eq!(predictions.len(), 2);
    assert!(predictions.iter().all(|&cluster| cluster < 2));
}

#[test]
fn learning_errors_are_typed_and_displayable() {
    let err = FerrousError::InvalidK {
        k: 0,
        sample_count: 2,
    };
    let error: &dyn std::error::Error = &err;

    assert_eq!(err.to_string(), "k=0 is invalid for sample count 2");
    assert_eq!(error.to_string(), "k=0 is invalid for sample count 2");
}

#[test]
fn knn_requires_fit_before_prediction() {
    let model =
        KNearestNeighboursRegressor::new(1, WeightingFunction::Uniform, DistanceMetric::Euclidean);
    assert_eq!(
        model.predict(&[vec![1.0, 2.0]]),
        Err(FerrousError::PredictionBeforeFit)
    );
}

#[test]
fn linear_regression_rejects_invalid_training_inputs_and_prediction_mismatch() {
    let mut model = LinearRegression::new(0.01, 10);
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

    assert_eq!(
        model.fit(&data, &[], false),
        Err(FerrousError::EmptyTargets)
    );

    let ragged = vec![vec![1.0, 2.0], vec![3.0]];
    assert!(matches!(
        model.fit(&ragged, &[1.0, 2.0], false),
        Err(FerrousError::RaggedMatrix {
            row: 2,
            expected: 2,
            actual: 1,
        })
    ));

    let targets = vec![1.0];
    assert!(matches!(
        model.fit(&data, &targets, false),
        Err(FerrousError::TargetLengthMismatch {
            samples: 2,
            targets: 1,
        })
    ));

    let targets = vec![1.0, f64::INFINITY];
    assert!(matches!(
        model.fit(&data, &targets, false),
        Err(FerrousError::NonFiniteTarget { index: 2 })
    ));

    let data = vec![vec![1.0, f64::NAN], vec![3.0, 4.0]];
    let targets = vec![1.0, 2.0];
    assert!(matches!(
        model.fit(&data, &targets, false),
        Err(FerrousError::NonFiniteInput { row: 1, column: 2 })
    ));

    let mut fitted = LinearRegression::new(0.01, 10);
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let targets = vec![1.0, 2.0];
    fitted.fit(&data, &targets, false).unwrap();
    assert_eq!(fitted.predict(&[]), Err(FerrousError::EmptyDataset));
    assert!(matches!(
        fitted.predict(&[vec![]]),
        Err(FerrousError::EmptyRow { row: 1 })
    ));
    assert!(matches!(
        fitted.predict(&[vec![1.0, 2.0, 3.0]]),
        Err(FerrousError::FeatureCountMismatch {
            expected: 2,
            actual: 3,
        })
    ));
}

#[test]
fn linear_regression_requires_fit_before_prediction() {
    let model = LinearRegression::new(0.01, 10);
    assert_eq!(
        model.predict(&[vec![1.0, 2.0]]),
        Err(FerrousError::PredictionBeforeFit)
    );
}

#[test]
fn logistic_regression_rejects_invalid_training_inputs_and_prediction_mismatch() {
    let mut model = LogisticRegression::new(0.01, 10);
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

    assert_eq!(
        model.fit(&data, &[], false),
        Err(FerrousError::EmptyTargets)
    );

    let targets = vec![0.0];
    assert!(matches!(
        model.fit(&data, &targets, false),
        Err(FerrousError::TargetLengthMismatch {
            samples: 2,
            targets: 1,
        })
    ));

    let targets = vec![0.0, f64::NAN];
    assert!(matches!(
        model.fit(&data, &targets, false),
        Err(FerrousError::NonFiniteTarget { index: 2 })
    ));

    let data = vec![vec![1.0, f64::INFINITY], vec![3.0, 4.0]];
    let targets = vec![0.0, 1.0];
    assert!(matches!(
        model.fit(&data, &targets, false),
        Err(FerrousError::NonFiniteInput { row: 1, column: 2 })
    ));

    let negative_targets = vec![0.0, -0.1];
    let clean_data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    assert!(matches!(
        model.fit(&clean_data, &negative_targets, false),
        Err(FerrousError::TargetOutOfRange { index: 2 })
    ));

    let above_one_targets = vec![0.0, 1.1];
    assert!(matches!(
        model.fit(&clean_data, &above_one_targets, false),
        Err(FerrousError::TargetOutOfRange { index: 2 })
    ));

    let mut fitted = LogisticRegression::new(0.01, 10);
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let targets = vec![0.0, 1.0];
    fitted.fit(&data, &targets, false).unwrap();
    assert_eq!(fitted.predict(&[]), Err(FerrousError::EmptyDataset));
    assert!(matches!(
        fitted.predict(&[vec![]]),
        Err(FerrousError::EmptyRow { row: 1 })
    ));
    assert!(matches!(
        fitted.predict(&[vec![1.0, 2.0, 3.0]]),
        Err(FerrousError::FeatureCountMismatch {
            expected: 2,
            actual: 3,
        })
    ));
}

#[test]
fn logistic_regression_requires_fit_before_prediction() {
    let model = LogisticRegression::new(0.01, 10);
    assert_eq!(
        model.predict(&[vec![1.0, 2.0]]),
        Err(FerrousError::PredictionBeforeFit)
    );
}

#[test]
fn learning_errors_display_and_error_trait_cover_new_variants() {
    let err = FerrousError::TargetOutOfRange { index: 2 };
    let error: &dyn std::error::Error = &err;

    assert!(err.to_string().contains('2'));
    assert_eq!(err.to_string(), error.to_string());

    let pca_err = FerrousError::ZeroVarianceFeature { column: 3 };
    assert!(pca_err.to_string().contains('3'));
}
