use ferrouslearn::{
    pairwise_distance, standardise, DistanceMetric, KMeans, KNearestNeighboursRegressor,
    LogisticRegression, PrincipalComponentAnalysis, Verbosity, WeightingFunction,
};
use proptest::prelude::*;

const CASES: u32 = 32;
const DISTANCE_EPS: f64 = 1e-10;

fn pair_vector_strategy() -> impl Strategy<Value = (Vec<f64>, Vec<f64>)> {
    prop::collection::vec((-20i32..=20, -20i32..=20), 1..=8).prop_map(|pairs| {
        let mut left = Vec::with_capacity(pairs.len());
        let mut right = Vec::with_capacity(pairs.len());
        for (l, r) in pairs {
            left.push(l as f64 / 3.0);
            right.push(r as f64 / 3.0);
        }
        (left, right)
    })
}

fn same_vector_strategy() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-15i32..=15, 1..=8)
        .prop_map(|values| values.into_iter().map(|value| value as f64 / 4.0).collect())
}

fn standardisable_matrix_strategy() -> impl Strategy<Value = Vec<Vec<f64>>> {
    (3usize..=7, 2usize..=5).prop_flat_map(|(rows, cols)| {
        let columns = prop::collection::vec((1i32..=5, -20i32..=20), cols);
        columns.prop_map(move |specs| {
            (0..rows)
                .map(|row| {
                    specs
                        .iter()
                        .enumerate()
                        .map(|(col, (slope, intercept))| {
                            *intercept as f64 + (*slope as f64) * row as f64 + col as f64 * 0.25
                        })
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>()
        })
    })
}

fn pca_matrix_strategy() -> impl Strategy<Value = Vec<Vec<f64>>> {
    (3usize..=8, 2usize..=2).prop_flat_map(|(rows, cols)| {
        let offset = (-10i32..=10, -10i32..=10, 1i32..=4);
        offset.prop_map(move |(base0, base1, slope)| {
            (0..rows)
                .map(|row| {
                    let x = row as f64;
                    let quadratic = x * x * 0.5;
                    vec![
                        base0 as f64 + slope as f64 * x,
                        base1 as f64 + (slope + cols as i32) as f64 * x + quadratic,
                    ]
                })
                .collect::<Vec<Vec<f64>>>()
        })
    })
}

fn training_targets_strategy() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-50i32..=50, 3..=6)
        .prop_map(|values| values.into_iter().map(|value| value as f64 / 5.0).collect())
}

fn kmeans_seed_dataset() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![10.0, 10.0],
        vec![10.0, 11.0],
        vec![20.0, 20.0],
        vec![20.0, 21.0],
    ]
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_stddev(values: &[f64]) -> f64 {
    let avg = mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - avg;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    variance.sqrt()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]

    #[test]
    fn distances_are_non_negative_symmetric_and_identity_holds((left, right) in pair_vector_strategy()) {
        for metric in [DistanceMetric::Euclidean, DistanceMetric::Manhattan] {
            let forward = pairwise_distance(&left, &right, metric).unwrap();
            let backward = pairwise_distance(&right, &left, metric).unwrap();
            prop_assert!(forward >= 0.0);
            prop_assert!((forward - backward).abs() <= DISTANCE_EPS);
        }
    }

    #[test]
    fn distances_are_near_zero_on_the_identity(values in same_vector_strategy()) {
        for metric in [DistanceMetric::Euclidean, DistanceMetric::Manhattan] {
            let distance = pairwise_distance(&values, &values, metric).unwrap();
            prop_assert!(distance.abs() <= DISTANCE_EPS);
        }
    }

    #[test]
    fn standardised_nonconstant_columns_have_zero_mean_and_unit_sample_stddev(data in standardisable_matrix_strategy()) {
        let standardised = standardise(&data).unwrap();
        prop_assert_eq!(standardised.len(), data.len());
        prop_assert_eq!(standardised[0].len(), data[0].len());
        prop_assert!(standardised.iter().flatten().all(|value| value.is_finite()));

        for column in 0..standardised[0].len() {
            let column_values: Vec<f64> = standardised.iter().map(|row| row[column]).collect();
            prop_assert!(mean(&column_values).abs() <= 1e-10);
            prop_assert!((sample_stddev(&column_values) - 1.0).abs() <= 1e-10);
        }
    }

    #[test]
    fn pca_returns_the_requested_dimensions_and_finite_values(matrix in pca_matrix_strategy(), components in 1usize..=2) {
        let pca = PrincipalComponentAnalysis::with_max_iterations(components, 1e-12, 256);
        let transformed = pca.transform(&matrix).unwrap();

        prop_assert_eq!(transformed.len(), matrix.len());
        prop_assert_eq!(transformed[0].len(), components);
        prop_assert!(transformed.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn knn_uniform_predictions_stay_within_the_training_target_range(targets in training_targets_strategy(), k in 1usize..=6) {
        let x_train = [vec![0.0], vec![2.0], vec![4.0], vec![6.0], vec![8.0], vec![10.0]];
        let sample_count = targets.len();
        let x_train = &x_train[..sample_count];
        let mut model = KNearestNeighboursRegressor::new(k.min(sample_count), WeightingFunction::Uniform, DistanceMetric::Euclidean);

        model.fit(x_train, &targets, Verbosity::Silent).unwrap();
        let predictions = model.predict(&[vec![1.5], vec![5.5], vec![9.5]]).unwrap();
        let min_target = targets.iter().copied().fold(f64::INFINITY, f64::min);
        let max_target = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        prop_assert!(predictions.iter().all(|prediction| *prediction >= min_target - 1e-10 && *prediction <= max_target + 1e-10));
    }

    #[test]
    fn seeded_kmeans_is_reproducible_for_any_seed(seed in any::<u64>()) {
        let data = kmeans_seed_dataset();
        let mut first = KMeans::new(2, 32, 1e-12);
        let mut second = KMeans::new(2, 32, 1e-12);

        let first_result = first.fit(&data, seed);
        let second_result = second.fit(&data, seed);
        let first_ok = first_result.is_ok();
        prop_assert_eq!(first_result, second_result);

        if first_ok {
            let first_predictions = first.predict(&data).unwrap();
            let second_predictions = second.predict(&data).unwrap();
            prop_assert_eq!(first_predictions, second_predictions);
        }
    }

    #[test]
    fn logistic_regression_predictions_stay_bounded_and_monotone(lr in 0.01f64..=0.25, iterations in 500usize..=2000) {
        let x_train = vec![vec![-2.0], vec![-1.0], vec![1.0], vec![2.0]];
        let y_train = vec![0.0, 0.0, 1.0, 1.0];
        let mut model = LogisticRegression::new(lr, iterations);

        model.fit(&x_train, &y_train, false).unwrap();
        let predictions = model.predict(&[vec![-3.0], vec![-1.0], vec![0.0], vec![1.0], vec![3.0]]).unwrap();

        prop_assert!(predictions.iter().all(|value| (0.0..=1.0).contains(value)));
        prop_assert!(predictions.windows(2).all(|pair| pair[0] <= pair[1] + 1e-8));
    }
}
