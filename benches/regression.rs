mod common;

use std::hint::black_box;
use std::time::Duration;

use common::{
    fitted_linear_model, fitted_logistic_model, linear_targets, logistic_targets,
    regression_features,
};
use criterion::{
    black_box as criterion_black_box, criterion_group, criterion_main, BatchSize, BenchmarkId,
    Criterion, Throughput,
};
use ferrouslearn::{LinearRegression, LogisticRegression};

fn bench_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for &samples in &[128usize, 1_024usize] {
        let features = regression_features(samples, 4);
        let linear_target = linear_targets(&features);
        let logistic_target = logistic_targets(&features);
        let linear = fitted_linear_model(&features, &linear_target);
        let logistic = fitted_logistic_model(&features, &logistic_target);
        let prediction_inputs = regression_features(samples / 2, 4);

        group.throughput(Throughput::Elements(samples as u64));
        group.bench_function(BenchmarkId::new("linear_fit", samples), |b| {
            b.iter_batched(
                || LinearRegression::new(0.05, 400),
                |mut model| {
                    model
                        .fit(
                            criterion_black_box(&features),
                            criterion_black_box(&linear_target),
                            false,
                        )
                        .unwrap();
                    criterion_black_box(model)
                },
                BatchSize::SmallInput,
            )
        });

        group.throughput(Throughput::Elements(prediction_inputs.len() as u64));
        group.bench_function(
            BenchmarkId::new("linear_predict", prediction_inputs.len()),
            |b| {
                b.iter(|| {
                    let predictions = linear.predict(black_box(&prediction_inputs)).unwrap();
                    black_box(predictions)
                })
            },
        );

        group.throughput(Throughput::Elements(samples as u64));
        group.bench_function(BenchmarkId::new("logistic_fit", samples), |b| {
            b.iter_batched(
                || LogisticRegression::new(0.08, 500),
                |mut model| {
                    model
                        .fit(
                            criterion_black_box(&features),
                            criterion_black_box(&logistic_target),
                            false,
                        )
                        .unwrap();
                    criterion_black_box(model)
                },
                BatchSize::SmallInput,
            )
        });

        group.throughput(Throughput::Elements(prediction_inputs.len() as u64));
        group.bench_function(
            BenchmarkId::new("logistic_predict", prediction_inputs.len()),
            |b| {
                b.iter(|| {
                    let predictions = logistic.predict(black_box(&prediction_inputs)).unwrap();
                    black_box(predictions)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_regression
);
criterion_main!(benches);
