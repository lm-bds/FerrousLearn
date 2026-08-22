mod common;

use std::hint::black_box;
use std::time::Duration;

use common::{fitted_kmeans, kmeans_data, kmeans_prediction_data, KMEANS_CLUSTERS, KMEANS_SEED};
use criterion::{
    black_box as criterion_black_box, criterion_group, criterion_main, BatchSize, BenchmarkId,
    Criterion, Throughput,
};
use ferrouslearn::KMeans;

fn bench_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for &samples in &[100usize, 1_000usize] {
        let data = kmeans_data(samples, 4);
        let predictions = kmeans_prediction_data(samples, 4);
        let fitted = fitted_kmeans(&data);

        group.throughput(Throughput::Elements(samples as u64));
        group.bench_function(BenchmarkId::new("fit", samples), |b| {
            b.iter_batched(
                || KMeans::new(KMEANS_CLUSTERS, 100, 1e-8),
                |mut model| {
                    let result = model.fit(criterion_black_box(&data), KMEANS_SEED);
                    result.unwrap();
                    criterion_black_box(model)
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_function(BenchmarkId::new("predict", samples), |b| {
            b.iter(|| {
                let labels = fitted.predict(black_box(&predictions)).unwrap();
                black_box(labels)
            })
        });
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_kmeans
);
criterion_main!(benches);
