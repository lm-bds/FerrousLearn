mod common;

use std::hint::black_box;
use std::time::Duration;

use common::{fitted_pca, pca_data};
use criterion::{
    black_box as criterion_black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
    Throughput,
};

fn bench_pca(c: &mut Criterion) {
    let mut group = c.benchmark_group("pca");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for &(samples, features, components) in &[(64usize, 4usize, 3usize), (128usize, 4usize, 3usize)]
    {
        let data = pca_data(samples, features);
        let pca = fitted_pca(&data, components);
        let transformed = pca.transform(&data).unwrap();
        assert!(transformed.iter().all(|row| row.len() == components));
        assert!(transformed.iter().flatten().all(|value| value.is_finite()));

        group.throughput(Throughput::Elements(samples as u64));
        group.bench_function(
            BenchmarkId::new(format!("transform_{components}d"), samples),
            |b| {
                b.iter(|| {
                    let output = pca.transform(criterion_black_box(&data)).unwrap();
                    black_box(output)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_pca
);
criterion_main!(benches);
