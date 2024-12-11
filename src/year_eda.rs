use csv::ReaderBuilder;
use serde::Deserialize;
use plotters::prelude::*;
use statrs::statistics::{Data, Median, Statistics};
use petgraph::graph::Graph;
use std::error::Error;
use std::fs::File;

#[derive(Debug, Deserialize)]
pub struct Record {
    Country: String,
    Region: String,
    #[serde(rename = "Happiness Rank")]
    HappinessRank: f64, // This is the numerical field we'll use
}

pub(crate) fn load_csv(path: &str) -> Result<Vec<Record>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(File::open(path)?);

    let mut data = Vec::new();
    for result in reader.deserialize() {
        let record: Record = result?;
        data.push(record);
    }

    Ok(data)
}

pub(crate) fn calculate_statistics(data: &[Record]) -> Vec<f64> {
    data.iter().map(|record| record.HappinessRank).collect() // Use HappinessRank
}

pub(crate) fn print_statistics(values: &[f64]) {
    let mean = values.mean();
    let median = values.mean();
    let std_dev = values.std_dev();

    println!("Mean: {:.2}", if mean.is_nan() { 0.0 } else { mean });
    println!("Median: {:.2}", if median.is_nan() { 0.0 } else { median });
    println!("Std Dev: {:.2}", if std_dev.is_nan() { 0.0 } else { std_dev });
}

pub(crate) fn create_visualization(data: &[Record], output_path: &str) -> Result<(), Box<dyn Error>> {
    println!("Creating visualization: {}", output_path);

    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Data Visualization", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..data.len(), 0.0..100.0)?;

    chart.configure_mesh().draw()?;

    let points: Vec<(usize, f64)> = data
        .iter()
        .enumerate()
        .map(|(idx, record)| (idx, record.HappinessRank)) // Use HappinessRank
        .collect();

    chart.draw_series(PointSeries::of_element(
        points,
        5,
        &BLUE,
        &|c, s, st| Circle::new(c, s, st.filled()),
    ))?;

    root.present()?;
    Ok(())
}
