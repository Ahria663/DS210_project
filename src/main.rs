use csv::Reader;
use ndarray::{Array2, Axis};
use plotters::prelude::*;
use std::error::Error;
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

// Custom wrapper for f64 to implement Ord
#[derive(Debug)]
struct FloatOrd(f64);

impl PartialEq for FloatOrd {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for FloatOrd {}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

fn load_csv_to_array(file_path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;

    // Collect data as a vector of rows
    let mut data = Vec::new();

    for record in reader.records() {
        let record = record?;
        let row: Vec<f64> = record
            .iter()
            .skip(3) // Skip the first three columns (e.g., Country, Region, Rank)
            .map(|value| value.parse::<f64>().unwrap_or(0.0))
            .collect();
        data.push(row);
    }

    // Convert to a 2D array
    let rows = data.len();
    let cols = data[0].len();
    let flat_data: Vec<f64> = data.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((rows, cols), flat_data)?)
}

fn calculate_correlation(x: &ndarray::ArrayView1<f64>, y: &ndarray::ArrayView1<f64>) -> Option<f64> {
    if x.len() != y.len() {
        return None;
    }

    let x_mean = x.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);

    let numerator: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean)).sum();
    let x_variance: f64 = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
    let y_variance: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    let denominator = (x_variance * y_variance).sqrt();

    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn find_top_countries(file_path: &str, happiness_index: usize) -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;

    // Read headers and ensure country column exists
    let headers = reader.headers()?.clone();
    let country_column = 0; // Assuming country is in the first column

    // Collect data with country names and happiness scores
    let mut country_scores = Vec::new();

    for record in reader.records() {
        let record = record?;
        let country = record.get(country_column).unwrap_or("").to_string();
        let happiness_score: f64 = record.get(happiness_index).unwrap_or("0").parse().unwrap_or(0.0);
        country_scores.push((FloatOrd(happiness_score), country));
    }

    // Find top 5 countries using a binary heap
    let mut heap = BinaryHeap::new();

    for (score, country) in country_scores {
        heap.push((score, country));
        if heap.len() > 5 {
            heap.pop();
        }
    }

    // Extract top 5 countries
    let mut top_countries = Vec::new();
    while let Some(entry) = heap.pop() {
        top_countries.push(entry);
    }

    top_countries.reverse(); // Reverse to show highest scores first
    println!("Top 5 countries in {}:", file_path);
    for (FloatOrd(score), country) in top_countries {
        println!("{}: {:.2}", country, score);
    }

    Ok(())
}

fn create_scatter_plot(
    file_2015: &str,
    file_2016: &str,
    trust_index: usize,
    happiness_index: usize,
    output_file: &str,
) -> Result<(), Box<dyn Error>> {
    let mut reader_2015 = Reader::from_path(file_2015)?;
    let mut reader_2016 = Reader::from_path(file_2016)?;

    let mut trust_scores_2015 = vec![];
    let mut happiness_scores_2015 = vec![];
    let mut trust_scores_2016 = vec![];
    let mut happiness_scores_2016 = vec![];

    for record in reader_2015.records() {
        let record = record?;
        let trust: f64 = record.get(trust_index).unwrap_or("0").parse().unwrap_or(0.0);
        let happiness: f64 = record.get(happiness_index).unwrap_or("0").parse().unwrap_or(0.0);
        trust_scores_2015.push(trust);
        happiness_scores_2015.push(happiness);
    }

    for record in reader_2016.records() {
        let record = record?;
        let trust: f64 = record.get(trust_index).unwrap_or("0").parse().unwrap_or(0.0);
        let happiness: f64 = record.get(happiness_index).unwrap_or("0").parse().unwrap_or(0.0);
        trust_scores_2016.push(trust);
        happiness_scores_2016.push(happiness);
    }

    let root = BitMapBackend::new(output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Trust in Government vs Happiness Score (2015 & 2016)", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..2.0, 0.0..8.0)?;

    chart.configure_mesh()
        .x_desc("Trust in Government")
        .y_desc("Happiness Score")
        .draw()?;

    chart.draw_series(
        trust_scores_2015.iter().zip(happiness_scores_2015.iter()).map(|(&trust, &happiness)| {
            Circle::new((trust, happiness), 5, BLUE.filled())
        }),
    )?.label("2015")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

    chart.draw_series(
        trust_scores_2016.iter().zip(happiness_scores_2016.iter()).map(|(&trust, &happiness)| {
            Circle::new((trust, happiness), 5, RED.filled())
        }),
    )?.label("2016")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).draw()?;

    println!("Scatter plot saved to {}", output_file);

    Ok(())
}


fn main() -> Result<(), Box<dyn Error>> {
    let file_2015 = "./archive/2015.csv";
    let file_2016 = "./archive/2016.csv";

    // Correct column indices (update based on inspection of CSV)
    let happiness_index = 3; // Assuming "Happiness Score" is in the 4th column (index 3)
    let life_expectancy_index = 5; // Assuming "Health (Life Expectancy)" is in the 6th column (index 5)

    // Load CSV data into 2D arrays
    let data_2015 = load_csv_to_array(file_2015)?;
    let data_2016 = load_csv_to_array(file_2016)?;

    // Analysis for 2015
    println!("Analysis for 2015 dataset:");
    let happiness_column = data_2015.column(happiness_index);
    let life_expectancy_column_2015 = data_2015.column(life_expectancy_index);
    let correlation_life_expectancy_2015 = calculate_correlation(&happiness_column, &life_expectancy_column_2015);
    println!(
        "Correlation between Happiness and Life Expectancy (2015): {:.2}",
        correlation_life_expectancy_2015.unwrap_or(0.0)
    );

    // Analysis for 2016
    println!("\nAnalysis for 2016 dataset:");
    let happiness_column_2016 = data_2016.column(happiness_index);
    let life_expectancy_column_2016 = data_2016.column(life_expectancy_index);
    let correlation_life_expectancy_2016 = calculate_correlation(&happiness_column_2016, &life_expectancy_column_2016);
    println!(
        "Correlation between Happiness and Life Expectancy (2016): {:.2}",
        correlation_life_expectancy_2016.unwrap_or(0.0)
    );

    // Find and print top 5 countries
    find_top_countries(file_2015, happiness_index)?;
    find_top_countries(file_2016, happiness_index)?;

    // Indices for Freedom and Trust in Government (adjust based on dataset structure)
    let freedom_index = 5; // Assuming "Freedom" is at index 5
    let trust_index = 6;   // Assuming "Trust in Government" is at index 6

    // Generate scatter plot
    create_scatter_plot(file_2015, file_2016, trust_index, happiness_index, "trust_vs_happiness.png")?;


    Ok(())
}
