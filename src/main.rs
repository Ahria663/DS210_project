use ordered_float::NotNan;
use std::cmp::Reverse;
use csv::Reader;
use ndarray::Array2;
use plotters::prelude::*;
use std::error::Error;
use std::collections::{BinaryHeap, HashMap};
use std::f64;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::Walker;
use plotters::prelude::full_palette::PURPLE;
use plotters::style::full_palette::ORANGE;
use plotters::prelude::*;

// Load and Clean Data
fn load_csv_to_array(file_path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;
    let mut data = Vec::new();

    let mut max_cols = 0;
    for record in reader.records() {
        let record = record?;
        let row: Vec<f64> = record
            .iter()
            .filter_map(|value| value.parse::<f64>().ok())
            .collect();
        max_cols = max_cols.max(row.len());
        data.push(row);
    }

    // Pad rows to the same length
    for row in &mut data {
        row.resize(max_cols, f64::NAN); // Fill with NaN for missing values
    }

    let rows = data.len();
    let cols = max_cols;
    let flat_data: Vec<f64> = data.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((rows, cols), flat_data)?)
}

// Statistics

// EDA
fn find_top_countries(file_path: &str, country_column: usize, year_column: usize, life_expectancy_column: usize) -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;

    let mut year_data: HashMap<String, BinaryHeap<Reverse<(NotNan<f64>, String)>>> = HashMap::new();

    for result in reader.records() {
        let record = result?;
        let country = record.get(country_column).unwrap_or("").to_string();
        let year = record.get(year_column).unwrap_or("").to_string();
        let life_expectancy = record
            .get(life_expectancy_column)
            .unwrap_or("0")
            .parse::<f64>()
            .ok()
            .and_then(|val| NotNan::new(val).ok())
            .unwrap_or_else(|| NotNan::new(0.0).unwrap());

        year_data
            .entry(year.clone())
            .or_insert_with(BinaryHeap::new)
            .push(Reverse((life_expectancy, country)));
    }

    for (year, mut heap) in year_data {
        println!("Top 5 countries in year {}:", year);
        let mut top_countries = Vec::new();
        while let Some(Reverse((life_expectancy, country))) = heap.pop() {
            top_countries.push((country, life_expectancy));
            if top_countries.len() == 5 {
                break;
            }
        }

        for (country, life_expectancy) in top_countries {
            println!("{}: {:.2}", country, life_expectancy);
        }
        println!();
    }

    Ok(())
}

// Calculate the correlation between two variables
fn create_correlation_heatmap(
    file_path: &str,
    output_file: &str,
    exclude_columns: &[usize], // Columns to exclude (e.g., Year, Country)
    feature_names: &[String],  // Names of all columns (for labeling the heatmap)
) -> Result<(), Box<dyn Error>> {
    // Load CSV data
    let mut reader = csv::Reader::from_path(file_path)?;
    // let headers = reader.headers()?.clone();

    // Parse the data into a matrix
    let mut data_matrix: Vec<Vec<f64>> = Vec::new();
    for record in reader.records() {
        let record = record?;
        let row: Vec<f64> = record
            .iter()
            .enumerate()
            .filter(|(i, _)| !exclude_columns.contains(i)) // Exclude specific columns
            .map(|(_, value)| value.parse::<f64>().unwrap_or(0.0)) // Parse as f64, default to 0.0
            .collect();
        data_matrix.push(row);
    }

    // Convert to ndarray for processing
    let data = Array2::from_shape_vec(
        (data_matrix.len(), data_matrix[0].len()),
        data_matrix.into_iter().flatten().collect(),
    )?;

    let (_, cols) = data.dim();
    if cols == 0 {
        return Err("No columns to process".into());
    }

    // Calculate the correlation matrix
    let mut correlation_matrix = Array2::zeros((cols, cols));
    for i in 0..cols {
        for j in 0..cols {
            let col_i = data.column(i);
            let col_j = data.column(j);
            let correlation = calculate_correlation(&col_i.view(), &col_j.view()).unwrap_or(0.0);
            correlation_matrix[(i, j)] = correlation;
        }
    }

    // Set up the drawing area
    let root = BitMapBackend::new(output_file, (1024, 1024)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Feature Correlation Heatmap", ("sans-serif", 30))
        .margin(5)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(0..cols as u32, 0..cols as u32)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(cols)
        .y_labels(cols)
        .x_desc("Features")
        .y_desc("Features")
        .label_style(("sans-serif", 15))
        .axis_desc_style(("sans-serif", 20))
        .draw()?;

    // Add labels for axes (feature names)
    chart.configure_mesh().x_label_formatter(&|x| {
        feature_names
            .get(*x as usize)
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string())
    });
    chart.configure_mesh().y_label_formatter(&|y| {
        feature_names
            .get(*y as usize)
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string())
    });

    // Draw heatmap rectangles
    for i in 0..cols {
        for j in 0..cols {
            let value = correlation_matrix[(i, j)];
            let color = if value >= 0.0 {
                RGBColor((255.0 * (1.0 - value)) as u8, (255.0 * value) as u8, 0)
            } else {
                RGBColor(0, (255.0 * (1.0 + value)) as u8, (255.0 * (-value)) as u8)
            };
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (j as u32, cols as u32 - i as u32 - 1),
                    ((j + 1) as u32, cols as u32 - i as u32),
                ],
                color.filled(),
            )))?;
        }
    }

    println!("Heatmap saved to {}", output_file);
    Ok(())
}

// Helper function to calculate correlation
fn calculate_correlation(x: &ndarray::ArrayView1<f64>, y: &ndarray::ArrayView1<f64>) -> Option<f64> {
    let x_mean = x.mean()?;
    let y_mean = y.mean()?;
    let numerator = x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean)).sum::<f64>();
    let denominator_x = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum::<f64>().sqrt();
    let denominator_y = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>().sqrt();
    if denominator_x > 0.0 && denominator_y > 0.0 {
        Some(numerator / (denominator_x * denominator_y))
    } else {
        None
    }
}

fn create_scatter_plot(file_path: &str, output_file: &str, income_comp_column: usize, schooling_column: usize) -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;

    let mut income = Vec::new();
    let mut schoolings = Vec::new();

    for result in reader.records() {
        let record = result?;

        if let (Some(income_value), Some(schooling)) = (
            record.get(income_comp_column),
            record.get(schooling_column),
        ) {
            if let (Ok(income_value), Ok(schooling)) = (
                income_value.parse::<f64>(),
                schooling.parse::<f64>(),
            ) {
                income.push(income_value);
                schoolings.push(schooling);
            }
        }
    }

    let root = BitMapBackend::new(output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Income vs. Schooling Rates", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0.0..income.iter().cloned().fold(f64::NAN, f64::max),
            0.0..schoolings.iter().cloned().fold(f64::NAN, f64::max),
        )?;

    chart.configure_mesh()
        .x_desc("Income")
        .y_desc("Schooling Rates")
        .draw()?;

    chart.draw_series(
        income.iter().zip(schoolings.iter()).map(|(&x, &y)| {
            Circle::new((x, y), 3, RGBAColor(190, 86, 131, 0.5).filled())
        }),
    )?;

    println!("Scatter plot saved to {}", output_file);
    Ok(())
}

// Graph Algorithm
fn build_similarity_graph(
    file_path: &str,
    features: &[usize],
    threshold: f64, // Similarity threshold
) -> Result<Graph<String, f64>, Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(file_path)?;

    let mut graph = Graph::<String, f64>::new();
    let mut nodes = Vec::new();
    let mut feature_data = Vec::new();

    for record in reader.records() {
        let record = record?;
        let country = record.get(0).unwrap_or("").to_string();
        nodes.push(country);
        let features_row: Vec<f64> = features
            .iter()
            .filter_map(|&idx| record.get(idx).and_then(|val| val.parse::<f64>().ok()))
            .collect();
        feature_data.push(features_row);
    }

    // Add nodes to the graph
    let node_indices: Vec<_> = nodes
        .iter()
        .map(|country| graph.add_node(country.clone()))
        .collect();

    // Calculate pairwise similarity and add edges
    for i in 0..feature_data.len() {
        for j in (i + 1)..feature_data.len() {
            let similarity = calculate_similarity(&feature_data[i], &feature_data[j]);
            if similarity >= threshold {
                graph.add_edge(node_indices[i], node_indices[j], similarity);
            }
        }
    }

    Ok(graph)
}

// Calculate similarity between two feature vectors
fn calculate_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
    let dot_product: f64 = vec1.iter().zip(vec2).map(|(x, y)| x * y).sum();
    let magnitude1: f64 = vec1.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let magnitude2: f64 = vec2.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

    if magnitude1 > 0.0 && magnitude2 > 0.0 {
        dot_product / (magnitude1 * magnitude2)
    } else {
        0.0
    }
}

// Perform graph clustering and identify representatives
fn cluster_graph(graph: &Graph<String, f64>, _k: usize) -> HashMap<usize, String> {
    use petgraph::unionfind::UnionFind;

    // Determine connected components
    let mut uf = UnionFind::new(graph.node_count());
    for edge in graph.edge_indices() {
        let (a, b) = graph.edge_endpoints(edge).unwrap();
        uf.union(a.index(), b.index());
    }

    // Map each component to its nodes
    let mut clusters: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
    for node in graph.node_indices() {
        let component_id = uf.find(node.index());
        clusters.entry(component_id).or_default().push(node);
    }

    // Select a representative for each cluster
    let mut representatives = HashMap::new();
    for (cluster_id, nodes) in clusters {
        if let Some(representative) = select_representative(&graph, &nodes) {
            representatives.insert(cluster_id, graph[representative].clone());
        }
    }

    representatives
}

// Select a representative node based on centrality
fn select_representative(
    graph: &Graph<String, f64>,
    nodes: &[NodeIndex],
) -> Option<NodeIndex> {
    nodes
        .iter()
        .max_by_key(|&&node| graph.edges(node).count())
        .cloned()
}

// Calculate average life expectancy developing vs developed countries
fn calculate_average_life_expectancy(
    file_path: &str,
    country_column: usize,
    status_column: usize,
    life_expectancy_column: usize,
) -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;

    let mut totals: HashMap<String, (f64, usize)> = HashMap::new();

    for result in reader.records() {
        let record = result?;
        let country_status = record.get(status_column).unwrap_or("").to_string();
        let life_expectancy = record
            .get(life_expectancy_column)
            .unwrap_or("0")
            .parse::<f64>()
            .unwrap_or(0.0);

        if !country_status.is_empty() {
            let entry = totals.entry(country_status).or_insert((0.0, 0));
            entry.0 += life_expectancy;
            entry.1 += 1;
        }
    }

    for (status, (total_life_expectancy, count)) in totals {
        let average = total_life_expectancy / count as f64;
        println!(
            "Average life expectancy for {} countries: {:.2}",
            status, average
        );
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "./Life Expectancy Data.csv";

    // Load CSV data
    let _data = load_csv_to_array(file_path)?;
    let output_file = "correlation_heatmap.png";
    let exclude_columns = [0, 1]; // Adjust based on your CSV structure

    // Specify the feature names for labeling
    let feature_names: Vec<String> = vec![
        "Adult Mortality", "Infant Deaths", "Alcohol", "Percentage Expenditure",
        "Measles", "BMI", "Under-Five Deaths", "Polio", "Total Expenditure",
        "Diphtheria", "Hepatitis B", "HIV/AIDS", "GDP", "Population", "Schooling",
    ]
        .into_iter()
        .map(String::from)
        .collect();

    create_correlation_heatmap(file_path, output_file, &exclude_columns, &feature_names)?;

    // To 5 countries
    let country_column = 0;
    let year_column = 1;
    let life_expectancy_column = 3;
    let _income_comp = 20;

    find_top_countries(file_path, country_column, year_column, life_expectancy_column)?;

    let income_comp_column = 20;
    let schooling_column = 21;

    create_scatter_plot(file_path, "scatter_plot.png", income_comp_column, schooling_column)?;

    // graph
    let features = vec![3, 16, 17]; // Life expectancy, GDP, and Population columns
    let threshold = 0.8;

    let graph = build_similarity_graph(file_path, &features, threshold)?;

    // Cluster the graph
    let k = 5; // Number of desired representatives
    let representatives = cluster_graph(&graph, k);

    println!("Top {} representatives:", k);
    for (cluster_id, representative) in representatives {
        println!("Cluster {}: {}", cluster_id, representative);
    }


    // average life_expectancy vs status
    let status_column = 2; // Assuming column 2 indicates development status
    calculate_average_life_expectancy(file_path, country_column, status_column, life_expectancy_column)?;

    Ok(())
}
