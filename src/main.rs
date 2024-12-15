// Final Project

mod load_clean;
mod eda_statistics;
mod graph;

// imports

use plotters::prelude::*;
use std::error::Error;
use std::io::Write;



fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "./Life Expectancy Data.csv";

    // Load CSV data
    let _data = load_clean::load_csv_to_array(file_path)?;
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

    eda_statistics::create_correlation_heatmap(file_path, output_file, &exclude_columns, &feature_names)?;

    // To 5 countries
    let country_column = 0;
    let year_column = 1;
    let life_expectancy_column = 3;
    let _income_comp = 20;

    eda_statistics::find_top_countries(file_path, country_column, year_column, life_expectancy_column)?;

    let income_comp_column = 20;
    let schooling_column = 21;

    eda_statistics::create_scatter_plot(file_path, "scatter_plot.png", income_comp_column, schooling_column)?;

    // graph
    let features = vec![3, 16, 17]; // Life expectancy, GDP, and Population columns
    let threshold = 0.8;

    let graph = graph::build_similarity_graph(file_path, &features, threshold)?;

    // visualize clusters
    let output_file = "graph_edge_list.csv";
    graph::export_graph_to_csv(&graph, output_file)?;

    println!("Edge list exported to {}", output_file);

    // Cluster the graph
    let k = 5; // Number of desired representatives
    let representatives = graph::cluster_graph(&graph, k);

    println!("Top {} representatives:", k);
    for (cluster_id, representative) in representatives {
        println!("Cluster {}: {}", cluster_id, representative);
    }


    // average life_expectancy vs status
    let status_column = 2; // Assuming column 2 indicates development status
    eda_statistics::calculate_average_life_expectancy(file_path, country_column, status_column, life_expectancy_column)?;

    // plot developed vs developing across Adult Mortality

    let output_file_feature = "developed_vs_developing_plot_adult_mortality.png"; // Output file for the plot

    // Specify the column indices
    let feature_column = 4; // The column index of Adult Mortality
    let year_column = 1; // The column index of the year
    let status_column = 2; // The column index indicating "Developed" or "Developing"

    // Call the function
    eda_statistics::create_developed_vs_developing_plot(
        file_path,
        output_file_feature,
        feature_column,
        year_column,
        status_column,
    )?;

    // plot developed vs developing across Infant Mortality

    let output_file_feature_two = "developed_vs_developing_plot_infant_mortality.png";
    let feature_column_two = 5; // The column index of Infant Mortality
    eda_statistics::create_developed_vs_developing_plot_infant(
        file_path,
        output_file_feature_two,
        feature_column_two,
        year_column,
        status_column,

    )?;

    let output_file = "comparison_bar_plot.png";
    let feature_columns = [4, 5, 7, 8, 9, 10, 11];
    let feature_names = ["Measles", "Polio", "BMI", "Diphtheria", "Hepatitis B", "HIV/AIDS"];

    eda_statistics::create_features_comparison_bar_plot(
        file_path,
        output_file,
        &feature_columns,
        year_column,
        status_column,
        &feature_names,
    )?;

    Ok(())
}
