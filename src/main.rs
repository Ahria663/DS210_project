mod models;
mod clean;
mod eda;
// mod year_clean;
mod year_eda;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_file = "Life Expectancy Data.csv";
    let output_file = "Cleaned_Life_Expectancy_Data.csv";

    if let Err(e) = clean::clean_data(input_file, output_file) {
        eprintln!("Error cleaning data: {}", e);
        return Ok(());
    }

    let records = clean::load_cleaned_data(output_file).expect("Failed to load cleaned data");
    if let Err(e) = eda::perform_eda(&records) {
        eprintln!("Error performing EDA: {}", e);
    }

    // let file_paths = vec!["2015.csv", "2016.csv", "2017.csv", "2019.csv"];

    // File paths
    let file_paths = vec!["./archive/2015.csv", "./archive/2016.csv"];

    // Process each file
    for path in file_paths {
        println!("Processing file: {}", path);

        let data = crate::year_eda::load_csv(path)?;
        let stats = crate::year_eda::calculate_statistics(&data);
        crate::year_eda::print_statistics(&stats);
        crate::year_eda::create_visualization(&data, "output_chart.png")?;
    }

    Ok(())
}



