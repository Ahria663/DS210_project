use std::error::Error;
use csv::Reader;
use ndarray::Array2;

// Load and Clean Data
pub(crate) fn load_csv_to_array(file_path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
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
        row.resize(max_cols, f64::NAN); // Fills in missing values
    }

    let rows = data.len();
    let cols = max_cols;
    let flat_data: Vec<f64> = data.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((rows, cols), flat_data)?)
}